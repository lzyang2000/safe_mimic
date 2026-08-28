"""GPU-resident multi-motion library for mjlab tracker NPZ files.

The loader allocates the final packed tensors exactly once.  Each compressed
NPZ is decompressed on the CPU, its tracked bodies are selected there, and the
result is copied directly into its final slice.  In particular, it never
materializes a second, concatenated copy of the motion bank on the GPU.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from safe_mimic.motions.g1_tracker_npz import TRACKER_NPZ_FIELDS


@dataclass(frozen=True)
class PackedMotionSource:
  """One manifest entry resolved to an NPZ on disk."""

  path: Path
  weight: float
  frame_count: int | None
  fps: float | None
  metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PackedMotionFrame:
  """Interpolated tracker state for a batch of motion/time pairs."""

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor


def _resolve_path(path: str | Path, root: Path) -> Path:
  candidate = Path(path).expanduser()
  if not candidate.is_absolute():
    candidate = root / candidate
  return candidate.resolve()


def _jsonl_npz_path(
  record: Mapping[str, Any], manifest_path: Path, npz_root: Path | None
) -> Path:
  root = npz_root or manifest_path.parent / "npz_50hz"
  if record.get("npz_path"):
    return _resolve_path(str(record["npz_path"]), manifest_path.parent)
  if record.get("file"):
    return _resolve_path(str(record["file"]), root)
  if not record.get("csv_path"):
    raise ValueError("JSONL motion entry needs npz_path, file, or csv_path")

  relative = Path(str(record["csv_path"]))
  if relative.parts[:2] == ("g1", "csv"):
    relative = Path(*relative.parts[2:])
  return _resolve_path(relative.with_suffix(".npz"), root)


def load_packed_npz_manifest(
  manifest_file: str | Path,
  *,
  npz_root: str | Path | None = None,
  splits: str | Collection[str] | None = None,
) -> tuple[PackedMotionSource, ...]:
  """Resolve a conversion JSONL, dataset YAML, or single NPZ.

  JSONL conversion manifests may omit ``npz_path``; in that case the standard
  sibling ``npz_50hz`` directory is inferred from ``csv_path``.  All original
  metadata is retained so curricula can distinguish WBC/dance and
  original/mirror entries without a second index.
  """
  manifest_path = Path(manifest_file).expanduser().resolve()
  if isinstance(splits, str):
    wanted_splits = {splits}
  elif splits is None:
    wanted_splits = None
  else:
    wanted_splits = set(splits)
  root_override = (
    Path(npz_root).expanduser().resolve() if npz_root is not None else None
  )

  if manifest_path.suffix == ".npz":
    return (
      PackedMotionSource(
        path=manifest_path,
        weight=1.0,
        frame_count=None,
        fps=None,
        metadata={},
      ),
    )

  raw_records: list[dict[str, Any]]
  default_fps: float | None = None
  yaml_root: Path | None = None
  if manifest_path.suffix in {".yaml", ".yml"}:
    with manifest_path.open(encoding="utf-8") as stream:
      config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("motions"), list):
      raise ValueError("motion YAML must contain a motions list")
    default_fps = float(config["fps"]) if config.get("fps") is not None else None
    configured_root = root_override or Path(str(config.get("root_path", ".")))
    yaml_root = _resolve_path(configured_root, manifest_path.parent)
    raw_records = [dict(record) for record in config["motions"]]
  elif manifest_path.suffix == ".jsonl":
    with manifest_path.open(encoding="utf-8") as stream:
      raw_records = [json.loads(line) for line in stream if line.strip()]
  else:
    raise ValueError("motion input must be an NPZ, JSONL, YAML, or YML file")

  sources: list[PackedMotionSource] = []
  for record in raw_records:
    split = record.get("split")
    if wanted_splits is not None and split not in wanted_splits:
      continue
    if yaml_root is not None:
      if not record.get("file"):
        raise ValueError("each YAML motion entry must have a file")
      path = _resolve_path(str(record["file"]), yaml_root)
    else:
      path = _jsonl_npz_path(record, manifest_path, root_override)
    weight = float(record.get("weight", 1.0))
    if not np.isfinite(weight) or weight <= 0.0:
      raise ValueError(f"motion weight must be positive and finite: {weight}")
    frame_count_value = record.get("tracker_frame_count", record.get("frame_count"))
    frame_count = int(frame_count_value) if frame_count_value is not None else None
    fps_value = record.get("fps", record.get("target_fps", default_fps))
    fps = float(fps_value) if fps_value is not None else None
    sources.append(
      PackedMotionSource(
        path=path,
        weight=weight,
        frame_count=frame_count,
        fps=fps,
        metadata=record,
      )
    )
  if not sources:
    raise ValueError("motion manifest selection is empty")
  return tuple(sources)


def _npz_array_shape(path: Path, field: str) -> tuple[int, ...]:
  """Read an NPY member shape without decompressing its tensor payload."""
  member_name = f"{field}.npy"
  with zipfile.ZipFile(path) as archive:
    try:
      with archive.open(member_name) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
          shape, _, _ = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
          shape, _, _ = np.lib.format.read_array_header_2_0(stream)
        else:
          shape, _, _ = np.lib.format._read_array_header(stream, version)  # noqa: SLF001
    except KeyError as exc:
      raise ValueError(f"{path} is missing {field}") from exc
  return tuple(int(value) for value in shape)


def _slerp_wxyz(
  quaternion_0: torch.Tensor,
  quaternion_1: torch.Tensor,
  blend: torch.Tensor,
) -> torch.Tensor:
  """Shortest-path quaternion interpolation with a stable linear limit."""
  while blend.ndim < quaternion_0.ndim:
    blend = blend.unsqueeze(-1)
  dot = (quaternion_0 * quaternion_1).sum(dim=-1, keepdim=True)
  quaternion_1 = torch.where(dot < 0.0, -quaternion_1, quaternion_1)
  dot = dot.abs().clamp(max=1.0)

  linear = dot > 0.9995
  theta = torch.acos(dot.clamp(max=1.0 - 1e-7))
  sin_theta = torch.sin(theta).clamp(min=1e-7)
  scale_0 = torch.sin((1.0 - blend) * theta) / sin_theta
  scale_1 = torch.sin(blend * theta) / sin_theta
  spherical = scale_0 * quaternion_0 + scale_1 * quaternion_1
  interpolated = torch.where(
    linear,
    (1.0 - blend) * quaternion_0 + blend * quaternion_1,
    spherical,
  )
  return torch.nn.functional.normalize(interpolated, dim=-1)


class PackedNpzMotionLib:
  """Pack tracker NPZ clips into final CPU or CUDA tensors without ``cat``.

  Args:
    motion_file: JSONL/YAML manifest or one tracker NPZ.
    body_indexes: Body-axis indexes to retain from the 30-body G1 arrays.
    device: Final resident tensor device.
    npz_root: Optional override for the manifest's NPZ root.
    splits: Optional split name or names to retain.
    default_fps: Used only when neither manifest nor NPZ supplies an FPS.
    verbose: Show startup progress bars.
  """

  def __init__(
    self,
    motion_file: str | Path,
    body_indexes: torch.Tensor | Sequence[int],
    device: str | torch.device = "cpu",
    *,
    npz_root: str | Path | None = None,
    splits: str | Collection[str] | None = None,
    default_fps: float = 50.0,
    verbose: bool = True,
  ) -> None:
    self._device = torch.device(device)
    self.sources = load_packed_npz_manifest(
      motion_file, npz_root=npz_root, splits=splits
    )
    indexes = torch.as_tensor(body_indexes, dtype=torch.long, device="cpu")
    if indexes.ndim != 1 or indexes.numel() == 0:
      raise ValueError("body_indexes must be a non-empty one-dimensional sequence")
    if torch.any(indexes < 0) or indexes.unique().numel() != indexes.numel():
      raise ValueError("body_indexes must be unique and non-negative")
    self._body_indexes_numpy = indexes.numpy()
    self._tracked_body_count = indexes.numel()
    self._default_fps = float(default_fps)
    if not np.isfinite(self._default_fps) or self._default_fps <= 0.0:
      raise ValueError("default_fps must be positive and finite")

    frame_counts = self._resolve_frame_counts(verbose)
    first_joint_shape = _npz_array_shape(self.sources[0].path, "joint_pos")
    if len(first_joint_shape) != 2:
      raise ValueError("joint_pos must have shape [frames, joints]")
    self._joint_count = first_joint_shape[1]
    total_frames = int(sum(frame_counts))
    self._allocate(total_frames)
    self._load_into_final_slices(frame_counts, verbose)

  def _resolve_frame_counts(self, verbose: bool) -> list[int]:
    frame_counts: list[int] = []
    missing = sum(source.frame_count is None for source in self.sources)
    if verbose and missing:
      print(
        f"[PackedNpzMotionLib] Reading {missing:,} NPZ headers for frame counts"
      )
    for source in self.sources:
      if not source.path.is_file():
        raise FileNotFoundError(source.path)
      frame_count = source.frame_count
      if frame_count is None:
        shape = _npz_array_shape(source.path, "joint_pos")
        if len(shape) != 2:
          raise ValueError(f"{source.path}: joint_pos must be two-dimensional")
        frame_count = shape[0]
      if frame_count < 2:
        raise ValueError(f"{source.path}: motions need at least two frames")
      frame_counts.append(frame_count)
    return frame_counts

  def _allocate(self, total_frames: int) -> None:
    bodies = self._tracked_body_count
    kwargs = {"dtype": torch.float32, "device": self._device}
    self._all_joint_pos = torch.empty((total_frames, self._joint_count), **kwargs)
    self._all_joint_vel = torch.empty((total_frames, self._joint_count), **kwargs)
    self._all_body_pos_w = torch.empty((total_frames, bodies, 3), **kwargs)
    self._all_body_quat_w = torch.empty((total_frames, bodies, 4), **kwargs)
    self._all_body_lin_vel_w = torch.empty((total_frames, bodies, 3), **kwargs)
    self._all_body_ang_vel_w = torch.empty((total_frames, bodies, 3), **kwargs)

  @staticmethod
  def _copy_array(target: torch.Tensor, array: np.ndarray) -> None:
    source = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    target.copy_(source)

  def _load_into_final_slices(
    self, frame_counts: Sequence[int], verbose: bool
  ) -> None:
    starts = np.cumsum(np.asarray([0, *frame_counts[:-1]], dtype=np.int64))
    fps_values: list[float] = []
    motion_count = len(self.sources)
    for source, frame_count, start in tqdm(
      zip(self.sources, frame_counts, starts, strict=True),
      total=motion_count,
      desc="[PackedNpzMotionLib] Loading motions",
      unit="motion",
      dynamic_ncols=True,
      disable=not verbose,
    ):
      with np.load(source.path) as data:
        missing = set(TRACKER_NPZ_FIELDS) - set(data.files)
        if missing:
          raise ValueError(f"{source.path}: missing fields {sorted(missing)}")
        joint_pos = np.asarray(data["joint_pos"])
        joint_vel = np.asarray(data["joint_vel"])
        body_pos = np.asarray(data["body_pos_w"])
        body_quat = np.asarray(data["body_quat_w"])
        body_lin_vel = np.asarray(data["body_lin_vel_w"])
        body_ang_vel = np.asarray(data["body_ang_vel_w"])
        fps_array = np.asarray(data["fps"]).reshape(-1)

      expected_joint = (frame_count, self._joint_count)
      if joint_pos.shape != expected_joint or joint_vel.shape != expected_joint:
        raise ValueError(
          f"{source.path}: joint arrays must both have shape {expected_joint}"
        )
      body_count = body_pos.shape[1] if body_pos.ndim == 3 else -1
      if self._body_indexes_numpy.max(initial=-1) >= body_count:
        raise ValueError(
          f"{source.path}: body index exceeds available count {body_count}"
        )
      expected_body_shapes = {
        "body_pos_w": (frame_count, body_count, 3),
        "body_quat_w": (frame_count, body_count, 4),
        "body_lin_vel_w": (frame_count, body_count, 3),
        "body_ang_vel_w": (frame_count, body_count, 3),
      }
      actual_body_shapes = {
        "body_pos_w": body_pos.shape,
        "body_quat_w": body_quat.shape,
        "body_lin_vel_w": body_lin_vel.shape,
        "body_ang_vel_w": body_ang_vel.shape,
      }
      if actual_body_shapes != expected_body_shapes:
        raise ValueError(
          f"{source.path}: body shapes {actual_body_shapes}, "
          f"expected {expected_body_shapes}"
        )
      if fps_array.size != 1:
        raise ValueError(f"{source.path}: fps must contain one value")
      file_fps = float(fps_array[0])
      fps = source.fps if source.fps is not None else file_fps
      if not np.isfinite(fps) or fps <= 0.0:
        fps = self._default_fps
      if not np.isclose(file_fps, fps, rtol=1e-5, atol=1e-6):
        raise ValueError(
          f"{source.path}: manifest FPS {fps:g} disagrees with NPZ {file_fps:g}"
        )
      fps_values.append(fps)

      stop = int(start) + frame_count
      target_slice = slice(int(start), stop)
      self._copy_array(self._all_joint_pos[target_slice], joint_pos)
      self._copy_array(self._all_joint_vel[target_slice], joint_vel)
      self._copy_array(
        self._all_body_pos_w[target_slice],
        np.take(body_pos, self._body_indexes_numpy, axis=1),
      )
      self._copy_array(
        self._all_body_quat_w[target_slice],
        np.take(body_quat, self._body_indexes_numpy, axis=1),
      )
      self._copy_array(
        self._all_body_lin_vel_w[target_slice],
        np.take(body_lin_vel, self._body_indexes_numpy, axis=1),
      )
      self._copy_array(
        self._all_body_ang_vel_w[target_slice],
        np.take(body_ang_vel, self._body_indexes_numpy, axis=1),
      )
    metadata_kwargs = {"device": self._device}
    self._motion_num_frames = torch.tensor(
      frame_counts, dtype=torch.long, **metadata_kwargs
    )
    self._motion_start_idx = torch.tensor(
      starts, dtype=torch.long, **metadata_kwargs
    )
    self._motion_fps = torch.tensor(
      fps_values, dtype=torch.float32, **metadata_kwargs
    )
    self._motion_lengths = (
      (self._motion_num_frames - 1).float() / self._motion_fps
    )
    weights = torch.tensor(
      [source.weight for source in self.sources],
      dtype=torch.float32,
      **metadata_kwargs,
    )
    self._motion_weights = weights / weights.sum()

  @property
  def device(self) -> torch.device:
    return self._device

  @property
  def total_frames(self) -> int:
    return self._all_joint_pos.shape[0]

  @property
  def resident_bytes(self) -> int:
    """Bytes occupied by packed frame tensors, excluding tiny indexes."""
    tensors = (
      self._all_joint_pos,
      self._all_joint_vel,
      self._all_body_pos_w,
      self._all_body_quat_w,
      self._all_body_lin_vel_w,
      self._all_body_ang_vel_w,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

  @property
  def motion_num_frames(self) -> torch.Tensor:
    return self._motion_num_frames

  @property
  def motion_start_idx(self) -> torch.Tensor:
    return self._motion_start_idx

  @property
  def motion_weights(self) -> torch.Tensor:
    return self._motion_weights

  @property
  def motion_fps(self) -> torch.Tensor:
    return self._motion_fps

  @property
  def motion_lengths(self) -> torch.Tensor:
    return self._motion_lengths

  def num_motions(self) -> int:
    return len(self.sources)

  def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
    motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self._device)
    return self._motion_lengths[motion_ids]

  def sample_motions(
    self, count: int, *, generator: torch.Generator | None = None
  ) -> torch.Tensor:
    """Sample motion IDs with replacement according to manifest weights."""
    if count < 0:
      raise ValueError("count must be non-negative")
    return torch.multinomial(
      self._motion_weights,
      num_samples=count,
      replacement=True,
      generator=generator,
    )

  def sample_time(
    self,
    motion_ids: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
  ) -> torch.Tensor:
    """Sample an independent continuous phase within every selected clip."""
    motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self._device)
    phase = torch.rand(
      motion_ids.shape, device=self._device, generator=generator
    )
    return phase * self._motion_lengths[motion_ids]

  def get_frame(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> PackedMotionFrame:
    """Interpolate frames without allowing either index to leave its clip."""
    motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self._device)
    motion_times = torch.as_tensor(
      motion_times, dtype=torch.float32, device=self._device
    )
    if motion_ids.shape != motion_times.shape:
      raise ValueError("motion_ids and motion_times must have the same shape")

    lengths = self._motion_lengths[motion_ids]
    num_frames = self._motion_num_frames[motion_ids]
    starts = self._motion_start_idx[motion_ids]
    fps = self._motion_fps[motion_ids]
    wrapped_times = torch.remainder(motion_times, lengths.clamp(min=1e-6))
    fractional_index = (wrapped_times * fps).clamp(min=0.0)
    index_0_local = torch.minimum(fractional_index.long(), num_frames - 1)
    index_1_local = torch.minimum(index_0_local + 1, num_frames - 1)
    blend = fractional_index - index_0_local.float()
    index_0 = starts + index_0_local
    index_1 = starts + index_1_local

    joint_blend = blend.unsqueeze(-1)
    body_blend = joint_blend.unsqueeze(-1)
    joint_pos = torch.lerp(
      self._all_joint_pos[index_0], self._all_joint_pos[index_1], joint_blend
    )
    joint_vel = torch.lerp(
      self._all_joint_vel[index_0], self._all_joint_vel[index_1], joint_blend
    )
    body_pos = torch.lerp(
      self._all_body_pos_w[index_0], self._all_body_pos_w[index_1], body_blend
    )
    body_quat = _slerp_wxyz(
      self._all_body_quat_w[index_0],
      self._all_body_quat_w[index_1],
      blend,
    )
    body_lin_vel = torch.lerp(
      self._all_body_lin_vel_w[index_0],
      self._all_body_lin_vel_w[index_1],
      body_blend,
    )
    body_ang_vel = torch.lerp(
      self._all_body_ang_vel_w[index_0],
      self._all_body_ang_vel_w[index_1],
      body_blend,
    )
    return PackedMotionFrame(
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      body_lin_vel_w=body_lin_vel,
      body_ang_vel_w=body_ang_vel,
    )


__all__ = [
  "PackedMotionFrame",
  "PackedMotionSource",
  "PackedNpzMotionLib",
  "load_packed_npz_manifest",
]
