"""Curation helpers for the BONES-SEED Unitree G1 motion library."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

GROUND_MOTION_TOKENS = (
  "cartwheel",
  "crawl",
  "crutch",
  "handstand",
  "injured",
  "kneel",
  "ladder",
  "lying",
  "on all fours",
  "on_all_fours",
  "seated",
  "sitting",
)

_JUMP_TEXT_FIELDS = (
  "move_name",
  "description",
  "movement",
  "take_name",
  "category",
  "content_natural_desc_1",
  "content_type_of_movement",
)
_EXPLICIT_STATIONARY_JUMP_PHRASES = ("in place", "one place")


@dataclass(frozen=True)
class G1MotionFilterCfg:
  """Thresholds for a conservative upright G1 imitation library."""

  native_fps: float = 120.0
  target_fps: float = 30.0
  min_duration_s: float = 0.5
  min_root_height_m: float = 0.40
  max_root_height_m: float = 0.95
  expressive_min_root_height_m: float = 0.35
  expressive_max_root_height_m: float = 1.20
  max_root_tilt_deg: float = 100.0
  max_root_speed_mps: float = 5.0
  max_joint_speed_rps: float = 35.0
  joint_limit_tolerance_rad: float = 0.02


@dataclass(frozen=True)
class G1MotionQuality:
  """Kinematic quality statistics measured from one retargeted G1 CSV."""

  frame_count: int
  duration_s: float
  root_height_min_m: float
  root_height_max_m: float
  root_tilt_max_deg: float
  root_speed_max_mps: float
  joint_speed_max_rps: float
  joint_limit_violation_max_rad: float


@dataclass(frozen=True)
class G1MotionFilterResult:
  """Result of metadata and kinematic filtering for one motion."""

  accepted: bool
  reason: str | None
  quality: G1MotionQuality | None


def parse_metadata_bool(value: object) -> bool:
  """Interpret the bool spellings used by BONES-SEED metadata."""
  return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def is_dance_motion(metadata: Mapping[str, object]) -> bool:
  """Return whether metadata identifies a dance or dance-like clip."""
  fields = (
    metadata.get("package", ""),
    metadata.get("category", ""),
    metadata.get("content_type_of_movement", ""),
    metadata.get("content_uniform_style", ""),
  )
  return "danc" in " ".join(str(value) for value in fields).lower()


def is_jump_like_motion(metadata: Mapping[str, object]) -> bool:
  """Recognize jump, hop, and leap actions without matching the hip-hop genre."""
  text = " ".join(str(metadata.get(field, "")) for field in _JUMP_TEXT_FIELDS)
  text = re.sub(r"hip[- ]hop", "", text.lower())
  words = re.findall(r"[a-z]+", text)
  return any(word.startswith(("jump", "hop", "leap")) for word in words)


def is_stationary_jump_motion(
  metadata: Mapping[str, object],
  root_pos_w: np.ndarray,
  *,
  max_planar_excursion_m: float = 0.35,
) -> bool:
  """Identify jump-like clips that remain near their initial XY position."""
  if max_planar_excursion_m < 0.0:
    raise ValueError("max_planar_excursion_m must be non-negative")
  if root_pos_w.ndim != 2 or root_pos_w.shape[0] == 0 or root_pos_w.shape[1] < 2:
    raise ValueError("root_pos_w must have shape [frames, >=2]")
  if not is_jump_like_motion(metadata):
    return False

  text = " ".join(str(metadata.get(field, "")) for field in _JUMP_TEXT_FIELDS).lower()
  if any(phrase in text for phrase in _EXPLICIT_STATIONARY_JUMP_PHRASES):
    return True
  planar_offset = root_pos_w[:, :2] - root_pos_w[0, :2]
  max_excursion = float(np.linalg.norm(planar_offset, axis=1).max())
  return max_excursion <= max_planar_excursion_m


def is_expressive_height_motion(metadata: Mapping[str, object]) -> bool:
  """Allow a wider root-height envelope for dance and jumping clips."""
  fields = (
    metadata.get("package", ""),
    metadata.get("category", ""),
    metadata.get("content_type_of_movement", ""),
  )
  text = " ".join(str(value) for value in fields).lower()
  return is_dance_motion(metadata) or "jump" in text


def metadata_rejection_reason(metadata: Mapping[str, object]) -> str | None:
  """Reject duplicate mirrors and clearly non-upright G1 motions.

  Dances are intentionally not rejected. Mirroring is deferred to online
  augmentation so the GPU motion bank does not store duplicate trajectories.
  """
  if parse_metadata_bool(metadata.get("is_mirror", False)):
    return "mirrored duplicate (mirror online)"

  package = str(metadata.get("package", "")).strip().lower()
  category = str(metadata.get("category", "")).strip().lower()
  if package == "stunts" or category == "stunts":
    return "stunt category"

  fields = (
    metadata.get("move_name", ""),
    metadata.get("content_body_position", ""),
    metadata.get("content_type_of_movement", ""),
    category,
  )
  text = " ".join(str(value) for value in fields).lower().replace("-", " ")
  for token in GROUND_MOTION_TOKENS:
    if token in text:
      return f"non-upright metadata: {token}"
  return None


def load_g1_csv(csv_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Load root position, root Euler angles, and 29 joint angles from CSV."""
  raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
  raw = np.atleast_2d(raw)
  if raw.shape[1] != 36:
    raise ValueError(f"expected 36 CSV columns, found {raw.shape[1]}")
  if not np.isfinite(raw).all():
    raise ValueError("motion contains non-finite values")
  root_pos_m = raw[:, 1:4] / 100.0
  root_euler_deg = raw[:, 4:7]
  joint_pos_rad = np.deg2rad(raw[:, 7:36])
  return root_pos_m, root_euler_deg, joint_pos_rad


def _sample_indices(
  frame_count: int, native_fps: float, target_fps: float
) -> np.ndarray:
  duration_s = (frame_count - 1) / native_fps
  target_count = max(2, int(round(duration_s * target_fps)) + 1)
  return np.round(np.linspace(0, frame_count - 1, target_count)).astype(np.int64)


def measure_g1_motion_quality(
  root_pos_m: np.ndarray,
  root_euler_deg: np.ndarray,
  joint_pos_rad: np.ndarray,
  joint_lower_rad: np.ndarray,
  joint_upper_rad: np.ndarray,
  cfg: G1MotionFilterCfg,
) -> G1MotionQuality:
  """Measure quality after downsampling to the intended training rate."""
  frame_count = root_pos_m.shape[0]
  if frame_count < 2:
    raise ValueError("motion must contain at least two frames")
  if root_euler_deg.shape != (frame_count, 3):
    raise ValueError("root Euler angles must have shape [frames, 3]")
  if joint_pos_rad.shape != (frame_count, 29):
    raise ValueError("joint positions must have shape [frames, 29]")
  if joint_lower_rad.shape != (29,) or joint_upper_rad.shape != (29,):
    raise ValueError("joint limits must each have shape [29]")

  indices = _sample_indices(frame_count, cfg.native_fps, cfg.target_fps)
  root_pos = root_pos_m[indices]
  root_euler = root_euler_deg[indices]
  joint_pos = joint_pos_rad[indices]
  sample_times = indices.astype(np.float64) / cfg.native_fps
  dt = np.diff(sample_times).clip(min=1e-6)

  root_speed = np.linalg.norm(np.diff(root_pos[:, :2], axis=0) / dt[:, None], axis=1)
  joint_speed = np.abs(np.diff(joint_pos, axis=0) / dt[:, None])
  rotations = Rotation.from_euler("xyz", root_euler, degrees=True)
  up_world = rotations.apply(np.array([0.0, 0.0, 1.0]))
  tilt_deg = np.rad2deg(np.arccos(np.clip(up_world[:, 2], -1.0, 1.0)))

  below = np.maximum(joint_lower_rad[None] - joint_pos, 0.0)
  above = np.maximum(joint_pos - joint_upper_rad[None], 0.0)
  joint_limit_violation = np.maximum(below, above)

  return G1MotionQuality(
    frame_count=frame_count,
    duration_s=(frame_count - 1) / cfg.native_fps,
    root_height_min_m=float(root_pos[:, 2].min()),
    root_height_max_m=float(root_pos[:, 2].max()),
    root_tilt_max_deg=float(tilt_deg.max()),
    root_speed_max_mps=float(root_speed.max(initial=0.0)),
    joint_speed_max_rps=float(joint_speed.max(initial=0.0)),
    joint_limit_violation_max_rad=float(joint_limit_violation.max(initial=0.0)),
  )


def quality_rejection_reason(
  quality: G1MotionQuality,
  metadata: Mapping[str, object],
  cfg: G1MotionFilterCfg,
) -> str | None:
  """Return the first deterministic kinematic rejection reason."""
  if quality.duration_s < cfg.min_duration_s:
    return f"duration {quality.duration_s:.3f}s below {cfg.min_duration_s:.3f}s"

  expressive = is_expressive_height_motion(metadata)
  min_height = (
    cfg.expressive_min_root_height_m if expressive else cfg.min_root_height_m
  )
  max_height = (
    cfg.expressive_max_root_height_m if expressive else cfg.max_root_height_m
  )
  if quality.root_height_min_m < min_height:
    return f"root height {quality.root_height_min_m:.3f}m below {min_height:.3f}m"
  if quality.root_height_max_m > max_height:
    return f"root height {quality.root_height_max_m:.3f}m above {max_height:.3f}m"
  if quality.root_tilt_max_deg > cfg.max_root_tilt_deg:
    return (
      f"root tilt {quality.root_tilt_max_deg:.1f}deg above "
      f"{cfg.max_root_tilt_deg:.1f}deg"
    )
  if quality.root_speed_max_mps > cfg.max_root_speed_mps:
    return (
      f"root speed {quality.root_speed_max_mps:.2f}m/s above "
      f"{cfg.max_root_speed_mps:.2f}m/s"
    )
  if quality.joint_speed_max_rps > cfg.max_joint_speed_rps:
    return (
      f"joint speed {quality.joint_speed_max_rps:.2f}rad/s above "
      f"{cfg.max_joint_speed_rps:.2f}rad/s"
    )
  if quality.joint_limit_violation_max_rad > cfg.joint_limit_tolerance_rad:
    return (
      f"joint-limit violation {quality.joint_limit_violation_max_rad:.4f}rad above "
      f"{cfg.joint_limit_tolerance_rad:.4f}rad"
    )
  return None


def filter_g1_motion(
  csv_path: Path | str,
  metadata: Mapping[str, object],
  joint_lower_rad: np.ndarray,
  joint_upper_rad: np.ndarray,
  cfg: G1MotionFilterCfg,
) -> G1MotionFilterResult:
  """Apply metadata and kinematic filters to one BONES-SEED G1 motion."""
  reason = metadata_rejection_reason(metadata)
  if reason is not None:
    return G1MotionFilterResult(False, reason, None)
  try:
    arrays = load_g1_csv(csv_path)
    quality = measure_g1_motion_quality(
      *arrays, joint_lower_rad, joint_upper_rad, cfg
    )
  except Exception as exc:  # noqa: BLE001 - preserve per-file failure in manifest
    return G1MotionFilterResult(False, f"load/format error: {exc}", None)
  reason = quality_rejection_reason(quality, metadata, cfg)
  return G1MotionFilterResult(reason is None, reason, quality)


def stable_fraction(key: str) -> float:
  """Map a stable identifier to [0, 1) without depending on Python hash seed."""
  digest = hashlib.sha256(key.encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_split(metadata: Mapping[str, object], validation_fraction: float) -> str:
  """Split by source take so related clips cannot leak across partitions."""
  key = str(metadata.get("take_name") or metadata.get("move_name") or "")
  return "validation" if stable_fraction(key) < validation_fraction else "train"


def balanced_package_sample(
  records: Sequence[Mapping[str, object]],
  max_per_package: int,
) -> list[Mapping[str, object]]:
  """Select a deterministic category-round-robin subset per top-level package."""
  if max_per_package < 1:
    raise ValueError("max_per_package must be positive")
  by_package: dict[str, dict[str, list[Mapping[str, object]]]] = {}
  for record in records:
    package = str(record.get("package", "Unknown"))
    category = str(record.get("category", "Unknown"))
    by_package.setdefault(package, {}).setdefault(category, []).append(record)

  selected: list[Mapping[str, object]] = []
  for categories in by_package.values():
    for category_records in categories.values():
      category_records.sort(
        key=lambda row: stable_fraction(str(row.get("move_name", "")))
      )
    category_names = sorted(categories)
    category_offsets = {name: 0 for name in category_names}
    package_selected: list[Mapping[str, object]] = []
    while len(package_selected) < max_per_package:
      added = False
      for name in category_names:
        offset = category_offsets[name]
        if offset >= len(categories[name]):
          continue
        package_selected.append(categories[name][offset])
        category_offsets[name] += 1
        added = True
        if len(package_selected) >= max_per_package:
          break
      if not added:
        break
    selected.extend(package_selected)
  return selected


def quality_as_dict(quality: G1MotionQuality | None) -> dict[str, object]:
  """Serialize optional quality metrics for JSON manifests."""
  return {} if quality is None else asdict(quality)


def load_metadata_rows(path: Path | str) -> Iterable[dict[str, str]]:
  """Stream BONES-SEED metadata without a pandas dependency."""
  with Path(path).open(newline="", encoding="utf-8") as stream:
    yield from csv.DictReader(stream)


__all__ = [
  "G1MotionFilterCfg",
  "G1MotionFilterResult",
  "G1MotionQuality",
  "assign_split",
  "balanced_package_sample",
  "filter_g1_motion",
  "is_dance_motion",
  "load_g1_csv",
  "load_metadata_rows",
  "measure_g1_motion_quality",
  "metadata_rejection_reason",
  "parse_metadata_bool",
  "quality_as_dict",
  "quality_rejection_reason",
  "stable_fraction",
]
