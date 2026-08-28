"""Disk-backed capsule paths with current-frame-only online GPU sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _nlerp_quaternions(
  first: np.ndarray, second: np.ndarray, blend: np.ndarray
) -> np.ndarray:
  sign = np.where(np.sum(first * second, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
  result = first * (1.0 - blend) + second * sign * blend
  result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
  return result


def _nlerp_quaternions_torch(
  first: torch.Tensor, second: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  sign = torch.where(
    torch.sum(first * second, dim=-1, keepdim=True) < 0.0, -1.0, 1.0
  )
  result = first * (1.0 - blend) + second * sign * blend
  return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(
    1e-12
  )


def _rotate_xy_torch(points: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  cosine = torch.cos(yaw)
  sine = torch.sin(yaw)
  while cosine.ndim < points.ndim - 1:
    cosine = cosine.unsqueeze(-1)
    sine = sine.unsqueeze(-1)
  x = points[..., 0]
  y = points[..., 1]
  return torch.stack(
    (cosine * x - sine * y, sine * x + cosine * y, points[..., 2]), dim=-1
  )


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
  lw, lx, ly, lz = left.unbind(dim=-1)
  rw, rx, ry, rz = right.unbind(dim=-1)
  return torch.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    dim=-1,
  )


def _quat_z_axis(quaternion: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quaternion.unbind(dim=-1)
  return torch.stack(
    (
      2.0 * (x * z + w * y),
      2.0 * (y * z - w * x),
      1.0 - 2.0 * (x * x + y * y),
    ),
    dim=-1,
  )


def _quat_from_z(vectors: torch.Tensor) -> torch.Tensor:
  unit = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(
    1e-12
  )
  dot = unit[..., 2].clamp(-1.0, 1.0)
  quaternion = torch.stack(
    (1.0 + dot, -unit[..., 1], unit[..., 0], torch.zeros_like(dot)), dim=-1
  )
  opposite = dot < -0.999999
  fallback = torch.zeros_like(quaternion)
  fallback[..., 1] = 1.0
  quaternion = torch.where(opposite[..., None], fallback, quaternion)
  return quaternion / torch.linalg.vector_norm(
    quaternion, dim=-1, keepdim=True
  ).clamp_min(1e-12)


@dataclass(frozen=True)
class CapsulePathFrames:
  """CPU current-frame samples fetched from a memory-mapped path bank."""

  centers_m: np.ndarray
  quaternions_wxyz: np.ndarray
  root_positions_m: np.ndarray
  radii_m: np.ndarray
  half_lengths_m: np.ndarray
  active: np.ndarray


class CapsulePathBank:
  """Memory-map prebuilt paths without loading the complete bank into RAM."""

  def __init__(self, path: Path | str, *, preload: bool = False) -> None:
    self.path = Path(path)
    with (self.path / "bank.json").open() as source:
      self.config = json.load(source)
    if self.config.get("format_version") != 1 or not self.config.get("complete"):
      raise ValueError(f"{self.path} is not a complete capsule path bank")
    self.fps = float(self.config["fps"])
    self.max_frames = int(self.config["max_frames"])
    self.capsule_names = tuple(self.config["capsule_names"])
    self.preloaded = preload
    self.centers = self._load_array("centers.npy")
    self.quaternions = self._load_array("quaternions.npy")
    self.root_positions = self._load_array("root_positions.npy")
    self.facing_yaw = self._load_array("facing_yaw.npy")
    self.radii = self._load_array("radii.npy")
    self.half_lengths = self._load_array("half_lengths.npy")
    self.frame_counts = self._load_array("frame_counts.npy")
    self.ground_z = self._load_array("ground_z.npy")
    self._validate_shapes()

  def _load_array(self, name: str) -> np.ndarray:
    array = np.load(self.path / name, mmap_mode="r")
    # A concrete copy avoids random SSD page faults during thousands of
    # scattered per-environment frame gathers. The mmap remains the low-RAM
    # fallback and still benefits from the operating system's page cache.
    return np.array(array, copy=True) if self.preloaded else array

  def _validate_shapes(self) -> None:
    path_count = int(self.config["path_count"])
    capsule_count = len(self.capsule_names)
    expected_pose = (path_count, self.max_frames, capsule_count)
    if self.centers.shape != (*expected_pose, 3):
      raise ValueError(f"invalid centers shape {self.centers.shape}")
    if self.quaternions.shape != (*expected_pose, 4):
      raise ValueError(f"invalid quaternions shape {self.quaternions.shape}")
    if self.root_positions.shape != (path_count, self.max_frames, 3):
      raise ValueError(f"invalid root_positions shape {self.root_positions.shape}")
    if self.facing_yaw.shape != (path_count, self.max_frames):
      raise ValueError(f"invalid facing_yaw shape {self.facing_yaw.shape}")
    if self.radii.shape != (path_count, capsule_count):
      raise ValueError(f"invalid radii shape {self.radii.shape}")
    if self.half_lengths.shape != (path_count, capsule_count):
      raise ValueError(f"invalid half_lengths shape {self.half_lengths.shape}")
    if self.frame_counts.shape != (path_count,) or self.ground_z.shape != (path_count,):
      raise ValueError("invalid path scalar-array shapes")
    if np.any(self.frame_counts < 1) or np.any(self.frame_counts > self.max_frames):
      raise ValueError("path frame count lies outside the bank capacity")

  def __len__(self) -> int:
    return len(self.frame_counts)

  def duration_s(self, path_ids: np.ndarray) -> np.ndarray:
    path_ids = np.asarray(path_ids, dtype=np.int64)
    return (self.frame_counts[path_ids] - 1) / self.fps

  def sample(
    self, path_ids: np.ndarray, local_times_s: np.ndarray
  ) -> CapsulePathFrames:
    """Interpolate current frames on CPU; no full path is copied into memory."""

    path_ids = np.asarray(path_ids, dtype=np.int64)
    local_times_s = np.asarray(local_times_s, dtype=np.float64)
    if path_ids.shape != local_times_s.shape or path_ids.ndim != 1:
      raise ValueError("path_ids and local_times_s must be equal-length vectors")
    if np.any(path_ids < 0) or np.any(path_ids >= len(self)):
      raise ValueError("path id is outside the bank")

    last_frames = self.frame_counts[path_ids].astype(np.int64) - 1
    frame_coordinates = local_times_s * self.fps
    active = (frame_coordinates >= 0.0) & (frame_coordinates <= last_frames)
    clamped = np.clip(frame_coordinates, 0.0, last_frames)
    first_frames = np.floor(clamped).astype(np.int64)
    second_frames = np.minimum(first_frames + 1, last_frames)
    blend = (clamped - first_frames).astype(np.float32)

    first_centers = np.asarray(self.centers[path_ids, first_frames])
    second_centers = np.asarray(self.centers[path_ids, second_frames])
    first_root = np.asarray(self.root_positions[path_ids, first_frames])
    second_root = np.asarray(self.root_positions[path_ids, second_frames])
    first_quaternions = np.asarray(self.quaternions[path_ids, first_frames])
    second_quaternions = np.asarray(self.quaternions[path_ids, second_frames])
    centers = first_centers * (1.0 - blend[:, None, None]) + second_centers * (
      blend[:, None, None]
    )
    roots = first_root * (1.0 - blend[:, None]) + second_root * blend[:, None]
    quaternions = _nlerp_quaternions(
      first_quaternions, second_quaternions, blend[:, None, None]
    )
    return CapsulePathFrames(
      centers_m=centers,
      quaternions_wxyz=quaternions,
      root_positions_m=roots,
      radii_m=np.asarray(self.radii[path_ids]),
      half_lengths_m=np.asarray(self.half_lengths[path_ids]),
      active=active,
    )

  def path_heading(self, path_ids: np.ndarray, local_times_s: np.ndarray) -> np.ndarray:
    """Return path tangent, falling back to stored pelvis facing when stationary."""

    path_ids = np.asarray(path_ids, dtype=np.int64)
    local_times_s = np.asarray(local_times_s, dtype=np.float64)
    window_s = 2.0 / self.fps
    before = self.sample(path_ids, local_times_s - window_s).root_positions_m[:, :2]
    after = self.sample(path_ids, local_times_s + window_s).root_positions_m[:, :2]
    displacement = after - before
    heading = np.arctan2(displacement[:, 1], displacement[:, 0])
    stationary = np.linalg.norm(displacement, axis=-1) < 0.03
    if np.any(stationary):
      frame = np.rint(local_times_s[stationary] * self.fps).astype(np.int64)
      ids = path_ids[stationary]
      frame = np.clip(frame, 0, self.frame_counts[ids].astype(np.int64) - 1)
      heading[stationary] = self.facing_yaw[ids, frame]
    return heading


@dataclass(frozen=True)
class OnlineHumanPoses:
  """Only the current per-environment human geometry needed by MJLab."""

  centers_w: torch.Tensor
  quaternions_wxyz: torch.Tensor
  radii_m: torch.Tensor
  half_lengths_m: torch.Tensor
  root_positions_w: torch.Tensor
  active: torch.Tensor


class OnlineCapsulePathSampler:
  """Schedule path intersections and stream only current frames to a device."""

  def __init__(
    self,
    bank: CapsulePathBank,
    num_envs: int,
    device: str | torch.device,
    *,
    inactive_height_m: float = -100.0,
    update_hz: float = 10.0,
    resident_path_ids: np.ndarray | None = None,
    preload_to_device: bool = False,
  ) -> None:
    if num_envs < 1:
      raise ValueError("num_envs must be positive")
    if update_hz <= 0.0:
      raise ValueError("update_hz must be positive")
    self.bank = bank
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.inactive_height_m = inactive_height_m
    self.update_period_s = 1.0 / update_hz
    self.resident_path_ids: np.ndarray | None = None
    self.device_storage_bytes = 0
    self._device_path_lookup: torch.Tensor | None = None
    self._device_centers: torch.Tensor | None = None
    self._device_quaternions: torch.Tensor | None = None
    self._device_root_positions: torch.Tensor | None = None
    self._device_radii: torch.Tensor | None = None
    self._device_half_lengths: torch.Tensor | None = None
    self._device_frame_counts: torch.Tensor | None = None
    self._device_facing_yaw: torch.Tensor | None = None
    self._device_ground_z: torch.Tensor | None = None
    if preload_to_device:
      self._preload_paths_to_device(resident_path_ids)
    self.path_ids = np.zeros(num_envs, dtype=np.int64)
    self.path_durations_s = np.zeros(num_envs, dtype=np.float64)
    self.global_intersection_times_s = np.zeros(num_envs, dtype=np.float64)
    self.local_intersection_times_s = np.zeros(num_envs, dtype=np.float64)
    self.next_update_times_s = np.full(num_envs, -np.inf, dtype=np.float64)
    self.dirty = np.ones(num_envs, dtype=np.bool_)
    self.placement_yaw = torch.zeros(num_envs, device=self.device)
    self.translation_w = torch.zeros((num_envs, 3), device=self.device)
    self.body_scale_xyz = torch.ones((num_envs, 3), device=self.device)
    self.radius_scale = torch.ones(num_envs, device=self.device)
    self.radius_margin_m = torch.zeros(num_envs, device=self.device)
    capsule_count = len(bank.capsule_names)
    self._poses = OnlineHumanPoses(
      centers_w=torch.zeros((num_envs, capsule_count, 3), device=self.device),
      quaternions_wxyz=torch.zeros(
        (num_envs, capsule_count, 4), device=self.device
      ),
      radii_m=torch.zeros((num_envs, capsule_count), device=self.device),
      half_lengths_m=torch.zeros((num_envs, capsule_count), device=self.device),
      root_positions_w=torch.zeros((num_envs, 3), device=self.device),
      active=torch.zeros(num_envs, dtype=torch.bool, device=self.device),
    )
    self._poses.quaternions_wxyz[..., 0] = 1.0

  def _preload_paths_to_device(
    self, resident_path_ids: np.ndarray | None
  ) -> None:
    """Copy a compact source-row subset into device tensors once at startup."""

    if resident_path_ids is None:
      resident_path_ids = np.arange(len(self.bank), dtype=np.int64)
    else:
      resident_path_ids = np.unique(
        np.asarray(resident_path_ids, dtype=np.int64)
      )
    if resident_path_ids.ndim != 1 or len(resident_path_ids) == 0:
      raise ValueError("resident_path_ids must be a non-empty vector")
    if np.any(resident_path_ids < 0) or np.any(resident_path_ids >= len(self.bank)):
      raise ValueError("resident path id is outside the bank")

    def copy_rows(array: np.ndarray) -> torch.Tensor:
      # Advanced indexing intentionally materializes only accepted rows, not
      # the complete on-disk bank. The temporary CPU allocation is released
      # as soon as the device copy completes.
      selected = np.asarray(array[resident_path_ids])
      return torch.as_tensor(selected.copy(), device=self.device)

    self.resident_path_ids = resident_path_ids
    self._device_centers = copy_rows(self.bank.centers)
    self._device_quaternions = copy_rows(self.bank.quaternions)
    self._device_root_positions = copy_rows(self.bank.root_positions)
    self._device_radii = copy_rows(self.bank.radii)
    self._device_half_lengths = copy_rows(self.bank.half_lengths)
    self._device_frame_counts = copy_rows(self.bank.frame_counts).long()
    self._device_facing_yaw = copy_rows(self.bank.facing_yaw)
    self._device_ground_z = copy_rows(self.bank.ground_z).float()
    lookup = torch.full(
      (len(self.bank),), -1, dtype=torch.long, device=self.device
    )
    source_ids = torch.as_tensor(
      resident_path_ids, dtype=torch.long, device=self.device
    )
    lookup[source_ids] = torch.arange(len(source_ids), device=self.device)
    self._device_path_lookup = lookup
    tensors = (
      self._device_centers,
      self._device_quaternions,
      self._device_root_positions,
      self._device_radii,
      self._device_half_lengths,
      self._device_frame_counts,
      self._device_facing_yaw,
      self._device_ground_z,
      self._device_path_lookup,
    )
    self.device_storage_bytes = sum(
      tensor.numel() * tensor.element_size() for tensor in tensors
    )

  def _sample_device(
    self,
    path_ids: np.ndarray | torch.Tensor,
    local_times_s: np.ndarray | torch.Tensor,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    """Interpolate current frames directly from the compact device bank."""

    assert self._device_path_lookup is not None
    assert self._device_centers is not None
    assert self._device_quaternions is not None
    assert self._device_root_positions is not None
    assert self._device_radii is not None
    assert self._device_half_lengths is not None
    assert self._device_frame_counts is not None
    source_ids = torch.as_tensor(path_ids, dtype=torch.long, device=self.device)
    rows = self._device_path_lookup[source_ids]
    if torch.any(rows < 0):
      raise ValueError("scheduled path is not resident on the target device")
    times = torch.as_tensor(local_times_s, dtype=torch.float32, device=self.device)
    last_frames = self._device_frame_counts[rows] - 1
    coordinates = times * self.bank.fps
    active = (coordinates >= 0.0) & (coordinates <= last_frames)
    clamped = torch.minimum(
      torch.clamp_min(coordinates, 0.0), last_frames.float()
    )
    first_frames = torch.floor(clamped).long()
    second_frames = torch.minimum(first_frames + 1, last_frames)
    blend = (clamped - first_frames).float()

    first_centers = self._device_centers[rows, first_frames].float()
    second_centers = self._device_centers[rows, second_frames].float()
    centers = torch.lerp(first_centers, second_centers, blend[:, None, None])
    first_roots = self._device_root_positions[rows, first_frames].float()
    second_roots = self._device_root_positions[rows, second_frames].float()
    roots = torch.lerp(first_roots, second_roots, blend[:, None])
    quaternions = _nlerp_quaternions_torch(
      self._device_quaternions[rows, first_frames].float(),
      self._device_quaternions[rows, second_frames].float(),
      blend[:, None, None],
    )
    return (
      centers,
      quaternions,
      roots,
      self._device_radii[rows].float(),
      self._device_half_lengths[rows].float(),
      active,
    )

  def _path_heading_device(
    self, path_ids: torch.Tensor, local_times_s: torch.Tensor
  ) -> torch.Tensor:
    assert self._device_path_lookup is not None
    assert self._device_frame_counts is not None
    assert self._device_facing_yaw is not None
    window_s = 2.0 / self.bank.fps
    before = self._sample_device(path_ids, local_times_s - window_s)[2][..., :2]
    after = self._sample_device(path_ids, local_times_s + window_s)[2][..., :2]
    displacement = after - before
    heading = torch.atan2(displacement[:, 1], displacement[:, 0])
    stationary = torch.linalg.vector_norm(displacement, dim=-1) < 0.03
    if torch.any(stationary):
      rows = self._device_path_lookup[path_ids]
      last = self._device_frame_counts[rows] - 1
      frame = torch.round(local_times_s * self.bank.fps).long()
      frame = torch.minimum(torch.clamp_min(frame, 0), last)
      facing = self._device_facing_yaw[rows, frame].float()
      heading = torch.where(stationary, facing, heading)
    return heading

  def _schedule_intersections_device(
    self,
    env_ids: np.ndarray,
    *,
    path_ids: np.ndarray | torch.Tensor,
    global_intersection_times_s: np.ndarray | torch.Tensor,
    robot_positions_at_intersection_w: np.ndarray | torch.Tensor,
    robot_yaw_at_intersection: np.ndarray | torch.Tensor,
    robot_path_heading_at_intersection: np.ndarray | torch.Tensor,
    intersection_phase: np.ndarray | torch.Tensor | float,
    crossing_angle_rad: np.ndarray | torch.Tensor | float,
    offset_robot_m: np.ndarray | torch.Tensor | None,
    ground_height_m: np.ndarray | torch.Tensor | float,
    body_scale_xyz: np.ndarray | torch.Tensor | None,
    radius_scale: np.ndarray | torch.Tensor | float,
    radius_margin_m: np.ndarray | torch.Tensor | float,
  ) -> None:
    """Solve placement using only compact GPU-resident motion tensors."""

    assert self._device_path_lookup is not None
    assert self._device_frame_counts is not None
    assert self._device_ground_z is not None
    count = len(env_ids)
    device_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def tensor(value: object, shape: tuple[int, ...]) -> torch.Tensor:
      result = torch.as_tensor(value, dtype=torch.float32, device=self.device)
      return torch.broadcast_to(result, shape)

    source_ids = torch.as_tensor(path_ids, dtype=torch.long, device=self.device)
    if source_ids.shape != (count,):
      raise ValueError("env_ids and path_ids must be equal-length vectors")
    if torch.any((source_ids < 0) | (source_ids >= len(self.bank))):
      raise ValueError("path id is outside the bank")
    rows = self._device_path_lookup[source_ids]
    if torch.any(rows < 0):
      raise ValueError("scheduled path is not resident on the target device")

    global_times = tensor(global_intersection_times_s, (count,))
    robot_positions = tensor(robot_positions_at_intersection_w, (count, 3))
    robot_yaw = tensor(robot_yaw_at_intersection, (count,))
    robot_heading = tensor(robot_path_heading_at_intersection, (count,))
    phase = tensor(intersection_phase, (count,))
    crossing = tensor(crossing_angle_rad, (count,))
    ground = tensor(ground_height_m, (count,))
    radius_scale_tensor = tensor(radius_scale, (count,))
    radius_margin_tensor = tensor(radius_margin_m, (count,))
    offset = (
      torch.zeros((count, 2), device=self.device)
      if offset_robot_m is None
      else tensor(offset_robot_m, (count, 2))
    )
    body_scale = (
      torch.ones((count, 3), device=self.device)
      if body_scale_xyz is None
      else tensor(body_scale_xyz, (count, 3))
    )
    if torch.any((phase < 0.0) | (phase > 1.0)):
      raise ValueError("intersection phase must be in [0, 1]")
    if torch.any(body_scale <= 0.0) or torch.any(radius_scale_tensor <= 0.0):
      raise ValueError("body and radius scales must be positive")
    if torch.any(radius_margin_tensor < 0.0):
      raise ValueError("radius margin must be non-negative")

    durations = (self._device_frame_counts[rows] - 1).float() / self.bank.fps
    local_times = phase * durations
    source_heading = self._path_heading_device(source_ids, local_times)
    placement_yaw = robot_heading + crossing - source_heading
    local_root = self._sample_device(source_ids, local_times)[2] * body_scale
    cosine = torch.cos(placement_yaw)
    sine = torch.sin(placement_yaw)
    rotated_root_xy = torch.stack(
      (
        cosine * local_root[:, 0] - sine * local_root[:, 1],
        sine * local_root[:, 0] + cosine * local_root[:, 1],
      ),
      dim=-1,
    )
    offset_xy = torch.stack(
      (
        torch.cos(robot_yaw) * offset[:, 0]
        - torch.sin(robot_yaw) * offset[:, 1],
        torch.sin(robot_yaw) * offset[:, 0]
        + torch.cos(robot_yaw) * offset[:, 1],
      ),
      dim=-1,
    )
    target_xy = robot_positions[:, :2] + offset_xy
    translation = torch.cat(
      (
        target_xy - rotated_root_xy,
        (
          ground - self._device_ground_z[rows] * body_scale[:, 2]
        ).unsqueeze(-1),
      ),
      dim=-1,
    )

    self.path_ids[env_ids] = source_ids.cpu().numpy()
    self.path_durations_s[env_ids] = durations.cpu().numpy()
    self.global_intersection_times_s[env_ids] = global_times.cpu().numpy()
    self.local_intersection_times_s[env_ids] = local_times.cpu().numpy()
    self.placement_yaw[device_ids] = placement_yaw
    self.translation_w[device_ids] = translation
    self.body_scale_xyz[device_ids] = body_scale
    self.radius_scale[device_ids] = radius_scale_tensor
    self.radius_margin_m[device_ids] = radius_margin_tensor
    self.dirty[env_ids] = True

  def schedule_intersections(
    self,
    env_ids: np.ndarray | torch.Tensor,
    *,
    path_ids: np.ndarray | torch.Tensor,
    global_intersection_times_s: np.ndarray | torch.Tensor,
    robot_positions_at_intersection_w: np.ndarray | torch.Tensor,
    robot_yaw_at_intersection: np.ndarray | torch.Tensor,
    robot_path_heading_at_intersection: np.ndarray | torch.Tensor,
    intersection_phase: np.ndarray | torch.Tensor | float = 0.5,
    crossing_angle_rad: np.ndarray | torch.Tensor | float = np.pi / 2.0,
    offset_robot_m: np.ndarray | torch.Tensor | None = None,
    ground_height_m: np.ndarray | torch.Tensor | float = 0.0,
    body_scale_xyz: np.ndarray | torch.Tensor | None = None,
    radius_scale: np.ndarray | torch.Tensor | float = 1.0,
    radius_margin_m: np.ndarray | torch.Tensor | float = 0.0,
  ) -> None:
    """Assign paths and solve their rigid intersection placement at reset time."""

    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids.detach().cpu().numpy()
    env_ids = np.asarray(env_ids, dtype=np.int64)
    count = len(env_ids)
    if env_ids.ndim != 1:
      raise ValueError("env_ids must be a vector")
    if np.any(env_ids < 0) or np.any(env_ids >= self.num_envs):
      raise ValueError("environment id is out of range")
    if self._device_path_lookup is not None:
      self._schedule_intersections_device(
        env_ids,
        path_ids=path_ids,
        global_intersection_times_s=global_intersection_times_s,
        robot_positions_at_intersection_w=robot_positions_at_intersection_w,
        robot_yaw_at_intersection=robot_yaw_at_intersection,
        robot_path_heading_at_intersection=robot_path_heading_at_intersection,
        intersection_phase=intersection_phase,
        crossing_angle_rad=crossing_angle_rad,
        offset_robot_m=offset_robot_m,
        ground_height_m=ground_height_m,
        body_scale_xyz=body_scale_xyz,
        radius_scale=radius_scale,
        radius_margin_m=radius_margin_m,
      )
      return

    path_ids = np.asarray(path_ids, dtype=np.int64)
    if path_ids.shape != (count,):
      raise ValueError("env_ids and path_ids must be equal-length vectors")
    if np.any(path_ids < 0) or np.any(path_ids >= len(self.bank)):
      raise ValueError("path id is out of range")

    global_times = np.broadcast_to(
      np.asarray(global_intersection_times_s, dtype=np.float64), (count,)
    )
    robot_positions = np.broadcast_to(
      np.asarray(robot_positions_at_intersection_w, dtype=np.float64), (count, 3)
    )
    robot_yaw = np.broadcast_to(
      np.asarray(robot_yaw_at_intersection, dtype=np.float64), (count,)
    )
    robot_heading = np.broadcast_to(
      np.asarray(robot_path_heading_at_intersection, dtype=np.float64), (count,)
    )
    phase = np.broadcast_to(np.asarray(intersection_phase, dtype=np.float64), (count,))
    crossing = np.broadcast_to(
      np.asarray(crossing_angle_rad, dtype=np.float64), (count,)
    )
    ground = np.broadcast_to(np.asarray(ground_height_m, dtype=np.float64), (count,))
    radius_scale_array = np.broadcast_to(
      np.asarray(radius_scale, dtype=np.float64), (count,)
    )
    radius_margin_array = np.broadcast_to(
      np.asarray(radius_margin_m, dtype=np.float64), (count,)
    )
    if np.any((phase < 0.0) | (phase > 1.0)):
      raise ValueError("intersection phase must be in [0, 1]")
    if offset_robot_m is None:
      offset = np.zeros((count, 2), dtype=np.float64)
    else:
      offset = np.broadcast_to(np.asarray(offset_robot_m, dtype=np.float64), (count, 2))
    if body_scale_xyz is None:
      body_scale = np.ones((count, 3), dtype=np.float64)
    else:
      body_scale = np.broadcast_to(
        np.asarray(body_scale_xyz, dtype=np.float64), (count, 3)
      )
    if np.any(body_scale <= 0.0) or np.any(radius_scale_array <= 0.0):
      raise ValueError("body and radius scales must be positive")
    if np.any(radius_margin_array < 0.0):
      raise ValueError("radius margin must be non-negative")

    local_times = phase * self.bank.duration_s(path_ids)
    source_heading = self.bank.path_heading(path_ids, local_times)
    placement_yaw = robot_heading + crossing - source_heading
    local_root = self.bank.sample(path_ids, local_times).root_positions_m * body_scale
    cosine = np.cos(placement_yaw)
    sine = np.sin(placement_yaw)
    rotated_root_xy = np.column_stack(
      (
        cosine * local_root[:, 0] - sine * local_root[:, 1],
        sine * local_root[:, 0] + cosine * local_root[:, 1],
      )
    )
    offset_xy = np.column_stack(
      (
        np.cos(robot_yaw) * offset[:, 0] - np.sin(robot_yaw) * offset[:, 1],
        np.sin(robot_yaw) * offset[:, 0] + np.cos(robot_yaw) * offset[:, 1],
      )
    )
    target_xy = robot_positions[:, :2] + offset_xy
    translation = np.column_stack(
      (
        target_xy - rotated_root_xy,
        ground - self.bank.ground_z[path_ids] * body_scale[:, 2],
      )
    )

    self.path_ids[env_ids] = path_ids
    self.path_durations_s[env_ids] = self.bank.duration_s(path_ids)
    self.global_intersection_times_s[env_ids] = global_times
    self.local_intersection_times_s[env_ids] = local_times
    device_ids = torch.as_tensor(env_ids, device=self.device)
    self.placement_yaw[device_ids] = torch.as_tensor(
      placement_yaw.copy(), dtype=torch.float32, device=self.device
    )
    self.translation_w[device_ids] = torch.as_tensor(
      translation.copy(), dtype=torch.float32, device=self.device
    )
    self.body_scale_xyz[device_ids] = torch.as_tensor(
      body_scale.copy(), dtype=torch.float32, device=self.device
    )
    self.radius_scale[device_ids] = torch.as_tensor(
      radius_scale_array.copy(), dtype=torch.float32, device=self.device
    )
    self.radius_margin_m[device_ids] = torch.as_tensor(
      radius_margin_array.copy(), dtype=torch.float32, device=self.device
    )
    self.dirty[env_ids] = True

  def _update_envs(self, global_time_s: float, env_ids: np.ndarray) -> None:
    """Fetch and place current frames for a selected set of environments."""

    local_times = (
      global_time_s
      - self.global_intersection_times_s[env_ids]
      + self.local_intersection_times_s[env_ids]
    )
    device_ids = torch.as_tensor(env_ids, device=self.device)
    if self._device_path_lookup is not None:
      (
        centers,
        quaternions,
        roots,
        base_radii,
        base_half_lengths,
        active,
      ) = self._sample_device(self.path_ids[env_ids], local_times)
    else:
      frames = self.bank.sample(self.path_ids[env_ids], local_times)
      centers = torch.as_tensor(
        frames.centers_m.copy(), dtype=torch.float32, device=self.device
      )
      quaternions = torch.as_tensor(
        frames.quaternions_wxyz.copy(), dtype=torch.float32, device=self.device
      )
      roots = torch.as_tensor(
        frames.root_positions_m.copy(), dtype=torch.float32, device=self.device
      )
      base_radii = torch.as_tensor(
        frames.radii_m.copy(), dtype=torch.float32, device=self.device
      )
      base_half_lengths = torch.as_tensor(
        frames.half_lengths_m.copy(), dtype=torch.float32, device=self.device
      )
      active = torch.as_tensor(frames.active, device=self.device)

    body_scale = self.body_scale_xyz[device_ids]
    placement_yaw = self.placement_yaw[device_ids]
    translation = self.translation_w[device_ids]
    centers = centers * body_scale[:, None, :]
    roots = roots * body_scale
    capsule_axes = _quat_z_axis(quaternions)
    scaled_axes = capsule_axes * body_scale[:, None, :]
    axis_scales = torch.linalg.vector_norm(scaled_axes, dim=-1).clamp_min(1e-12)
    quaternions = _quat_from_z(scaled_axes)
    half_lengths = base_half_lengths * axis_scales

    centers = _rotate_xy_torch(centers, placement_yaw) + translation[:, None]
    roots = _rotate_xy_torch(roots, placement_yaw) + translation
    yaw_quaternion = torch.stack(
      (
        torch.cos(0.5 * placement_yaw),
        torch.zeros_like(placement_yaw),
        torch.zeros_like(placement_yaw),
        torch.sin(0.5 * placement_yaw),
      ),
      dim=-1,
    )
    quaternions = _quat_multiply(yaw_quaternion[:, None, :], quaternions)
    cross_section_scale = body_scale[:, :2].mean(dim=-1)
    radii = (
      base_radii
      * cross_section_scale[:, None]
      * self.radius_scale[device_ids][:, None]
      + self.radius_margin_m[device_ids][:, None]
    )

    inactive_centers = torch.zeros_like(centers)
    inactive_centers[..., 2] = self.inactive_height_m
    centers = torch.where(active[:, None, None], centers, inactive_centers)
    self._poses.centers_w[device_ids] = centers
    self._poses.quaternions_wxyz[device_ids] = quaternions
    self._poses.radii_m[device_ids] = radii
    self._poses.half_lengths_m[device_ids] = half_lengths
    self._poses.root_positions_w[device_ids] = roots
    self._poses.active[device_ids] = active
    self.dirty[env_ids] = False
    self.next_update_times_s[env_ids] = global_time_s + self.update_period_s

  def sample(self, global_time_s: float) -> OnlineHumanPoses:
    """Force a current-frame refresh for every environment."""

    self._update_envs(global_time_s, np.arange(self.num_envs))
    return self._poses

  def sample_held(self, global_time_s: float) -> OnlineHumanPoses:
    """Refresh due/dirty environments and hold poses between sensor scans."""

    update = self.dirty | (global_time_s + 1e-12 >= self.next_update_times_s)
    env_ids = np.flatnonzero(update)
    if len(env_ids):
      self._update_envs(global_time_s, env_ids)
    return self._poses
