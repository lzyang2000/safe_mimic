"""Compile a SOMA human motion into a timed robot-path intersection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safe_mimic.motions.human_capsules import (
  CapsuleDomainRandomization,
  fit_soma_capsules,
)
from safe_mimic.motions.soma_bvh import BvhMotion


def _as_float_array(value: np.ndarray, shape_tail: tuple[int, ...]) -> np.ndarray:
  array = np.asarray(value, dtype=np.float64)
  if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
    raise ValueError(f"expected shape (N, {shape_tail}), got {array.shape}")
  if len(array) < 1 or not np.isfinite(array).all():
    raise ValueError("trajectory arrays must be non-empty and finite")
  return array


def _yaw_from_wxyz(quaternions: np.ndarray) -> np.ndarray:
  w, x, y, z = np.moveaxis(quaternions, -1, 0)
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_z(points: np.ndarray, yaw_rad: float) -> np.ndarray:
  cosine = np.cos(yaw_rad)
  sine = np.sin(yaw_rad)
  rotated = np.array(points, dtype=np.float64, copy=True)
  x = points[..., 0]
  y = points[..., 1]
  rotated[..., 0] = cosine * x - sine * y
  rotated[..., 1] = sine * x + cosine * y
  return rotated


def _multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = np.moveaxis(left, -1, 0)
  rw, rx, ry, rz = np.moveaxis(right, -1, 0)
  return np.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    axis=-1,
  )


def _bvh_to_mujoco(points: np.ndarray) -> np.ndarray:
  """Convert BVH X-right/Y-up/Z-forward to MuJoCo X/Y/Z-up."""

  return np.stack((points[..., 0], points[..., 2], points[..., 1]), axis=-1)


@dataclass(frozen=True)
class RobotTrajectory:
  """A timestamped robot root trajectory in the MuJoCo world frame."""

  times_s: np.ndarray
  root_positions_w: np.ndarray
  root_yaw_w: np.ndarray

  def __post_init__(self) -> None:
    times = np.asarray(self.times_s, dtype=np.float64)
    positions = _as_float_array(self.root_positions_w, (3,))
    yaw = np.asarray(self.root_yaw_w, dtype=np.float64)
    if times.ndim != 1 or yaw.ndim != 1:
      raise ValueError("times_s and root_yaw_w must be one-dimensional")
    if len(times) != len(positions) or len(yaw) != len(times):
      raise ValueError("robot trajectory arrays must have equal lengths")
    if not np.isfinite(times).all() or not np.isfinite(yaw).all():
      raise ValueError("robot trajectory arrays must be finite")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
      raise ValueError("robot trajectory timestamps must be strictly increasing")
    object.__setattr__(self, "times_s", times)
    object.__setattr__(self, "root_positions_w", positions)
    object.__setattr__(self, "root_yaw_w", np.unwrap(yaw))

  @property
  def duration_s(self) -> float:
    return float(self.times_s[-1] - self.times_s[0])

  def position_at(self, time_s: float | np.ndarray) -> np.ndarray:
    """Linearly sample root position, clamping beyond the recorded interval."""

    query = np.asarray(time_s, dtype=np.float64)
    sampled = np.stack(
      [
        np.interp(query, self.times_s, self.root_positions_w[:, axis])
        for axis in range(3)
      ],
      axis=-1,
    )
    return sampled

  def yaw_at(self, time_s: float | np.ndarray) -> np.ndarray:
    """Linearly sample unwrapped root yaw."""

    return np.interp(
      np.asarray(time_s, dtype=np.float64), self.times_s, self.root_yaw_w
    )

  def path_heading_at(self, time_s: float, minimum_speed_m_s: float = 0.05) -> float:
    """Return planar travel heading, falling back to body yaw when nearly still."""

    if len(self.times_s) < 2:
      return float(self.root_yaw_w[0])
    median_dt = float(np.median(np.diff(self.times_s)))
    half_window = max(0.1, 2.0 * median_dt)
    before_s = max(float(self.times_s[0]), time_s - half_window)
    after_s = min(float(self.times_s[-1]), time_s + half_window)
    elapsed_s = after_s - before_s
    if elapsed_s <= 0.0:
      return float(self.yaw_at(time_s))
    velocity_xy = (
      self.position_at(after_s)[:2] - self.position_at(before_s)[:2]
    ) / elapsed_s
    if np.linalg.norm(velocity_xy) < minimum_speed_m_s:
      return float(self.yaw_at(time_s))
    return float(np.arctan2(velocity_xy[1], velocity_xy[0]))


def load_mjlab_robot_trajectory(
  path: Path | str, root_body_index: int = 0
) -> RobotTrajectory:
  """Load the root path from an mjlab tracking-motion ``.npz`` file."""

  path = Path(path)
  with np.load(path, allow_pickle=False) as data:
    missing = {"fps", "body_pos_w", "body_quat_w"} - set(data.files)
    if missing:
      raise ValueError(f"{path} is missing mjlab arrays: {sorted(missing)}")
    fps_values = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
    if len(fps_values) != 1 or fps_values[0] <= 0.0:
      raise ValueError("mjlab motion fps must contain one positive value")
    positions = np.asarray(data["body_pos_w"], dtype=np.float64)
    quaternions = np.asarray(data["body_quat_w"], dtype=np.float64)
  if positions.ndim != 3 or positions.shape[-1] != 3:
    raise ValueError("body_pos_w must have shape (frames, bodies, 3)")
  if quaternions.shape != (*positions.shape[:2], 4):
    raise ValueError("body_quat_w must have shape (frames, bodies, 4)")
  if not 0 <= root_body_index < positions.shape[1]:
    raise ValueError(f"root_body_index {root_body_index} is out of range")
  times_s = np.arange(len(positions), dtype=np.float64) / fps_values[0]
  root_quaternions = quaternions[:, root_body_index]
  return RobotTrajectory(
    times_s=times_s,
    root_positions_w=positions[:, root_body_index],
    root_yaw_w=_yaw_from_wxyz(root_quaternions),
  )


@dataclass(frozen=True)
class CompiledHumanTrajectory:
  """World-space human joints and collision capsules synchronized to a robot."""

  times_s: np.ndarray
  source_frame_indices: np.ndarray
  joint_names: tuple[str, ...]
  joint_positions_w: np.ndarray
  root_positions_w: np.ndarray
  capsule_names: tuple[str, ...]
  capsule_centers_w: np.ndarray
  capsule_quaternions_wxyz: np.ndarray
  capsule_radii_m: np.ndarray
  capsule_half_lengths_m: np.ndarray
  robot_root_positions_w: np.ndarray
  intersection_frame: int
  intersection_time_s: float
  placement_yaw_rad: float
  source_path: str
  source_start_time_s: float = 0.0
  source_description: str = ""

  @property
  def intersection_distance_m(self) -> float:
    delta = (
      self.root_positions_w[self.intersection_frame, :2]
      - self.robot_root_positions_w[self.intersection_frame, :2]
    )
    return float(np.linalg.norm(delta))

  def save(self, path: Path | str) -> None:
    """Write the compiled trajectory without pickle-dependent arrays."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      path,
      format_version=np.asarray([1], dtype=np.int32),
      times_s=self.times_s,
      source_frame_indices=self.source_frame_indices,
      joint_names=np.asarray(self.joint_names),
      joint_positions_w=self.joint_positions_w.astype(np.float32),
      root_positions_w=self.root_positions_w.astype(np.float32),
      capsule_names=np.asarray(self.capsule_names),
      capsule_centers_w=self.capsule_centers_w.astype(np.float32),
      capsule_quaternions_wxyz=self.capsule_quaternions_wxyz.astype(np.float32),
      capsule_radii_m=self.capsule_radii_m.astype(np.float32),
      capsule_half_lengths_m=self.capsule_half_lengths_m.astype(np.float32),
      robot_root_positions_w=self.robot_root_positions_w.astype(np.float32),
      intersection_frame=np.asarray([self.intersection_frame], dtype=np.int32),
      intersection_time_s=np.asarray([self.intersection_time_s]),
      intersection_distance_m=np.asarray([self.intersection_distance_m]),
      placement_yaw_rad=np.asarray([self.placement_yaw_rad]),
      source_path=np.asarray(self.source_path),
      source_start_time_s=np.asarray([self.source_start_time_s]),
      source_description=np.asarray(self.source_description),
    )


def load_compiled_human_trajectory(path: Path | str) -> CompiledHumanTrajectory:
  """Load a trajectory produced by :meth:`CompiledHumanTrajectory.save`."""

  with np.load(path, allow_pickle=False) as data:
    version = int(np.asarray(data["format_version"]).reshape(-1)[0])
    if version != 1:
      raise ValueError(f"unsupported compiled human trajectory version {version}")
    return CompiledHumanTrajectory(
      times_s=np.asarray(data["times_s"], dtype=np.float64),
      source_frame_indices=np.asarray(data["source_frame_indices"], dtype=np.int64),
      joint_names=tuple(str(value) for value in data["joint_names"]),
      joint_positions_w=np.asarray(data["joint_positions_w"], dtype=np.float64),
      root_positions_w=np.asarray(data["root_positions_w"], dtype=np.float64),
      capsule_names=tuple(str(value) for value in data["capsule_names"]),
      capsule_centers_w=np.asarray(data["capsule_centers_w"], dtype=np.float64),
      capsule_quaternions_wxyz=np.asarray(
        data["capsule_quaternions_wxyz"], dtype=np.float64
      ),
      capsule_radii_m=np.asarray(data["capsule_radii_m"], dtype=np.float64),
      capsule_half_lengths_m=np.asarray(
        data["capsule_half_lengths_m"], dtype=np.float64
      ),
      robot_root_positions_w=np.asarray(
        data["robot_root_positions_w"], dtype=np.float64
      ),
      intersection_frame=int(data["intersection_frame"][0]),
      intersection_time_s=float(data["intersection_time_s"][0]),
      placement_yaw_rad=float(data["placement_yaw_rad"][0]),
      source_path=str(data["source_path"]),
      source_start_time_s=float(data["source_start_time_s"][0]),
      source_description=str(data["source_description"]),
    )


def _motion_facing_xy(motion: BvhMotion, frame_index: int) -> np.ndarray:
  indices = {name: index for index, name in enumerate(motion.joint_names)}
  root_index = indices.get("Hips", 0)
  forward_bvh = motion.rotations_world[frame_index, root_index] @ np.asarray(
    [0.0, 0.0, 1.0]
  )
  forward_xy = np.asarray([forward_bvh[0], forward_bvh[2]])
  norm = np.linalg.norm(forward_xy)
  if norm < 1e-8:
    return np.asarray([1.0, 0.0])
  return forward_xy / norm


def _motion_path_heading(
  motion: BvhMotion,
  root_positions_m: np.ndarray,
  frame_index: int,
  minimum_displacement_m: float = 0.03,
) -> float:
  before = max(0, frame_index - 2)
  after = min(len(root_positions_m) - 1, frame_index + 2)
  displacement = root_positions_m[after, :2] - root_positions_m[before, :2]
  if np.linalg.norm(displacement) < minimum_displacement_m:
    displacement = _motion_facing_xy(motion, frame_index)
  return float(np.arctan2(displacement[1], displacement[0]))


def compile_human_robot_intersection(
  motion: BvhMotion,
  robot: RobotTrajectory,
  *,
  intersection_time_s: float,
  intersection_phase: float = 0.5,
  crossing_angle_rad: float = np.pi / 2.0,
  intersection_offset_robot_m: tuple[float, float] = (0.0, 0.0),
  ground_height_m: float = 0.0,
  randomization: CapsuleDomainRandomization | None = None,
  source_start_time_s: float = 0.0,
  source_description: str = "",
) -> CompiledHumanTrajectory:
  """Place a human clip so its pelvis meets a robot trajectory at one time.

  ``crossing_angle_rad`` is relative to the robot's direction of travel. A value
  of 90 degrees produces a perpendicular crossing. ``intersection_offset_robot_m``
  is ``(forward, left)`` in the robot body frame; its default of zero creates an
  exact synchronized root-path intersection, while a nonzero value creates a
  controlled near miss.
  """

  frame_count = len(motion.positions_m)
  if frame_count < 1:
    raise ValueError("human motion has no frames")
  if not 0.0 <= intersection_phase <= 1.0:
    raise ValueError("intersection_phase must be in [0, 1]")
  if not float(robot.times_s[0]) <= intersection_time_s <= float(robot.times_s[-1]):
    raise ValueError("intersection_time_s lies outside the robot trajectory")
  if len(intersection_offset_robot_m) != 2:
    raise ValueError("intersection_offset_robot_m must contain (forward, left)")

  intersection_frame = int(round(intersection_phase * (frame_count - 1)))
  sample_times_s = motion.frame_indices.astype(np.float64) * motion.source_frame_time
  sample_times_s -= sample_times_s[intersection_frame]
  times_s = sample_times_s + intersection_time_s

  fit = fit_soma_capsules(motion.joint_names, motion.positions_m, randomization)
  joint_positions = _bvh_to_mujoco(motion.positions_m)
  if randomization is not None:
    joint_positions *= randomization.body_scale_xyz

  robot_path_heading = robot.path_heading_at(intersection_time_s)
  desired_human_heading = robot_path_heading + crossing_angle_rad
  source_human_heading = _motion_path_heading(
    motion, fit.root_path_m, intersection_frame
  )
  placement_yaw = desired_human_heading - source_human_heading

  joint_positions = _rotate_z(joint_positions, placement_yaw)
  root_positions = _rotate_z(fit.root_path_m, placement_yaw)
  capsule_centers = _rotate_z(fit.centers_m, placement_yaw)
  yaw_quaternion = np.asarray(
    [np.cos(0.5 * placement_yaw), 0.0, 0.0, np.sin(0.5 * placement_yaw)]
  )
  capsule_quaternions = _multiply_quaternions_wxyz(
    yaw_quaternion, fit.quaternions_wxyz
  )

  robot_target = np.asarray(robot.position_at(intersection_time_s), dtype=np.float64)
  robot_yaw = float(robot.yaw_at(intersection_time_s))
  forward_offset, left_offset = intersection_offset_robot_m
  target_xy = robot_target[:2] + _rotate_z(
    np.asarray([forward_offset, left_offset, 0.0]), robot_yaw
  )[:2]
  translation_xy = target_xy - root_positions[intersection_frame, :2]

  foot_names = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
  joint_indices = {name: index for index, name in enumerate(motion.joint_names)}
  support_indices = [
    joint_indices[name] for name in foot_names if name in joint_indices
  ]
  if support_indices:
    source_ground_z = float(np.percentile(joint_positions[:, support_indices, 2], 2.0))
  else:
    source_ground_z = float(np.min(joint_positions[..., 2]))
  translation = np.asarray(
    [translation_xy[0], translation_xy[1], ground_height_m - source_ground_z]
  )
  joint_positions += translation
  root_positions += translation
  capsule_centers += translation

  robot_positions = robot.position_at(times_s)
  return CompiledHumanTrajectory(
    times_s=times_s,
    source_frame_indices=motion.frame_indices.copy(),
    joint_names=motion.joint_names,
    joint_positions_w=joint_positions,
    root_positions_w=root_positions,
    capsule_names=fit.names,
    capsule_centers_w=capsule_centers,
    capsule_quaternions_wxyz=capsule_quaternions,
    capsule_radii_m=fit.radii_m,
    capsule_half_lengths_m=fit.half_lengths_m,
    robot_root_positions_w=robot_positions,
    intersection_frame=intersection_frame,
    intersection_time_s=float(intersection_time_s),
    placement_yaw_rad=float(placement_yaw),
    source_path=str(motion.path),
    source_start_time_s=float(source_start_time_s),
    source_description=source_description,
  )
