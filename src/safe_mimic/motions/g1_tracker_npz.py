"""Convert BONES-SEED G1 CSV clips to mjlab tracker-compatible NPZ data."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from safe_mimic.motions.g1_dataset import load_g1_csv

TRACKER_NPZ_FIELDS = (
  "fps",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _target_times(frame_count: int, native_fps: float, target_fps: float) -> np.ndarray:
  duration_s = (frame_count - 1) / native_fps
  target_count = max(2, int(np.floor(duration_s * target_fps)) + 1)
  return np.arange(target_count, dtype=np.float64) / target_fps


def resample_g1_motion(
  root_pos_m: np.ndarray,
  root_euler_deg: np.ndarray,
  joint_pos_rad: np.ndarray,
  *,
  native_fps: float,
  target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """Interpolate a 120 Hz G1 trajectory onto an evenly spaced target grid."""
  frame_count = root_pos_m.shape[0]
  source_times = np.arange(frame_count, dtype=np.float64) / native_fps
  target_times = _target_times(frame_count, native_fps, target_fps)

  root_pos = np.stack(
    [np.interp(target_times, source_times, root_pos_m[:, axis]) for axis in range(3)],
    axis=1,
  )
  joint_pos = np.stack(
    [
      np.interp(target_times, source_times, joint_pos_rad[:, joint_id])
      for joint_id in range(29)
    ],
    axis=1,
  )
  root_rotations = Rotation.from_euler("xyz", root_euler_deg, degrees=True)
  root_quat_xyzw = Slerp(source_times, root_rotations)(target_times).as_quat()
  root_quat_wxyz = np.roll(root_quat_xyzw, shift=1, axis=1)
  return root_pos, root_quat_wxyz, joint_pos, target_fps


def quaternion_angular_velocity_w(
  quaternions_wxyz: np.ndarray,
  fps: float,
) -> np.ndarray:
  """Compute central-difference world-frame angular velocity from quaternions."""
  if quaternions_wxyz.ndim != 3 or quaternions_wxyz.shape[-1] != 4:
    raise ValueError("quaternions must have shape [frames, bodies, 4]")
  frame_count, body_count, _ = quaternions_wxyz.shape
  if frame_count < 2:
    raise ValueError("at least two quaternion frames are required")
  xyzw = np.roll(quaternions_wxyz, shift=-1, axis=-1)
  rotations = Rotation.from_quat(xyzw.reshape(-1, 4))
  rotations = rotations.as_matrix().reshape(frame_count, body_count, 3, 3)
  step_delta = np.einsum(
    "tbij,tbkj->tbik",
    rotations[1:],
    rotations[:-1],
  )
  step_rotvec = Rotation.from_matrix(step_delta.reshape(-1, 3, 3)).as_rotvec()
  step_velocity = step_rotvec.reshape(frame_count - 1, body_count, 3) * fps
  velocity = np.empty((frame_count, body_count, 3), dtype=np.float64)
  velocity[0] = step_velocity[0]
  velocity[-1] = step_velocity[-1]
  if frame_count > 2:
    velocity[1:-1] = 0.5 * (step_velocity[:-1] + step_velocity[1:])
  return velocity


def build_tracker_motion_arrays(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  root_pos_m: np.ndarray,
  root_euler_deg: np.ndarray,
  joint_pos_rad: np.ndarray,
  *,
  native_fps: float = 120.0,
  target_fps: float = 50.0,
) -> dict[str, np.ndarray]:
  """Return the exact array schema consumed by mjlab's MotionLoader."""
  root_pos, root_quat, joint_pos, fps = resample_g1_motion(
    root_pos_m,
    root_euler_deg,
    joint_pos_rad,
    native_fps=native_fps,
    target_fps=target_fps,
  )
  frame_count = root_pos.shape[0]
  body_count = model.nbody - 1
  if model.nq != 36 or body_count != 30:
    raise ValueError(
      f"expected G1 model nq=36 and 30 bodies, got nq={model.nq}, bodies={body_count}"
    )

  body_pos_w = np.empty((frame_count, body_count, 3), dtype=np.float64)
  body_quat_w = np.empty((frame_count, body_count, 4), dtype=np.float64)
  for frame_id in range(frame_count):
    data.qpos[:3] = root_pos[frame_id]
    data.qpos[3:7] = root_quat[frame_id]
    data.qpos[7:] = joint_pos[frame_id]
    mujoco.mj_kinematics(model, data)
    body_pos_w[frame_id] = data.xpos[1:]
    body_quat_w[frame_id] = data.xquat[1:]

  joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0)
  body_lin_vel_w = np.gradient(body_pos_w, 1.0 / fps, axis=0)
  body_ang_vel_w = quaternion_angular_velocity_w(body_quat_w, fps)
  return {
    "fps": np.asarray([fps], dtype=np.float64),
    "joint_pos": joint_pos.astype(np.float32),
    "joint_vel": joint_vel.astype(np.float32),
    "body_pos_w": body_pos_w.astype(np.float32),
    "body_quat_w": body_quat_w.astype(np.float32),
    "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
    "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
  }


def convert_g1_csv_to_tracker_npz(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  csv_path: Path | str,
  output_path: Path | str,
  *,
  native_fps: float = 120.0,
  target_fps: float = 50.0,
  compressed: bool = True,
) -> dict[str, np.ndarray]:
  """Convert one CSV atomically and return the generated arrays."""
  arrays = build_tracker_motion_arrays(
    model,
    data,
    *load_g1_csv(csv_path),
    native_fps=native_fps,
    target_fps=target_fps,
  )
  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
  with temporary.open("wb") as stream:
    if compressed:
      np.savez_compressed(stream, **arrays)
    else:
      np.savez(stream, **arrays)
  temporary.replace(output)
  return arrays


def validate_tracker_npz(path: Path | str) -> dict[str, tuple[int, ...]]:
  """Validate field presence, shapes, finiteness, and quaternion normalization."""
  with np.load(path) as data:
    if tuple(data.files) != TRACKER_NPZ_FIELDS:
      raise ValueError(f"unexpected NPZ fields: {data.files}")
    frame_count = data["joint_pos"].shape[0]
    expected = {
      "fps": (1,),
      "joint_pos": (frame_count, 29),
      "joint_vel": (frame_count, 29),
      "body_pos_w": (frame_count, 30, 3),
      "body_quat_w": (frame_count, 30, 4),
      "body_lin_vel_w": (frame_count, 30, 3),
      "body_ang_vel_w": (frame_count, 30, 3),
    }
    for name, shape in expected.items():
      if data[name].shape != shape:
        raise ValueError(f"{name} has shape {data[name].shape}, expected {shape}")
      if not np.isfinite(data[name]).all():
        raise ValueError(f"{name} contains non-finite values")
    quaternion_norm = np.linalg.norm(data["body_quat_w"], axis=-1)
    if not np.allclose(quaternion_norm, 1.0, atol=2e-4):
      raise ValueError("body quaternions are not normalized")
  return expected


__all__ = [
  "TRACKER_NPZ_FIELDS",
  "build_tracker_motion_arrays",
  "convert_g1_csv_to_tracker_npz",
  "quaternion_angular_velocity_w",
  "resample_g1_motion",
  "validate_tracker_npz",
]
