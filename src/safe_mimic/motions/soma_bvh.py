"""Small, dependency-light reader for the BONES-SEED SOMA BVH files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BvhJoint:
  """One joint in a BVH hierarchy."""

  name: str
  parent: int
  offset: np.ndarray
  channels: tuple[str, ...]
  channel_start: int


@dataclass(frozen=True)
class BvhMotion:
  """Downsampled global joint poses from one BVH motion."""

  path: Path
  joints: tuple[BvhJoint, ...]
  positions_m: np.ndarray
  rotations_world: np.ndarray
  frame_indices: np.ndarray
  source_frame_count: int
  source_frame_time: float

  @property
  def joint_names(self) -> tuple[str, ...]:
    return tuple(joint.name for joint in self.joints)

  @property
  def duration_s(self) -> float:
    return max(0, self.source_frame_count - 1) * self.source_frame_time


def _axis_rotation(axis: int, angles_rad: np.ndarray) -> np.ndarray:
  count = len(angles_rad)
  rotations = np.zeros((count, 3, 3), dtype=np.float64)
  cosine = np.cos(angles_rad)
  sine = np.sin(angles_rad)
  rotations[:, axis, axis] = 1.0
  other = [value for value in range(3) if value != axis]
  first, second = other
  rotations[:, first, first] = cosine
  rotations[:, second, second] = cosine
  rotations[:, first, second] = -sine
  rotations[:, second, first] = sine
  if axis == 1:
    rotations[:, first, second] = sine
    rotations[:, second, first] = -sine
  return rotations


def _parse_hierarchy(handle) -> tuple[tuple[BvhJoint, ...], int]:
  joints: list[dict[str, object]] = []
  stack: list[int] = []
  pending_joint: int | None = None
  channel_count = 0

  for raw_line in handle:
    line = raw_line.strip()
    if line == "MOTION":
      break
    if not line or line == "HIERARCHY":
      continue
    if line.startswith(("ROOT ", "JOINT ")):
      name = line.split(maxsplit=1)[1]
      parent = stack[-1] if stack else -1
      pending_joint = len(joints)
      joints.append(
        {
          "name": name,
          "parent": parent,
          "offset": np.zeros(3, dtype=np.float64),
          "channels": (),
          "channel_start": channel_count,
        }
      )
      continue
    if line == "End Site":
      parent = stack[-1]
      pending_joint = len(joints)
      joints.append(
        {
          "name": f"{joints[parent]['name']}EndSite",
          "parent": parent,
          "offset": np.zeros(3, dtype=np.float64),
          "channels": (),
          "channel_start": channel_count,
        }
      )
      continue
    if line == "{":
      if pending_joint is None:
        raise ValueError("BVH opening brace without a joint")
      stack.append(pending_joint)
      pending_joint = None
      continue
    if line == "}":
      stack.pop()
      continue
    if line.startswith("OFFSET "):
      joints[stack[-1]]["offset"] = np.asarray(
        [float(value) for value in line.split()[1:4]], dtype=np.float64
      )
      continue
    if line.startswith("CHANNELS "):
      parts = line.split()
      count = int(parts[1])
      channels = tuple(parts[2 : 2 + count])
      joints[stack[-1]]["channels"] = channels
      joints[stack[-1]]["channel_start"] = channel_count
      channel_count += count

  parsed = tuple(BvhJoint(**joint) for joint in joints)
  return parsed, channel_count


def _sample_frames(
  handle,
  channel_count: int,
  sample_count: int | None,
  *,
  start_time_s: float | None = None,
  duration_s: float | None = None,
  output_fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
  frames_line = handle.readline().strip()
  frame_time_line = handle.readline().strip()
  if not frames_line.startswith("Frames:"):
    raise ValueError(f"Expected BVH frame count, got {frames_line!r}")
  if not frame_time_line.startswith("Frame Time:"):
    raise ValueError(f"Expected BVH frame time, got {frame_time_line!r}")

  frame_count = int(frames_line.split(":", maxsplit=1)[1])
  frame_time = float(frame_time_line.split(":", maxsplit=1)[1])
  if output_fps is None:
    if sample_count is None or sample_count < 1:
      raise ValueError("sample_count must be positive")
    indices = np.rint(
      np.linspace(0, frame_count - 1, min(sample_count, frame_count))
    ).astype(np.int64)
  else:
    if output_fps <= 0:
      raise ValueError("output_fps must be positive")
    if duration_s is None or duration_s <= 0:
      raise ValueError("duration_s must be positive for real-time sampling")
    source_duration_s = max(0.0, (frame_count - 1) * frame_time)
    start_time_s = float(np.clip(start_time_s or 0.0, 0.0, source_duration_s))
    sampled_duration_s = min(duration_s, source_duration_s - start_time_s)
    output_count = max(1, int(np.ceil(sampled_duration_s * output_fps)))
    times_s = start_time_s + np.arange(output_count) / output_fps
    indices = np.rint(times_s / frame_time).astype(np.int64)
    indices = np.clip(indices, 0, frame_count - 1)

  wanted: dict[int, list[int]] = {}
  for slot, index in enumerate(indices):
    wanted.setdefault(int(index), []).append(slot)
  frames = np.empty((len(indices), channel_count), dtype=np.float64)

  for frame_index, line in enumerate(handle):
    slots = wanted.get(frame_index)
    if slots is None:
      continue
    values = np.fromstring(line, sep=" ", dtype=np.float64)
    if len(values) < channel_count:
      raise ValueError(
        f"Frame {frame_index} has {len(values)} of {channel_count} channels"
      )
    frames[slots] = values[:channel_count]

  return frames, indices, frame_count, frame_time


def _forward_kinematics(
  joints: tuple[BvhJoint, ...], frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  sample_count = len(frames)
  joint_count = len(joints)
  world_positions = np.empty((sample_count, joint_count, 3), dtype=np.float64)
  world_rotations = np.empty((sample_count, joint_count, 3, 3), dtype=np.float64)
  identity = np.broadcast_to(np.eye(3), (sample_count, 3, 3))
  axis_index = {"X": 0, "Y": 1, "Z": 2}

  for joint_index, joint in enumerate(joints):
    has_translation = any(channel.endswith("position") for channel in joint.channels)
    translation = np.zeros((sample_count, 3), dtype=np.float64)
    if not has_translation:
      translation[:] = joint.offset
    rotation = identity.copy()

    for offset, channel in enumerate(joint.channels):
      values = frames[:, joint.channel_start + offset]
      axis = axis_index[channel[0]]
      if channel.endswith("position"):
        translation[:, axis] = values
      elif channel.endswith("rotation"):
        axis_rotation = _axis_rotation(axis, np.deg2rad(values))
        rotation = np.einsum("nij,njk->nik", rotation, axis_rotation)

    if joint.parent < 0:
      world_positions[:, joint_index] = translation
      world_rotations[:, joint_index] = rotation
      continue

    parent_rotation = world_rotations[:, joint.parent]
    rotated_translation = np.einsum("nij,nj->ni", parent_rotation, translation)
    world_positions[:, joint_index] = (
      world_positions[:, joint.parent] + rotated_translation
    )
    world_rotations[:, joint_index] = np.einsum(
      "nij,njk->nik", parent_rotation, rotation
    )

  return world_positions * 0.01, world_rotations


def load_bvh_samples(path: Path | str, sample_count: int = 180) -> BvhMotion:
  """Load evenly spaced frames and compute global joint positions in meters."""

  path = Path(path)
  with path.open(errors="replace") as handle:
    joints, channel_count = _parse_hierarchy(handle)
    frames, indices, frame_count, frame_time = _sample_frames(
      handle, channel_count, sample_count
    )
  positions, rotations = _forward_kinematics(joints, frames)
  if not np.isfinite(positions).all():
    raise ValueError(f"Non-finite joint positions in {path}")
  return BvhMotion(
    path=path,
    joints=joints,
    positions_m=positions,
    rotations_world=rotations,
    frame_indices=indices,
    source_frame_count=frame_count,
    source_frame_time=frame_time,
  )


def load_bvh_window(
  path: Path | str,
  *,
  start_time_s: float,
  duration_s: float,
  output_fps: float,
) -> BvhMotion:
  """Load a real-time BVH window at a requested output frame rate.

  Unlike :func:`load_bvh_samples`, this preserves elapsed source time instead
  of spreading a fixed number of output samples across the entire clip.
  """

  path = Path(path)
  with path.open(errors="replace") as handle:
    joints, channel_count = _parse_hierarchy(handle)
    frames, indices, frame_count, frame_time = _sample_frames(
      handle,
      channel_count,
      None,
      start_time_s=start_time_s,
      duration_s=duration_s,
      output_fps=output_fps,
    )
  positions, rotations = _forward_kinematics(joints, frames)
  if not np.isfinite(positions).all():
    raise ValueError(f"Non-finite joint positions in {path}")
  return BvhMotion(
    path=path,
    joints=joints,
    positions_m=positions,
    rotations_world=rotations,
    frame_indices=indices,
    source_frame_count=frame_count,
    source_frame_time=frame_time,
  )
