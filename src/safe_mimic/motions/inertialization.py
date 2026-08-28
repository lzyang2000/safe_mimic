"""Velocity-aware skeleton inertialization for composing SOMA BVH clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safe_mimic.motions.soma_bvh import BvhMotion

_BVH_TO_MUJOCO = np.asarray(
  ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
  dtype=np.float64,
)


@dataclass(frozen=True)
class InertializedTransition:
  """A composed BVH motion and the location of its decaying transition."""

  motion: BvhMotion
  transition_start_frame: int
  transition_end_frame: int
  alignment_yaw_rad: float


def _convert_basis_positions(points: np.ndarray) -> np.ndarray:
  return np.einsum("ij,...j->...i", _BVH_TO_MUJOCO, points)


def _convert_basis_rotations(rotations: np.ndarray) -> np.ndarray:
  return np.einsum(
    "ij,...jk,lk->...il", _BVH_TO_MUJOCO, rotations, _BVH_TO_MUJOCO
  )


def _matrix_to_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
  m00 = matrices[..., 0, 0]
  m01 = matrices[..., 0, 1]
  m02 = matrices[..., 0, 2]
  m10 = matrices[..., 1, 0]
  m11 = matrices[..., 1, 1]
  m12 = matrices[..., 1, 2]
  m20 = matrices[..., 2, 0]
  m21 = matrices[..., 2, 1]
  m22 = matrices[..., 2, 2]
  squared = np.maximum(
    np.stack(
      (
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
      ),
      axis=-1,
    ),
    0.0,
  )
  magnitudes = np.sqrt(squared)
  candidates = np.stack(
    (
      np.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), axis=-1),
      np.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), axis=-1),
      np.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), axis=-1),
      np.stack((m10 - m01, m02 + m20, m12 + m21, squared[..., 3]), axis=-1),
    ),
    axis=-2,
  )
  candidates /= np.maximum(2.0 * magnitudes[..., :, None], 1e-12)
  best = np.argmax(magnitudes, axis=-1)
  flat_candidates = candidates.reshape(-1, 4, 4)
  flat_best = best.reshape(-1)
  result = flat_candidates[np.arange(len(flat_best)), flat_best].reshape(
    (*best.shape, 4)
  )
  result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
  result = np.where(result[..., :1] < 0.0, -result, result)
  return result


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
  quaternion = quaternion / np.maximum(
    np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
  )
  w, x, y, z = np.moveaxis(quaternion, -1, 0)
  return np.stack(
    (
      1.0 - 2.0 * (y * y + z * z),
      2.0 * (x * y - z * w),
      2.0 * (x * z + y * w),
      2.0 * (x * y + z * w),
      1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - x * w),
      2.0 * (x * z - y * w),
      2.0 * (y * z + x * w),
      1.0 - 2.0 * (x * x + y * y),
    ),
    axis=-1,
  ).reshape((*quaternion.shape[:-1], 3, 3))


def _matrix_to_rotation_vector(matrices: np.ndarray) -> np.ndarray:
  quaternion = _matrix_to_quaternion_wxyz(matrices)
  vector = quaternion[..., 1:]
  vector_norm = np.linalg.norm(vector, axis=-1)
  angle = 2.0 * np.arctan2(vector_norm, quaternion[..., 0])
  scale = np.full_like(angle, 2.0)
  np.divide(angle, vector_norm, out=scale, where=vector_norm > 1e-8)
  return vector * scale[..., None]


def _rotation_vector_to_matrix(vectors: np.ndarray) -> np.ndarray:
  angle = np.linalg.norm(vectors, axis=-1)
  half_angle = 0.5 * angle
  scale = np.full_like(angle, 0.5)
  np.divide(np.sin(half_angle), angle, out=scale, where=angle > 1e-8)
  quaternion = np.concatenate(
    (np.cos(half_angle)[..., None], vectors * scale[..., None]), axis=-1
  )
  return _quaternion_wxyz_to_matrix(quaternion)


def _to_local_transforms(
  positions: np.ndarray,
  rotations: np.ndarray,
  parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  local_positions = np.empty_like(positions)
  local_rotations = np.empty_like(rotations)
  for joint_index, parent_index in enumerate(parents):
    if parent_index < 0:
      local_positions[:, joint_index] = positions[:, joint_index]
      local_rotations[:, joint_index] = rotations[:, joint_index]
      continue
    parent_rotation_t = np.swapaxes(rotations[:, parent_index], -1, -2)
    local_positions[:, joint_index] = np.einsum(
      "nij,nj->ni",
      parent_rotation_t,
      positions[:, joint_index] - positions[:, parent_index],
    )
    local_rotations[:, joint_index] = np.einsum(
      "nij,njk->nik", parent_rotation_t, rotations[:, joint_index]
    )
  return local_positions, local_rotations


def _forward_kinematics(
  local_positions: np.ndarray,
  local_rotations: np.ndarray,
  parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  positions = np.empty_like(local_positions)
  rotations = np.empty_like(local_rotations)
  for joint_index, parent_index in enumerate(parents):
    if parent_index < 0:
      positions[:, joint_index] = local_positions[:, joint_index]
      rotations[:, joint_index] = local_rotations[:, joint_index]
      continue
    positions[:, joint_index] = positions[:, parent_index] + np.einsum(
      "nij,nj->ni", rotations[:, parent_index], local_positions[:, joint_index]
    )
    rotations[:, joint_index] = np.einsum(
      "nij,njk->nik", rotations[:, parent_index], local_rotations[:, joint_index]
    )
  return positions, rotations


def _yaw_from_rotation(rotation: np.ndarray) -> float:
  forward = rotation @ np.asarray([0.0, 1.0, 0.0])
  return float(np.arctan2(forward[1], forward[0]))


def _z_rotation(yaw_rad: float) -> np.ndarray:
  cosine = np.cos(yaw_rad)
  sine = np.sin(yaw_rad)
  return np.asarray(
    ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
  )


def _hermite_decay(
  offset: np.ndarray,
  velocity_offset: np.ndarray,
  frame_count: int,
  duration_s: float,
) -> np.ndarray:
  phase = np.linspace(0.0, 1.0, frame_count)
  h00 = 2.0 * phase**3 - 3.0 * phase**2 + 1.0
  h10 = phase**3 - 2.0 * phase**2 + phase
  expansion = (slice(None),) + (None,) * offset.ndim
  return (
    h00[expansion] * offset[None]
    + h10[expansion] * duration_s * velocity_offset[None]
  )


def _check_compatible(outgoing: BvhMotion, incoming: BvhMotion) -> None:
  if outgoing.joint_names != incoming.joint_names:
    raise ValueError("motions have different joint names or ordering")
  outgoing_parents = tuple(joint.parent for joint in outgoing.joints)
  incoming_parents = tuple(joint.parent for joint in incoming.joints)
  if outgoing_parents != incoming_parents:
    raise ValueError("motions have different skeleton hierarchies")
  if len(outgoing.positions_m) < 2 or len(incoming.positions_m) < 2:
    raise ValueError("inertialization requires at least two frames per motion")


def inertialize_bvh_transition(
  outgoing: BvhMotion,
  incoming: BvhMotion,
  *,
  output_fps: float,
  transition_duration_s: float = 0.2,
  anchor_joint: str = "Hips",
) -> InertializedTransition:
  """Join two clips using root alignment and velocity-aware offset decay.

  The incoming clip is first aligned in planar position and yaw at
  ``anchor_joint``. Pose and velocity residuals are then expressed in local
  skeleton coordinates and decayed with a cubic Hermite inertialization window.
  Forward kinematics is run after blending, so bone lengths remain intact.
  """

  _check_compatible(outgoing, incoming)
  if output_fps <= 0.0 or transition_duration_s <= 0.0:
    raise ValueError("output_fps and transition_duration_s must be positive")
  if anchor_joint not in outgoing.joint_names:
    raise ValueError(f"anchor joint {anchor_joint!r} is not in the skeleton")
  anchor_index = outgoing.joint_names.index(anchor_joint)
  parents = np.asarray([joint.parent for joint in outgoing.joints], dtype=np.int64)

  outgoing_positions = _convert_basis_positions(outgoing.positions_m)
  outgoing_rotations = _convert_basis_rotations(outgoing.rotations_world)
  incoming_positions = _convert_basis_positions(incoming.positions_m)
  incoming_rotations = _convert_basis_rotations(incoming.rotations_world)

  outgoing_anchor_rotation = outgoing_rotations[-1, anchor_index]
  incoming_anchor_rotation = incoming_rotations[0, anchor_index]
  alignment_yaw = _yaw_from_rotation(outgoing_anchor_rotation) - _yaw_from_rotation(
    incoming_anchor_rotation
  )
  alignment_rotation = _z_rotation(alignment_yaw)
  incoming_positions = np.einsum(
    "ij,nkj->nki", alignment_rotation, incoming_positions
  )
  incoming_rotations = np.einsum(
    "ij,nkjl->nkil", alignment_rotation, incoming_rotations
  )
  translation = (
    outgoing_positions[-1, anchor_index]
    - incoming_positions[0, anchor_index]
  )
  incoming_positions += translation

  outgoing_local_pos, outgoing_local_rot = _to_local_transforms(
    outgoing_positions, outgoing_rotations, parents
  )
  incoming_local_pos, incoming_local_rot = _to_local_transforms(
    incoming_positions, incoming_rotations, parents
  )
  transition_frames = min(
    len(incoming_local_pos),
    max(2, int(round(transition_duration_s * output_fps)) + 1),
  )
  actual_duration_s = (transition_frames - 1) / output_fps

  position_offset = outgoing_local_pos[-1] - incoming_local_pos[0]
  outgoing_velocity = (
    outgoing_local_pos[-1] - outgoing_local_pos[-2]
  ) * output_fps
  incoming_velocity = (
    incoming_local_pos[1] - incoming_local_pos[0]
  ) * output_fps
  position_velocity_offset = outgoing_velocity - incoming_velocity
  position_decay = _hermite_decay(
    position_offset,
    position_velocity_offset,
    transition_frames,
    actual_duration_s,
  )
  blended_local_pos = incoming_local_pos.copy()
  blended_local_pos[:transition_frames] += position_decay

  rotation_offset = _matrix_to_rotation_vector(
    np.einsum(
      "nij,njk->nik",
      outgoing_local_rot[-1],
      np.swapaxes(incoming_local_rot[0], -1, -2),
    )
  )
  outgoing_angular_velocity = _matrix_to_rotation_vector(
    np.einsum(
      "nij,njk->nik",
      outgoing_local_rot[-1],
      np.swapaxes(outgoing_local_rot[-2], -1, -2),
    )
  ) * output_fps
  incoming_angular_velocity = _matrix_to_rotation_vector(
    np.einsum(
      "nij,njk->nik",
      incoming_local_rot[1],
      np.swapaxes(incoming_local_rot[0], -1, -2),
    )
  ) * output_fps
  rotation_decay = _hermite_decay(
    rotation_offset,
    outgoing_angular_velocity - incoming_angular_velocity,
    transition_frames,
    actual_duration_s,
  )
  blended_local_rot = incoming_local_rot.copy()
  blended_local_rot[:transition_frames] = np.einsum(
    "fnij,fnjk->fnik",
    _rotation_vector_to_matrix(rotation_decay),
    incoming_local_rot[:transition_frames],
  )

  incoming_blended_pos, incoming_blended_rot = _forward_kinematics(
    blended_local_pos, blended_local_rot, parents
  )
  combined_positions = np.concatenate(
    (outgoing_positions[:-1], incoming_blended_pos), axis=0
  )
  combined_rotations = np.concatenate(
    (outgoing_rotations[:-1], incoming_blended_rot), axis=0
  )
  positions_bvh = _convert_basis_positions(combined_positions)
  rotations_bvh = _convert_basis_rotations(combined_rotations)
  frame_count = len(positions_bvh)
  motion = BvhMotion(
    path=Path(f"{outgoing.path.stem}__to__{incoming.path.stem}.bvh"),
    joints=outgoing.joints,
    positions_m=positions_bvh,
    rotations_world=rotations_bvh,
    frame_indices=np.arange(frame_count, dtype=np.int64),
    source_frame_count=frame_count,
    source_frame_time=1.0 / output_fps,
  )
  transition_start = len(outgoing_positions) - 1
  return InertializedTransition(
    motion=motion,
    transition_start_frame=transition_start,
    transition_end_frame=transition_start + transition_frames - 1,
    alignment_yaw_rad=alignment_yaw,
  )


def inertialize_bvh_sequence(
  motions: list[BvhMotion] | tuple[BvhMotion, ...],
  *,
  output_fps: float,
  transition_duration_s: float = 0.2,
  anchor_joint: str = "Hips",
) -> BvhMotion:
  """Compose a sequence by inertializing every adjacent clip boundary."""

  if not motions:
    raise ValueError("at least one motion is required")
  composed = motions[0]
  for incoming in motions[1:]:
    composed = inertialize_bvh_transition(
      composed,
      incoming,
      output_fps=output_fps,
      transition_duration_s=transition_duration_s,
      anchor_joint=anchor_joint,
    ).motion
  return composed
