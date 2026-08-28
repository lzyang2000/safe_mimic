from pathlib import Path

import numpy as np

from safe_mimic.motions.inertialization import (
  inertialize_bvh_sequence,
  inertialize_bvh_transition,
)
from safe_mimic.motions.soma_bvh import BvhJoint, BvhMotion


def _yaw_rotation(angle: float) -> np.ndarray:
  cosine = np.cos(angle)
  sine = np.sin(angle)
  return np.asarray(
    ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
  )


def _make_motion(
  name: str,
  root_x: tuple[float, ...],
  root_yaw: float,
  hand_angles: tuple[float, ...],
) -> BvhMotion:
  names = ("Root", "Hips", "Hand", "Tip")
  parents = (-1, 0, 1, 2)
  joints = tuple(
    BvhJoint(
      name=joint_name,
      parent=parent,
      offset=np.zeros(3),
      channels=(),
      channel_start=0,
    )
    for joint_name, parent in zip(names, parents, strict=True)
  )
  frame_count = len(root_x)
  positions = np.zeros((frame_count, len(names), 3), dtype=np.float64)
  rotations = np.zeros((frame_count, len(names), 3, 3), dtype=np.float64)
  root_rotation = _yaw_rotation(root_yaw)
  for frame, (x_position, hand_angle) in enumerate(
    zip(root_x, hand_angles, strict=True)
  ):
    positions[frame, 0] = (x_position, 1.0, 0.0)
    rotations[frame, 0] = root_rotation
    positions[frame, 1] = positions[frame, 0]
    rotations[frame, 1] = root_rotation
    positions[frame, 2] = positions[frame, 1] + root_rotation @ np.asarray(
      [0.5, 0.0, 0.0]
    )
    rotations[frame, 2] = root_rotation @ _yaw_rotation(hand_angle)
    positions[frame, 3] = positions[frame, 2] + rotations[frame, 2] @ np.asarray(
      [0.5, 0.0, 0.0]
    )
    rotations[frame, 3] = rotations[frame, 2]
  return BvhMotion(
    path=Path(f"{name}.bvh"),
    joints=joints,
    positions_m=positions,
    rotations_world=rotations,
    frame_indices=np.arange(frame_count),
    source_frame_count=frame_count,
    source_frame_time=0.1,
  )


def test_skeleton_inertialization_aligns_boundary_and_preserves_bones() -> None:
  outgoing = _make_motion("out", (0.0, 0.1, 0.2), 0.0, (0.0, 0.3, 0.6))
  incoming = _make_motion(
    "in", (2.0, 2.1, 2.2, 2.3), np.pi / 2.0, (-0.8, -0.6, -0.4, -0.2)
  )

  transition = inertialize_bvh_transition(
    outgoing,
    incoming,
    output_fps=10.0,
    transition_duration_s=0.2,
  )
  motion = transition.motion
  boundary = transition.transition_start_frame

  assert np.allclose(motion.positions_m[boundary], outgoing.positions_m[-1])
  assert np.allclose(motion.rotations_world[boundary], outgoing.rotations_world[-1])
  assert transition.transition_end_frame == boundary + 2
  bone_lengths = np.linalg.norm(
    motion.positions_m[:, 3] - motion.positions_m[:, 2], axis=-1
  )
  assert np.allclose(bone_lengths, 0.5)
  identity = np.einsum(
    "...ji,...jk->...ik", motion.rotations_world, motion.rotations_world
  )
  assert np.allclose(identity, np.eye(3), atol=1e-7)


def test_inertialize_sequence_composes_more_than_two_clips() -> None:
  first = _make_motion("one", (0.0, 0.1), 0.0, (0.0, 0.1))
  second = _make_motion("two", (2.0, 2.1), 0.2, (0.3, 0.4))
  third = _make_motion("three", (-1.0, -0.9), -0.3, (-0.2, -0.1))

  composed = inertialize_bvh_sequence(
    (first, second, third), output_fps=10.0, transition_duration_s=0.1
  )

  assert len(composed.positions_m) == 4
  assert composed.source_frame_time == 0.1
  assert np.isfinite(composed.positions_m).all()
