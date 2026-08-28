from pathlib import Path

import numpy as np

from safe_mimic.motions.human_trajectory import (
  RobotTrajectory,
  compile_human_robot_intersection,
  load_compiled_human_trajectory,
  load_mjlab_robot_trajectory,
)
from safe_mimic.motions.soma_bvh import BvhJoint, BvhMotion


def _human_motion() -> BvhMotion:
  offsets = {
    "Hips": (0.0, 1.0, 0.0),
    "Spine2": (0.0, 1.25, 0.0),
    "Chest": (0.0, 1.5, 0.0),
    "Head": (0.0, 1.72, 0.0),
    "LeftArm": (-0.2, 1.48, 0.0),
    "LeftForeArm": (-0.45, 1.35, 0.0),
    "LeftHand": (-0.65, 1.22, 0.0),
    "RightArm": (0.2, 1.48, 0.0),
    "RightForeArm": (0.45, 1.35, 0.0),
    "RightHand": (0.65, 1.22, 0.0),
    "LeftLeg": (-0.1, 0.95, 0.0),
    "LeftShin": (-0.1, 0.52, 0.0),
    "LeftFoot": (-0.1, 0.0, 0.0),
    "LeftToeBase": (-0.1, 0.0, 0.18),
    "RightLeg": (0.1, 0.95, 0.0),
    "RightShin": (0.1, 0.52, 0.0),
    "RightFoot": (0.1, 0.0, 0.0),
    "RightToeBase": (0.1, 0.0, 0.18),
  }
  names = tuple(offsets)
  positions = np.asarray(
    [
      [(point[0] + root_x, point[1], point[2]) for point in offsets.values()]
      for root_x in (0.0, 1.0, 2.0)
    ],
    dtype=np.float64,
  )
  rotations = np.broadcast_to(
    np.eye(3), (len(positions), len(names), 3, 3)
  ).copy()
  joints = tuple(
    BvhJoint(
      name=name,
      parent=-1,
      offset=np.zeros(3),
      channels=(),
      channel_start=0,
    )
    for name in names
  )
  return BvhMotion(
    path=Path("synthetic.bvh"),
    joints=joints,
    positions_m=positions,
    rotations_world=rotations,
    frame_indices=np.arange(3),
    source_frame_count=3,
    source_frame_time=1.0,
  )


def test_compile_exact_perpendicular_robot_path_intersection(tmp_path: Path) -> None:
  robot = RobotTrajectory(
    times_s=np.asarray([0.0, 1.0, 2.0]),
    root_positions_w=np.asarray([[0.0, 0.0, 0.8], [1.0, 0.0, 0.8], [2.0, 0.0, 0.8]]),
    root_yaw_w=np.zeros(3),
  )
  compiled = compile_human_robot_intersection(
    _human_motion(),
    robot,
    intersection_time_s=1.0,
    intersection_phase=0.5,
    crossing_angle_rad=np.pi / 2.0,
  )

  assert np.allclose(compiled.times_s, (0.0, 1.0, 2.0))
  assert np.allclose(compiled.root_positions_w[:, :2], ((1, -1), (1, 0), (1, 1)))
  assert compiled.intersection_distance_m < 1e-10
  assert np.allclose(
    np.linalg.norm(compiled.capsule_quaternions_wxyz, axis=-1), 1.0
  )

  output = tmp_path / "compiled.npz"
  compiled.save(output)
  loaded = load_compiled_human_trajectory(output)
  assert loaded.joint_names == compiled.joint_names
  assert loaded.capsule_names == compiled.capsule_names
  assert loaded.intersection_distance_m < 1e-6
  assert np.allclose(loaded.capsule_centers_w, compiled.capsule_centers_w)


def test_load_mjlab_robot_root_trajectory(tmp_path: Path) -> None:
  positions = np.zeros((3, 2, 3), dtype=np.float32)
  positions[:, 0, 0] = (0.0, 0.5, 1.0)
  quaternions = np.zeros((3, 2, 4), dtype=np.float32)
  quaternions[..., 0] = 1.0
  path = tmp_path / "robot.npz"
  np.savez(path, fps=np.asarray([2.0]), body_pos_w=positions, body_quat_w=quaternions)

  robot = load_mjlab_robot_trajectory(path)

  assert np.allclose(robot.times_s, (0.0, 0.5, 1.0))
  assert np.allclose(robot.position_at(0.25), (0.25, 0.0, 0.0))
  assert robot.path_heading_at(0.5) == 0.0
