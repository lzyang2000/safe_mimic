from types import SimpleNamespace

import torch

from safe_mimic.tasks.human_target import robot_intersection_target_from_qpos


def test_robot_intersection_target_reads_reset_qpos_for_subset() -> None:
  yaw = torch.tensor([0.0, 0.6, -1.2])
  qpos = torch.zeros((3, 7))
  qpos[:, :3] = torch.tensor(
    [[10.0, 20.0, 0.8], [1.0, 2.0, 0.9], [-3.0, 4.0, 1.0]]
  )
  qpos[:, 3] = torch.cos(0.5 * yaw)
  qpos[:, 6] = torch.sin(0.5 * yaw)
  robot = SimpleNamespace(
    indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
    data=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
  )
  # Deliberately unrelated derived/source data: placement must only use the
  # just-written free-joint qpos during reset.
  env = SimpleNamespace(
    device="cpu",
    scene={"robot": robot},
    command_manager=SimpleNamespace(source_position=torch.full((3, 3), 99.0)),
  )

  position, target_yaw, path_heading = robot_intersection_target_from_qpos(
    env,
    torch.tensor([2, 0]),
    "robot",
  )

  torch.testing.assert_close(position, qpos[[2, 0], :3])
  torch.testing.assert_close(target_yaw, yaw[[2, 0]])
  torch.testing.assert_close(path_heading, yaw[[2, 0]])
