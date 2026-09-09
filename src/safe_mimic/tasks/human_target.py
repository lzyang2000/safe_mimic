"""Shared robot-relative targets for moving-human encounters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _yaw_from_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quaternion_wxyz.unbind(dim=-1)
  return torch.atan2(
    2.0 * (w * z + x * y),
    1.0 - 2.0 * (y * y + z * z),
  )


def robot_intersection_target_from_qpos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  robot_entity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return the robot's current reset pose as a fixed human action target.

  Event reset runs after the command has written qpos but before the next
  ``sim.forward()``. Reading qpos directly therefore avoids using stale derived
  link poses and gives packed training and online play identical placement
  semantics.
  """

  robot = env.scene[robot_entity]
  env_ids = env_ids.to(device=env.device, dtype=torch.long)
  q_addresses = robot.indexing.free_joint_q_adr
  if q_addresses.numel() < 7:
    raise ValueError(f"robot entity {robot_entity!r} has no free-joint pose")
  root_qpos = robot.data.data.qpos[
    env_ids[:, None], q_addresses[:7]
  ]
  position_w = root_qpos[:, :3]
  yaw_w = _yaw_from_quaternion(root_qpos[:, 3:7])
  # The target is intentionally fixed at scheduling time. The robot can escape
  # while the human continues toward the original action/intersection point.
  return position_w, yaw_w, yaw_w


__all__ = ["robot_intersection_target_from_qpos"]
