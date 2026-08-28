"""MDP terms for human-aware motion imitation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def human_capsule_vectors_b(
  env: ManagerBasedRlEnv,
  robot_entity: str,
  human_entity: str,
  max_distance: float,
  capsules_per_group: int | None = None,
  nearest_groups: int | None = None,
) -> torch.Tensor:
  """Privileged capsule-center vectors, optionally for nearest people only."""

  robot = env.scene[robot_entity]
  human = env.scene[human_entity]
  centers_w = human.data.geom_pos_w
  if nearest_groups is not None:
    group_ids = _nearest_capsule_group_ids(
      robot.data.root_link_pos_w,
      centers_w,
      capsules_per_group=capsules_per_group,
      nearest_groups=nearest_groups,
    )
    centers_w = _gather_capsule_groups(
      centers_w, group_ids, capsules_per_group=capsules_per_group
    )
  delta_w = centers_w - robot.data.root_link_pos_w[:, None, :]
  quat_w = robot.data.root_link_quat_w[:, None, :].expand(-1, delta_w.shape[1], -1)
  delta_b = quat_apply_inverse(quat_w, delta_w)
  return (delta_b / max_distance).flatten(start_dim=1)


def _nearest_capsule_group_ids(
  robot_positions_w: torch.Tensor,
  capsule_centers_w: torch.Tensor,
  *,
  capsules_per_group: int | None,
  nearest_groups: int,
) -> torch.Tensor:
  if capsules_per_group is None or capsules_per_group < 1:
    raise ValueError("capsules_per_group must be positive when selecting groups")
  if nearest_groups < 1:
    raise ValueError("nearest_groups must be positive")
  capsule_count = capsule_centers_w.shape[1]
  if capsule_count % capsules_per_group:
    raise ValueError("capsule count must be divisible by capsules_per_group")
  group_count = capsule_count // capsules_per_group
  selected_count = min(nearest_groups, group_count)
  # Crowd assets put their merged body/head proxy first in every person's
  # contiguous capsule block, making it a stable center for nearest selection.
  anchors_w = capsule_centers_w[:, ::capsules_per_group]
  distance_squared = (
    (anchors_w - robot_positions_w[:, None, :]).square().sum(dim=-1)
  )
  return torch.topk(
    distance_squared, k=selected_count, dim=1, largest=False, sorted=True
  ).indices


def _gather_capsule_groups(
  values: torch.Tensor,
  group_ids: torch.Tensor,
  *,
  capsules_per_group: int | None,
) -> torch.Tensor:
  if capsules_per_group is None:
    raise ValueError("capsules_per_group is required")
  env_count, capsule_count = values.shape[:2]
  group_count = capsule_count // capsules_per_group
  tail_shape = values.shape[2:]
  grouped = values.reshape(
    env_count, group_count, capsules_per_group, *tail_shape
  )
  gather_index = group_ids.reshape(
    env_count, len(group_ids[0]), 1, *(1 for _ in tail_shape)
  ).expand(-1, -1, capsules_per_group, *tail_shape)
  selected = torch.gather(grouped, 1, gather_index)
  return selected.reshape(
    env_count, len(group_ids[0]) * capsules_per_group, *tail_shape
  )


def _quaternion_z_axis(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quaternion_wxyz.unbind(dim=-1)
  return torch.stack(
    (
      2.0 * (x * z + w * y),
      2.0 * (y * z - w * x),
      1.0 - 2.0 * (x * x + y * y),
    ),
    dim=-1,
  )


def human_capsule_proximity_penalty(
  env: ManagerBasedRlEnv,
  robot_entity: str,
  human_entity: str,
  safe_clearance: float,
  robot_radius: float,
  capsules_per_group: int | None = None,
  nearest_groups: int | None = None,
) -> torch.Tensor:
  """Quadratic penalty near the human's planar capsule envelope.

  Planar distance is deliberate: avoidance should account for the occupied
  person-sized column even when a hand or foot happens to be above/below the
  robot root at the current frame.
  """

  robot = env.scene[robot_entity]
  human = env.scene[human_entity]
  geom_ids = human.indexing.geom_ids.to(dtype=torch.long)
  sizes = env.sim.model.geom_size[:, geom_ids]
  centers = human.data.geom_pos_w
  quaternions = human.data.geom_quat_w
  if nearest_groups is not None:
    group_ids = _nearest_capsule_group_ids(
      robot.data.root_link_pos_w,
      centers,
      capsules_per_group=capsules_per_group,
      nearest_groups=nearest_groups,
    )
    centers = _gather_capsule_groups(
      centers, group_ids, capsules_per_group=capsules_per_group
    )
    quaternions = _gather_capsule_groups(
      quaternions, group_ids, capsules_per_group=capsules_per_group
    )
    sizes = _gather_capsule_groups(
      sizes, group_ids, capsules_per_group=capsules_per_group
    )
  radii = sizes[..., 0]
  half_lengths = sizes[..., 1]
  axes = _quaternion_z_axis(quaternions)
  segment_half_xy = axes[..., :2] * half_lengths[..., None]
  start = centers[..., :2] - segment_half_xy
  segment = 2.0 * segment_half_xy
  robot_xy = robot.data.root_link_pos_w[:, None, :2]
  relative = robot_xy - start
  denominator = (segment * segment).sum(dim=-1).clamp_min(1e-12)
  fraction = ((relative * segment).sum(dim=-1) / denominator).clamp(0.0, 1.0)
  closest = start + fraction[..., None] * segment
  centerline_distance = torch.linalg.vector_norm(robot_xy - closest, dim=-1)
  surface_clearance = centerline_distance - radii - robot_radius
  clearance = surface_clearance.min(dim=1).values
  violation = torch.clamp((safe_clearance - clearance) / safe_clearance, min=0.0)
  return violation.square()


def command_metric(
  env: ManagerBasedRlEnv,
  command_name: str,
  metric_name: str,
) -> torch.Tensor:
  """Expose a command term's current per-environment tuning metric."""
  command = env.command_manager.get_term(command_name)
  try:
    return command.metrics[metric_name]
  except KeyError as error:
    raise KeyError(
      f"command {command_name!r} has no metric {metric_name!r}"
    ) from error


def reference_filter_clearance_violation(
  env: ManagerBasedRlEnv,
  command_name: str,
  safe_clearance_m: float,
  metric_name: str = "filter_minimum_clearance_m",
) -> torch.Tensor:
  """Positive shortfall below the requested reference-filter clearance."""
  minimum_clearance = command_metric(
    env,
    command_name=command_name,
    metric_name=metric_name,
  )
  return torch.clamp(safe_clearance_m - minimum_clearance, min=0.0)
