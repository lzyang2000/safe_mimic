"""MDP terms for human-aware motion imitation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
)
from safe_mimic.tasks.reference_filter import planar_capsule_geometry

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg


def _filtered_command(
  env: ManagerBasedRlEnv, command_name: str
) -> PlanarFilteredReplayMotionCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PlanarFilteredReplayMotionCommand):
    raise TypeError(
      f"command {command_name!r} must be a PlanarFilteredReplayMotionCommand"
    )
  return command


def filtered_planar_velocity_b(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Privileged safe planar-velocity target in the robot root frame."""
  command = _filtered_command(env, command_name)
  velocity_w = torch.nn.functional.pad(command.filtered_root_velocity_xy_w, (0, 1))
  velocity_b = quat_apply_inverse(command.robot_anchor_quat_w, velocity_w)
  return velocity_b[:, :2]


def filtered_joint_command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Privileged link-filtered joint position and velocity teacher."""
  command = _filtered_command(env, command_name)
  return torch.cat((command.filtered_joint_pos, command.filtered_joint_vel), dim=-1)


def avoidance_teacher_corrections(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Training-only planar and per-joint residuals from both CBF filters.

  The planar correction is rotated into the live robot root frame. Joint
  corrections are position residuals in action/joint order, so the resulting
  target can directly supervise a deployable actor head without exposing
  privileged obstacle geometry as an actor observation.
  """
  command = _filtered_command(env, command_name)
  raw_root_velocity_w = command._raw_body_lin_vel_w()[:, 0]  # noqa: SLF001
  planar_delta_w = torch.nn.functional.pad(
    command.filtered_root_velocity_xy_w - raw_root_velocity_w[:, :2],
    (0, 1),
  )
  planar_delta_b = quat_apply_inverse(
    command.robot_anchor_quat_w,
    planar_delta_w,
  )[:, :2]
  joint_delta = (
    command.filtered_joint_pos - command._raw_joint_pos()  # noqa: SLF001
  )
  return torch.cat((planar_delta_b, joint_delta), dim=-1)


def raw_motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Deployment-consistent actor observation of the anchor orientation.

  Identical to mjlab's ``motion_anchor_ori_b``, except the reference pose is
  the RAW (unfiltered) anchor rather than ``anchor_pos_w``/``anchor_quat_w``.
  At deployment only the raw reference and the learned adjuster exist, so the
  actor must not observe the privileged whole-body FK correction that
  ``anchor_quat_w`` carries in whole-body propagation mode; the critic,
  rewards, and terminations keep the filtered anchor. The live alignment
  already yaw-aligns the raw anchor to the robot each step, so this term
  equals mjlab's whenever no correction is active.
  """
  command = _filtered_command(env, command_name)
  anchor = command.motion_anchor_body_index
  raw_quat = command._raw_body_quat_w()[:, anchor]  # noqa: SLF001

  # The relative orientation does not depend on positions; skip the raw
  # position cloud (mjlab's term passes the anchor position, unused here).
  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    None,
    raw_quat,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def avoidance_conditioning_noise(
  env: ManagerBasedRlEnv,
  size: int,
) -> torch.Tensor:
  """Stored training noise for a robust deployable correction bottleneck."""
  return 2.0 * torch.rand((env.num_envs, size), device=env.device) - 1.0


def safe_planar_velocity_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward root XY velocity tracking of the privileged safe teacher."""
  command = _filtered_command(env, command_name)
  error = (
    (
      command.robot.data.root_link_lin_vel_w[:, :2]
      - command.filtered_root_velocity_xy_w
    )
    .square()
    .sum(dim=-1)
  )
  return torch.exp(-error / std**2)


def filtered_joint_position_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward tracking the privileged collision-aware joint reference."""
  command = _filtered_command(env, command_name)
  error = (
    (command.robot.data.joint_pos - command.filtered_joint_pos).square().mean(dim=-1)
  )
  return torch.exp(-error / std**2)


def active_correction_tracking(
  joint_error: torch.Tensor,
  weights: torch.Tensor,
  std: float,
) -> torch.Tensor:
  """Pure reward core for active-correction joint tracking.

  ``joint_error`` and ``weights`` share shape (num_envs, num_joints): the
  per-joint (robot - filtered) position residual and a 0/1 activation mask,
  respectively. Joints with zero weight are excluded from both the squared-
  error sum and the normalizing count, so they cannot dilute or inflate the
  reward; an environment with no active joint gets exactly zero reward
  regardless of ``joint_error``.
  """
  weight_sum = weights.sum(dim=-1)
  weighted_squared_error = (weights * joint_error.square()).sum(dim=-1)
  mean_squared_error = weighted_squared_error / weight_sum.clamp(min=1.0)
  reward = torch.exp(-mean_squared_error / std**2)
  return reward * (weight_sum > 0).to(reward.dtype)


def active_correction_joint_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.2,
  activation_threshold_rad: float = 0.05,
) -> torch.Tensor:
  """Reward joint tracking only where the CBF teacher actively corrects.

  ``filtered_joint_position_tracking_exp`` averages error over all 29 joints
  every step, so a correction confined to a couple of arm joints is diluted
  to near-zero signal. This term instead gates each joint's contribution by
  whether the teacher's own residual (filtered - raw) exceeds
  ``activation_threshold_rad``, rewarding precise tracking of the corrected
  pose exactly where the filter is intervening.
  """
  command = _filtered_command(env, command_name)
  teacher_residual = (
    command.filtered_joint_pos - command._raw_joint_pos()  # noqa: SLF001
  )
  weights = (teacher_residual.abs() >= activation_threshold_rad).to(
    teacher_residual.dtype
  )
  joint_error = command.robot.data.joint_pos - command.filtered_joint_pos
  return active_correction_tracking(joint_error, weights, std)


def safe_planar_freeze_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  minimum_target_speed: float = 0.1,
  minimum_progress_fraction: float = 0.25,
) -> torch.Tensor:
  """Penalize freezing when the safe teacher requests meaningful motion."""
  command = _filtered_command(env, command_name)
  target = command.filtered_root_velocity_xy_w
  target_speed = torch.linalg.vector_norm(target, dim=-1)
  direction = target / target_speed[:, None].clamp_min(1.0e-6)
  achieved = (command.robot.data.root_link_lin_vel_w[:, :2] * direction).sum(dim=-1)
  shortfall = torch.clamp(minimum_progress_fraction * target_speed - achieved, min=0.0)
  active = target_speed >= minimum_target_speed
  return torch.where(active, shortfall.square(), torch.zeros_like(shortfall))


def safe_planar_progress_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  minimum_target_speed: float = 0.1,
  normalization_speed: float = 0.5,
) -> torch.Tensor:
  """Reward actual root motion in the CBF teacher's escape direction."""
  command = _filtered_command(env, command_name)
  target = command.filtered_root_velocity_xy_w
  target_speed = torch.linalg.vector_norm(target, dim=-1)
  direction = target / target_speed[:, None].clamp_min(1.0e-6)
  progress_speed = (command.robot.data.root_link_lin_vel_w[:, :2] * direction).sum(
    dim=-1
  )
  normalized_progress = torch.clamp(
    progress_speed / normalization_speed,
    min=-1.0,
    max=1.0,
  )
  return torch.where(
    target_speed >= minimum_target_speed,
    normalized_progress,
    torch.zeros_like(normalized_progress),
  )


def urgency_weighted_outward_progress(
  clearance_m: torch.Tensor,
  closing_speed_mps: torch.Tensor,
  outward_speed_mps: torch.Tensor,
  *,
  ttc_horizon_s: float,
  normalization_speed: float,
  min_closing_speed_mps: float,
) -> torch.Tensor:
  """Scale outward planar progress by privileged time-to-contact urgency."""
  if ttc_horizon_s <= 0.0:
    raise ValueError("ttc_horizon_s must be positive")
  if normalization_speed <= 0.0:
    raise ValueError("normalization_speed must be positive")
  closing = closing_speed_mps > min_closing_speed_mps
  time_to_contact = torch.where(
    closing,
    torch.clamp(clearance_m, min=0.0) / closing_speed_mps.clamp_min(1.0e-6),
    torch.full_like(clearance_m, torch.inf),
  )
  urgency = torch.clamp(1.0 - time_to_contact / ttc_horizon_s, min=0.0, max=1.0)
  progress = torch.clamp(outward_speed_mps / normalization_speed, min=-1.0, max=1.0)
  return urgency * progress


def urgent_escape_progress_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  human_entity: str,
  ttc_horizon_s: float = 2.5,
  normalization_speed: float = 0.5,
  min_closing_speed_mps: float = 0.3,
) -> torch.Tensor:
  """Reward early outward root motion while a human is actively closing.

  The existing escape terms only pay once the CBF teacher requests motion;
  this term pays in the short window after an approach becomes threatening,
  using the reference filter's own obstacle-velocity estimate so geometry and
  velocity share one privileged view. The robot radius and vertical gate are
  read from the command's planar filter so the reward cannot desync from a
  retuned filter.
  """
  command = _filtered_command(env, command_name)
  planar_filter = command.cfg.planar_filter
  entity_slice = command.obstacle_entity_slices[human_entity]
  human = env.scene[human_entity]
  geom_ids = human.indexing.geom_ids.to(dtype=torch.long)
  centers_w = human.data.geom_pos_w
  quaternions_w = human.data.geom_quat_w
  sizes = env.sim.model.geom_size[:, geom_ids]
  robot = command.robot
  # Deliberately the live robot root, matching the planar filter's own view.
  robot_pos_w = robot.data.root_link_pos_w
  closest_xy, clearance, active = planar_capsule_geometry(
    robot_pos_w,
    centers_w,
    quaternions_w,
    sizes,
    robot_radius_m=planar_filter.robot_radius_m,
    vertical_gate_m=planar_filter.vertical_gate_m,
  )
  masked_clearance = torch.where(
    active, clearance, torch.full_like(clearance, torch.inf)
  )
  nearest_clearance, nearest_ids = masked_clearance.min(dim=1)
  has_active = torch.isfinite(nearest_clearance)

  gather_xy = nearest_ids[:, None, None].expand(-1, 1, 2)
  nearest_closest_xy = torch.gather(closest_xy, 1, gather_xy).squeeze(1)
  separation_xy = robot_pos_w[:, :2] - nearest_closest_xy
  normal_xy = separation_xy / torch.linalg.vector_norm(
    separation_xy, dim=-1, keepdim=True
  ).clamp_min(1.0e-6)

  # The velocity rows share the geometry tensors' entity/geom order, so the
  # nearest-capsule index from this entity's block selects its own velocity.
  obstacle_velocity_xy = command.obstacle_velocities_w[:, entity_slice, :2]
  nearest_velocity_xy = torch.gather(obstacle_velocity_xy, 1, gather_xy).squeeze(1)
  robot_velocity_xy = robot.data.root_link_lin_vel_w[:, :2]
  # Positive when the gap shrinks: the obstacle gains ground along the
  # capsule-to-robot normal faster than the robot yields it.
  closing_speed = ((nearest_velocity_xy - robot_velocity_xy) * normal_xy).sum(dim=-1)
  outward_speed = (robot_velocity_xy * normal_xy).sum(dim=-1)
  reward = urgency_weighted_outward_progress(
    torch.where(has_active, nearest_clearance, torch.zeros_like(nearest_clearance)),
    closing_speed,
    outward_speed,
    ttc_horizon_s=ttc_horizon_s,
    normalization_speed=normalization_speed,
    min_closing_speed_mps=min_closing_speed_mps,
  )
  return torch.where(has_active, reward, torch.zeros_like(reward))


def survival_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Small per-step reward that makes early failure strictly undesirable."""
  return torch.ones(env.num_envs, device=env.device)


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


def human_capsule_proximity_penalty(
  env: ManagerBasedRlEnv,
  robot_entity: str,
  human_entity: str,
  safe_clearance: float,
  robot_radius: float,
  capsules_per_group: int | None = None,
  nearest_groups: int | None = None,
  vertical_gate_m: float = 1.0,
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
  _, surface_clearance, active = planar_capsule_geometry(
    robot.data.root_link_pos_w,
    centers,
    quaternions,
    sizes,
    robot_radius_m=robot_radius,
    vertical_gate_m=vertical_gate_m,
  )
  active_clearance = torch.where(
    active,
    surface_clearance,
    torch.full_like(surface_clearance, torch.inf),
  )
  clearance = active_clearance.min(dim=1).values
  violation = torch.clamp((safe_clearance - clearance) / safe_clearance, min=0.0)
  return violation.square()


def human_capsule_collision(
  env: ManagerBasedRlEnv,
  robot_entity: str,
  human_entity: str,
  robot_radius: float,
  collision_margin: float = 0.0,
  vertical_gate_m: float = 1.0,
  capsules_per_group: int | None = None,
  nearest_groups: int | None = None,
) -> torch.Tensor:
  """Terminate when a root-envelope proxy overlaps an active human capsule."""
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
  _, clearance, active = planar_capsule_geometry(
    robot.data.root_link_pos_w,
    centers,
    quaternions,
    sizes,
    robot_radius_m=robot_radius,
    vertical_gate_m=vertical_gate_m,
  )
  return torch.any(active & (clearance < collision_margin), dim=1)


def capsule_link_surface_clearances(
  link_positions_w: torch.Tensor,
  capsule_centers_w: torch.Tensor,
  capsule_quaternions_wxyz: torch.Tensor,
  capsule_sizes: torch.Tensor,
  *,
  link_radius_m: float,
) -> torch.Tensor:
  """Return 3D link-sphere to capsule-surface clearances for every pair."""

  w, x, y, z = capsule_quaternions_wxyz.unbind(dim=-1)
  capsule_axes_w = torch.stack(
    (
      2.0 * (x * z + w * y),
      2.0 * (y * z - w * x),
      1.0 - 2.0 * (x * x + y * y),
    ),
    dim=-1,
  )
  half_segments_w = capsule_axes_w * capsule_sizes[..., 1, None]
  starts_w = capsule_centers_w - half_segments_w
  segments_w = 2.0 * half_segments_w
  relative_w = link_positions_w[:, :, None] - starts_w[:, None]
  segment_w = segments_w[:, None]
  denominator = segment_w.square().sum(dim=-1).clamp_min(1.0e-12)
  fraction = ((relative_w * segment_w).sum(dim=-1) / denominator).clamp(0.0, 1.0)
  closest_w = starts_w[:, None] + fraction[..., None] * segment_w
  distance = torch.linalg.vector_norm(
    link_positions_w[:, :, None] - closest_w,
    dim=-1,
  )
  return distance - capsule_sizes[:, None, :, 0] - link_radius_m


class _HumanCapsuleLinkTerm(ManagerTermBase):
  """Cache scene indices for batched robot-link/human-capsule queries."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    params = cfg.params
    self.robot_entity_name = str(params["robot_entity"])
    self.human_entity_name = str(params["human_entity"])
    self.link_radius_m = float(params["link_radius"])
    self.capsules_per_group = params.get("capsules_per_group")
    self.nearest_groups = params.get("nearest_groups")
    robot = env.scene[self.robot_entity_name]
    requested_names = tuple(params["robot_link_names"])
    link_ids, resolved_names = robot.find_bodies(
      requested_names,
      preserve_order=True,
    )
    if tuple(resolved_names) != requested_names:
      raise ValueError("clearance link order does not match its configuration")
    self._link_body_ids = torch.tensor(
      link_ids,
      dtype=torch.long,
      device=env.device,
    )
    human = env.scene[self.human_entity_name]
    self._human_geom_ids = human.indexing.geom_ids.to(dtype=torch.long)

  def _minimum_clearance(self) -> torch.Tensor:
    robot = self._env.scene[self.robot_entity_name]
    human = self._env.scene[self.human_entity_name]
    link_positions_w = robot.data.body_link_pos_w[:, self._link_body_ids]
    centers_w = human.data.geom_pos_w
    quaternions_wxyz = human.data.geom_quat_w
    sizes = self._env.sim.model.geom_size[:, self._human_geom_ids]
    if self.nearest_groups is not None:
      group_ids = _nearest_capsule_group_ids(
        robot.data.root_link_pos_w,
        centers_w,
        capsules_per_group=self.capsules_per_group,
        nearest_groups=int(self.nearest_groups),
      )
      centers_w = _gather_capsule_groups(
        centers_w,
        group_ids,
        capsules_per_group=self.capsules_per_group,
      )
      quaternions_wxyz = _gather_capsule_groups(
        quaternions_wxyz,
        group_ids,
        capsules_per_group=self.capsules_per_group,
      )
      sizes = _gather_capsule_groups(
        sizes,
        group_ids,
        capsules_per_group=self.capsules_per_group,
      )
    clearances = capsule_link_surface_clearances(
      link_positions_w,
      centers_w,
      quaternions_wxyz,
      sizes,
      link_radius_m=self.link_radius_m,
    )
    active_capsules = centers_w[..., 2] > -50.0
    return torch.where(
      active_capsules[:, None],
      clearances,
      torch.full_like(clearances, torch.inf),
    ).amin(dim=(1, 2))


class HumanCapsuleLinkClearance(_HumanCapsuleLinkTerm):
  """Measure minimum 3D clearance from configured robot links to a human."""

  def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
    del env
    return self._minimum_clearance()


class HumanCapsuleLinkCollision(_HumanCapsuleLinkTerm):
  """Terminate when a configured robot-link sphere reaches a human capsule."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.collision_margin_m = float(cfg.params.get("collision_margin", 0.0))

  def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
    del env
    return self._minimum_clearance() < self.collision_margin_m


_LIMB_JOINT_TOKENS: dict[str, tuple[str, ...]] = {
  "wrist": ("shoulder", "elbow", "wrist"),
  "ankle": ("hip", "knee", "ankle"),
}


def limb_joint_mask(joint_names: tuple[str, ...], body_name: str) -> torch.Tensor:
  """Boolean mask of the joints that move ``body_name``'s limb.

  Wrist links map to that side's shoulder/elbow/wrist joints, ankle links to
  its hip/knee/ankle joints. The side prefix (``left_``/``right_``) is read
  from the body name, so a correction on one arm never loosens the other.
  """
  side = body_name.split("_", 1)[0]
  limb = next((key for key in _LIMB_JOINT_TOKENS if key in body_name), None)
  if side not in ("left", "right") or limb is None:
    raise ValueError(
      f"{body_name!r} is not a sided wrist or ankle link; cannot map it to joints"
    )
  tokens = _LIMB_JOINT_TOKENS[limb]
  return torch.tensor(
    [
      name.startswith(f"{side}_") and any(token in name for token in tokens)
      for name in joint_names
    ],
    dtype=torch.bool,
  )


def bad_motion_body_pos_z_only_filter_gated(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  loosened_threshold: float,
  activation_threshold_rad: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """``bad_motion_body_pos_z_only`` that loosens per limb while the filter acts.

  The stock z-only check terminates when an end effector sits ``threshold``
  above or below its reference. Under the FK-propagated filtered reference an
  arm correction can move the reference wrist by more than that in a few
  steps, so the robot is terminated for lagging behind the very correction it
  is asked to follow (joint@16k: 124/126 ``ee_body_pos`` trips were wrists).
  Here each body uses ``loosened_threshold`` while any teacher joint residual
  on its own limb is at least ``activation_threshold_rad`` (the same gate as
  the active-joint reward) and ``threshold`` otherwise, so the strict check
  still guards falls and idle-limb drift.
  """
  command = _filtered_command(env, command_name)
  names = tuple(command.cfg.body_names)
  body_ids = [names.index(name) for name in body_names]
  masks = torch.stack(
    [limb_joint_mask(tuple(command.robot.joint_names), name) for name in body_names]
  ).to(device=command.filtered_joint_pos.device)
  teacher_residual = command.filtered_joint_pos - command._raw_joint_pos()  # noqa: SLF001
  joint_active = teacher_residual.abs() >= activation_threshold_rad
  limb_active = (joint_active[:, None, :] & masks[None]).any(dim=-1)
  bound = torch.where(
    limb_active,
    torch.full_like(limb_active, loosened_threshold, dtype=teacher_residual.dtype),
    torch.full_like(limb_active, threshold, dtype=teacher_residual.dtype),
  )
  z_error = torch.abs(
    command.body_pos_relative_w[:, body_ids, -1]
    - command.robot_body_pos_w[:, body_ids, -1]
  )
  return torch.any(z_error > bound, dim=-1)


def bad_motion_body_pos_z_only_lag_aware(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  lag_time_s: float,
  max_threshold: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """``bad_motion_body_pos_z_only`` whose bound widens while the target moves.

  The stock check ends the episode when an end effector is ``threshold`` above
  or below its reference height. It therefore punishes LAG, not divergence:
  a fast ballet arm sweep or a filter-driven arm correction moves the
  reference wrist faster than the robot follows and the 0.25 m line is crossed
  although the arm ends up in the right place (ballet@30k: 141/1024 trips on
  its own reference, six times more with a crowd). Here each body's bound is
  ``threshold + lag_time_s * |reference vertical speed|`` capped at
  ``max_threshold``: a moving target buys the robot ``lag_time_s`` of latency,
  a still target keeps the strict bound, so the pressure to ARRIVE at the
  corrected pose is intact. Unlike the filter-gated variant, nothing loosens
  merely because the filter is active.
  """
  command = _filtered_command(env, command_name)
  names = tuple(command.cfg.body_names)
  body_ids = [names.index(name) for name in body_names]
  z_error = torch.abs(
    command.body_pos_relative_w[:, body_ids, -1]
    - command.robot_body_pos_w[:, body_ids, -1]
  )
  target_vz = torch.abs(command.body_lin_vel_w[:, body_ids, -1])
  bound = torch.clamp(threshold + lag_time_s * target_vz, max=max_threshold)
  return torch.any(z_error > bound, dim=-1)


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


def reference_filter_clearance_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  safe_clearance_m: float,
  metric_name: str = "link_filter_minimum_clearance_m",
) -> torch.Tensor:
  """Quadratic dense penalty as actual robot links enter the safety margin."""
  if safe_clearance_m <= 0.0:
    raise ValueError("safe_clearance_m must be positive")
  minimum_clearance = command_metric(
    env,
    command_name=command_name,
    metric_name=metric_name,
  )
  normalized_shortfall = torch.clamp(
    (safe_clearance_m - minimum_clearance) / safe_clearance_m,
    min=0.0,
    max=1.0,
  )
  return normalized_shortfall.square()
