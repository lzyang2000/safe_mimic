"""Deterministic planar reference filtering against human capsules."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _limit_vector_norm(vectors: torch.Tensor, limit: float) -> torch.Tensor:
  norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
  scale = torch.clamp(limit / norms.clamp_min(1e-8), max=1.0)
  return vectors * scale


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


def planar_capsule_geometry(
  robot_position_w: torch.Tensor,
  capsule_centers_w: torch.Tensor,
  capsule_quaternions_w: torch.Tensor,
  capsule_sizes: torch.Tensor,
  *,
  robot_radius_m: float,
  vertical_gate_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return closest planar points, surface clearance, and active-mask.

  MuJoCo capsules use ``size[..., 0]`` as radius and ``size[..., 1]`` as
  centerline half-length. A vertical overlap gate rejects inactive crowd geoms
  parked far below the scene without discarding articulated horizontal limbs.
  """
  axes_w = _quaternion_z_axis(capsule_quaternions_w)
  radii = capsule_sizes[..., 0]
  half_lengths = capsule_sizes[..., 1]
  segment_half_w = axes_w * half_lengths[..., None]
  starts_w = capsule_centers_w - segment_half_w
  segments_w = 2.0 * segment_half_w

  robot_xy = robot_position_w[:, None, :2]
  segment_xy = segments_w[..., :2]
  relative_xy = robot_xy - starts_w[..., :2]
  denominator = (segment_xy * segment_xy).sum(dim=-1).clamp_min(1e-12)
  fraction = ((relative_xy * segment_xy).sum(dim=-1) / denominator).clamp(0.0, 1.0)
  closest_xy = starts_w[..., :2] + fraction[..., None] * segment_xy
  centerline_distance = torch.linalg.vector_norm(robot_xy - closest_xy, dim=-1)
  clearance = centerline_distance - radii - robot_radius_m

  ends_z = starts_w[..., 2] + segments_w[..., 2]
  lower_z = torch.minimum(starts_w[..., 2], ends_z) - radii
  upper_z = torch.maximum(starts_w[..., 2], ends_z) + radii
  robot_z = robot_position_w[:, None, 2]
  active = (lower_z <= robot_z + vertical_gate_m) & (
    upper_z >= robot_z - vertical_gate_m
  )
  return closest_xy, clearance, active


@dataclass(kw_only=True)
class PlanarCbfReferenceFilterCfg:
  """Tunable parameters for the privileged planar reference filter."""

  safe_clearance_m: float = 0.1
  robot_radius_m: float = 0.35
  activation_clearance_m: float = 1.5
  vertical_gate_m: float = 1.0
  cbf_alpha: float = 2.0
  recovery_gain: float = 0.8
  max_recovery_speed_mps: float = 0.4
  max_planar_speed_mps: float = 2.0
  max_intervention_speed_mps: float = 1.5
  nearest_obstacles: int = 16
  projection_iterations: int = 4
  obstacle_velocity_limit_mps: float = 3.0
  obstacle_velocity_smoothing: float = 0.5
  obstacle_velocity_decay: float = 0.5

  def __post_init__(self) -> None:
    positive_values = {
      "safe_clearance_m": self.safe_clearance_m,
      "robot_radius_m": self.robot_radius_m,
      "activation_clearance_m": self.activation_clearance_m,
      "vertical_gate_m": self.vertical_gate_m,
      "cbf_alpha": self.cbf_alpha,
      "max_recovery_speed_mps": self.max_recovery_speed_mps,
      "max_planar_speed_mps": self.max_planar_speed_mps,
      "max_intervention_speed_mps": self.max_intervention_speed_mps,
      "obstacle_velocity_limit_mps": self.obstacle_velocity_limit_mps,
      "obstacle_velocity_decay": self.obstacle_velocity_decay,
    }
    for name, value in positive_values.items():
      if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if self.recovery_gain < 0.0:
      raise ValueError("recovery_gain must be non-negative")
    if self.nearest_obstacles < 1:
      raise ValueError("nearest_obstacles must be positive")
    if self.projection_iterations < 1:
      raise ValueError("projection_iterations must be positive")
    if not 0.0 <= self.obstacle_velocity_smoothing < 1.0:
      raise ValueError("obstacle_velocity_smoothing must lie in [0, 1)")


@dataclass
class PlanarFilterResult:
  velocity_w: torch.Tensor
  intervention_w: torch.Tensor
  minimum_clearance_m: torch.Tensor
  maximum_cbf_violation_mps: torch.Tensor


@dataclass(kw_only=True)
class LinkCbfReferenceFilterCfg:
  """Tunable differential-IK filter for individual robot links."""

  body_names: tuple[str, ...] = (
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
  )
  safe_clearance_m: float = 0.10
  link_radius_m: float = 0.10
  activation_clearance_m: float = 0.80
  cbf_alpha: float = 4.0
  joint_recovery_gain: float = 3.0
  max_joint_velocity_correction_rps: float = 1.5
  max_joint_position_residual_rad: float = 0.75
  max_arm_joint_position_residual_rad: float = 1.5
  arm_joint_name_tokens: tuple[str, ...] = ("shoulder", "elbow", "wrist")
  posture_joint_name_tokens: tuple[str, ...] = ("shoulder", "elbow", "wrist")
  max_link_velocity_correction_mps: float = 0.8
  jacobian_damping: float = 0.05
  standing_posture_gain: float = 2.0
  max_standing_velocity_correction_rps: float = 1.5
  posture_lookahead_s: float = 0.5
  posture_activation_clearance_m: float = 0.8
  posture_hold_s: float = 1.0
  posture_motion_cost: float = 0.002
  standing_projection_iterations: int = 3

  def __post_init__(self) -> None:
    if not self.body_names:
      raise ValueError("body_names must not be empty")
    positive_values = {
      "safe_clearance_m": self.safe_clearance_m,
      "link_radius_m": self.link_radius_m,
      "activation_clearance_m": self.activation_clearance_m,
      "cbf_alpha": self.cbf_alpha,
      "joint_recovery_gain": self.joint_recovery_gain,
      "max_joint_velocity_correction_rps": (self.max_joint_velocity_correction_rps),
      "max_joint_position_residual_rad": self.max_joint_position_residual_rad,
      "max_arm_joint_position_residual_rad": (self.max_arm_joint_position_residual_rad),
      "max_link_velocity_correction_mps": (self.max_link_velocity_correction_mps),
      "jacobian_damping": self.jacobian_damping,
      "standing_posture_gain": self.standing_posture_gain,
      "max_standing_velocity_correction_rps": (
        self.max_standing_velocity_correction_rps
      ),
      "posture_lookahead_s": self.posture_lookahead_s,
      "posture_activation_clearance_m": (self.posture_activation_clearance_m),
      "posture_hold_s": self.posture_hold_s,
      "posture_motion_cost": self.posture_motion_cost,
    }
    for name, value in positive_values.items():
      if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not self.arm_joint_name_tokens:
      raise ValueError("arm_joint_name_tokens must not be empty")
    if not self.posture_joint_name_tokens:
      raise ValueError("posture_joint_name_tokens must not be empty")
    if self.posture_activation_clearance_m < self.safe_clearance_m:
      raise ValueError(
        "posture_activation_clearance_m must be at least safe_clearance_m"
      )
    if self.standing_projection_iterations < 1:
      raise ValueError("standing_projection_iterations must be positive")


@dataclass
class LinkFilterResult:
  link_velocity_correction_w: torch.Tensor
  minimum_clearance_m: torch.Tensor
  nearest_obstacle_ids: torch.Tensor
  required_outward_speed_mps: torch.Tensor
  normals_w: torch.Tensor
  active: torch.Tensor


def joint_position_residual_limits(
  cfg: LinkCbfReferenceFilterCfg,
  joint_names: tuple[str, ...],
  device: str | torch.device,
) -> torch.Tensor:
  """Return per-joint residual limits, widening shoulder/elbow/wrist motion."""

  arm_mask = torch.tensor(
    [
      any(token in joint_name for token in cfg.arm_joint_name_tokens)
      for joint_name in joint_names
    ],
    dtype=torch.bool,
    device=device,
  )
  limits = torch.full(
    (len(joint_names),),
    cfg.max_joint_position_residual_rad,
    device=device,
  )
  limits[arm_mask] = cfg.max_arm_joint_position_residual_rad
  return limits


def arm_posture_velocity_candidates(
  cfg: LinkCbfReferenceFilterCfg,
  *,
  joint_names: tuple[str, ...],
  joint_pos: torch.Tensor,
  standing_joint_pos: torch.Tensor,
  posture_joint_mask: torch.Tensor | None = None,
  left_joint_mask: torch.Tensor | None = None,
  right_joint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Return preserve/both-down/left-down/right-down batched candidates."""

  if posture_joint_mask is None:
    posture_flags = [
      any(token in joint_name for token in cfg.posture_joint_name_tokens)
      for joint_name in joint_names
    ]
    if not any(posture_flags):
      raise ValueError("lookahead posture selection requires posture joints")
    posture_joint_mask = torch.tensor(
      posture_flags,
      dtype=torch.bool,
      device=joint_pos.device,
    )
  if left_joint_mask is None:
    left_joint_mask = posture_joint_mask & torch.tensor(
      [joint_name.startswith("left_") for joint_name in joint_names],
      dtype=torch.bool,
      device=joint_pos.device,
    )
  if right_joint_mask is None:
    right_joint_mask = posture_joint_mask & torch.tensor(
      [joint_name.startswith("right_") for joint_name in joint_names],
      dtype=torch.bool,
      device=joint_pos.device,
    )
  both_down = cfg.standing_posture_gain * (standing_joint_pos - joint_pos)
  both_down = torch.clamp(
    both_down,
    -cfg.max_standing_velocity_correction_rps,
    cfg.max_standing_velocity_correction_rps,
  )
  both_down = torch.where(posture_joint_mask[None], both_down, 0.0)
  left_down = torch.where(left_joint_mask[None], both_down, 0.0)
  right_down = torch.where(right_joint_mask[None], both_down, 0.0)
  return torch.stack(
    (torch.zeros_like(both_down), both_down, left_down, right_down), dim=1
  )


def gate_joint_recovery_during_posture(
  joint_recovery_velocity: torch.Tensor,
  preferred_posture_velocity: torch.Tensor,
) -> torch.Tensor:
  """Do not let reference recovery cancel an active posture correction."""

  posture_active = preferred_posture_velocity.abs() > 1e-6
  return torch.where(posture_active, 0.0, joint_recovery_velocity)


def update_posture_hold_time(
  remaining_time_s: torch.Tensor,
  selected_sides: torch.Tensor,
  *,
  step_dt: float,
  hold_s: float,
) -> torch.Tensor:
  """Update batched left/right posture latches without environment loops."""

  decayed = torch.clamp_min(remaining_time_s - step_dt, 0.0)
  return torch.where(selected_sides, hold_s, decayed)


def select_lookahead_arm_posture_velocity(
  cfg: LinkCbfReferenceFilterCfg,
  *,
  link_names: tuple[str, ...],
  candidate_joint_velocities: torch.Tensor,
  candidate_link_positions_w: torch.Tensor,
  nearest_obstacle_ids: torch.Tensor,
  link_active: torch.Tensor,
  capsule_centers_w: torch.Tensor,
  capsule_quaternions_w: torch.Tensor,
  capsule_sizes: torch.Tensor,
  obstacle_velocities_w: torch.Tensor,
  arm_link_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Choose the candidate improving each threatened arm link/capsule pair."""

  if arm_link_mask is None:
    arm_link_flags = [
      any(token in link_name for token in cfg.arm_joint_name_tokens)
      for link_name in link_names
    ]
    if not any(arm_link_flags):
      raise ValueError("lookahead posture selection requires arm links")
    arm_link_mask = torch.tensor(
      arm_link_flags,
      dtype=torch.bool,
      device=candidate_joint_velocities.device,
    )

  predicted_centers = (
    capsule_centers_w + cfg.posture_lookahead_s * obstacle_velocities_w
  )
  gather_xyz = nearest_obstacle_ids[..., None].expand(-1, -1, 3)
  nearest_centers = torch.gather(predicted_centers, 1, gather_xyz)
  nearest_quaternions = torch.gather(
    capsule_quaternions_w,
    1,
    nearest_obstacle_ids[..., None].expand(-1, -1, 4),
  )
  nearest_sizes = torch.gather(capsule_sizes, 1, gather_xyz)
  axes_w = _quaternion_z_axis(nearest_quaternions)
  segment_half_w = axes_w * nearest_sizes[..., 1, None]
  starts_w = nearest_centers - segment_half_w
  segments_w = 2.0 * segment_half_w

  relative = candidate_link_positions_w - starts_w[:, None]
  segment = segments_w[:, None]
  denominator = (segment * segment).sum(dim=-1).clamp_min(1e-12)
  fraction = ((relative * segment).sum(dim=-1) / denominator).clamp(0.0, 1.0)
  closest_w = starts_w[:, None] + fraction[..., None] * segment
  distance = torch.linalg.vector_norm(candidate_link_positions_w - closest_w, dim=-1)
  clearance = distance - nearest_sizes[:, None, :, 0] - cfg.link_radius_m
  # A pair is worth considering when it is active now or when preserving the
  # reference will put it inside the activation margin at the lookahead time.
  # Score the capped mean across those pairs instead of the global minimum: an
  # unchanged shoulder must not hide a large clearance gain at the forearm or
  # wrist. Links that are already outside the margin provide no extra reward.
  threatened_arm_links = arm_link_mask[None] & (
    link_active | (clearance[:, 0] <= cfg.posture_activation_clearance_m)
  )
  capped_clearance = torch.clamp_max(clearance, cfg.posture_activation_clearance_m)
  threat_count = threatened_arm_links.sum(dim=1).clamp_min(1)
  predicted_clearance_score = (
    torch.where(threatened_arm_links[:, None], capped_clearance, 0.0).sum(dim=2)
    / threat_count[:, None]
  )
  motion_cost = cfg.posture_motion_cost * candidate_joint_velocities.square().mean(
    dim=-1
  )
  selected_ids = torch.argmax(predicted_clearance_score - motion_cost, dim=1)
  selected = torch.gather(
    candidate_joint_velocities,
    1,
    selected_ids[:, None, None].expand(-1, 1, candidate_joint_velocities.shape[-1]),
  ).squeeze(1)
  has_arm_threat = threatened_arm_links.any(dim=1)
  return torch.where(has_arm_threat[:, None], selected, 0.0)


def project_preferred_joint_velocity_to_cbf(
  cfg: LinkCbfReferenceFilterCfg,
  *,
  preferred_joint_velocity: torch.Tensor,
  link_linear_jacobian: torch.Tensor,
  link_normals_w: torch.Tensor,
  required_joint_outward_speed_mps: torch.Tensor,
  link_active: torch.Tensor,
) -> torch.Tensor:
  """Project a preferred posture velocity onto active link-CBF half-spaces."""

  outward_joint_axes = torch.einsum(
    "nlc,nljc->nlj", link_normals_w, link_linear_jacobian
  )
  velocity = preferred_joint_velocity.clone()
  for _ in range(cfg.standing_projection_iterations):
    for link_index in range(outward_joint_axes.shape[1]):
      axis = outward_joint_axes[:, link_index]
      deficiency = required_joint_outward_speed_mps[:, link_index] - (
        axis * velocity
      ).sum(dim=-1)
      deficiency = torch.where(
        link_active[:, link_index], torch.clamp_min(deficiency, 0.0), 0.0
      )
      velocity += (
        deficiency[:, None]
        * axis
        / (axis.square().sum(dim=-1, keepdim=True) + cfg.jacobian_damping**2)
      )
      velocity.clamp_(
        -cfg.max_joint_velocity_correction_rps,
        cfg.max_joint_velocity_correction_rps,
      )
  return velocity


def safe_standing_joint_velocity(
  cfg: LinkCbfReferenceFilterCfg,
  *,
  joint_pos: torch.Tensor,
  standing_joint_pos: torch.Tensor,
  link_joint_ancestry: torch.Tensor,
  link_linear_jacobian: torch.Tensor,
  link_normals_w: torch.Tensor,
  link_clearance_m: torch.Tensor,
  link_active: torch.Tensor,
) -> torch.Tensor:
  """Return threat-gated standing pull with inward link motion projected out."""
  clearance_span = max(cfg.activation_clearance_m - cfg.safe_clearance_m, 1e-6)
  threat_strength = torch.clamp(
    (cfg.activation_clearance_m - link_clearance_m) / clearance_span,
    min=0.0,
    max=1.0,
  )
  threat_strength = torch.where(link_active, threat_strength, 0.0)
  joint_threat_strength = (
    threat_strength[:, :, None] * link_joint_ancestry[None]
  ).amax(dim=1)
  standing_velocity = cfg.standing_posture_gain * (standing_joint_pos - joint_pos)
  standing_velocity *= joint_threat_strength
  standing_velocity = torch.clamp(
    standing_velocity,
    -cfg.max_standing_velocity_correction_rps,
    cfg.max_standing_velocity_correction_rps,
  )

  outward_joint_axes = torch.einsum(
    "nlc,nljc->nlj", link_normals_w, link_linear_jacobian
  )
  for _ in range(cfg.standing_projection_iterations):
    for link_index in range(len(cfg.body_names)):
      outward_axis = outward_joint_axes[:, link_index]
      inward_speed = torch.clamp(
        -(outward_axis * standing_velocity).sum(dim=-1), min=0.0
      )
      inward_speed = torch.where(link_active[:, link_index], inward_speed, 0.0)
      standing_velocity += (
        inward_speed[:, None]
        * outward_axis
        / outward_axis.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
      )
  return torch.clamp(
    standing_velocity,
    -cfg.max_standing_velocity_correction_rps,
    cfg.max_standing_velocity_correction_rps,
  )


def filter_link_velocities(
  cfg: LinkCbfReferenceFilterCfg,
  *,
  link_positions_w: torch.Tensor,
  link_velocities_w: torch.Tensor,
  capsule_centers_w: torch.Tensor,
  capsule_quaternions_w: torch.Tensor,
  capsule_sizes: torch.Tensor,
  obstacle_velocities_w: torch.Tensor,
) -> LinkFilterResult:
  """Compute nearest-capsule CBF velocity corrections for robot links."""
  axes_w = _quaternion_z_axis(capsule_quaternions_w)
  radii = capsule_sizes[..., 0]
  segment_half_w = axes_w * capsule_sizes[..., 1, None]
  starts_w = capsule_centers_w - segment_half_w
  segments_w = 2.0 * segment_half_w

  relative = link_positions_w[:, :, None, :] - starts_w[:, None, :, :]
  segment = segments_w[:, None, :, :]
  denominator = (segment * segment).sum(dim=-1).clamp_min(1e-12)
  fraction = ((relative * segment).sum(dim=-1) / denominator).clamp(0.0, 1.0)
  closest_w = starts_w[:, None, :, :] + fraction[..., None] * segment
  separation = link_positions_w[:, :, None, :] - closest_w
  distance = torch.linalg.vector_norm(separation, dim=-1)
  clearance = distance - radii[:, None, :] - cfg.link_radius_m
  minimum_clearance, nearest_ids = clearance.min(dim=-1)

  gather_ids = nearest_ids[..., None, None].expand(-1, -1, 1, 3)
  nearest_separation = torch.gather(separation, 2, gather_ids).squeeze(2)
  nearest_distance = torch.gather(distance, 2, nearest_ids[..., None]).squeeze(2)
  nearest_velocity = torch.gather(
    obstacle_velocities_w[:, None].expand(-1, len(cfg.body_names), -1, -1),
    2,
    gather_ids,
  ).squeeze(2)
  normals = nearest_separation / nearest_distance[..., None].clamp_min(1e-6)
  required_outward_speed = (normals * nearest_velocity).sum(dim=-1) - cfg.cbf_alpha * (
    minimum_clearance - cfg.safe_clearance_m
  )
  achieved_outward_speed = (normals * link_velocities_w).sum(dim=-1)
  deficiency = torch.clamp(required_outward_speed - achieved_outward_speed, min=0.0)
  active = (
    (minimum_clearance <= cfg.activation_clearance_m)
    & (nearest_distance > 1e-6)
    & (capsule_centers_w[..., 2].max(dim=1).values > -50.0)[:, None]
  )
  deficiency = torch.where(active, deficiency, 0.0)
  deficiency = torch.clamp(deficiency, max=cfg.max_link_velocity_correction_mps)
  return LinkFilterResult(
    link_velocity_correction_w=deficiency[..., None] * normals,
    minimum_clearance_m=minimum_clearance,
    nearest_obstacle_ids=nearest_ids,
    required_outward_speed_mps=required_outward_speed,
    normals_w=normals,
    active=active,
  )


def filter_planar_velocity(
  cfg: PlanarCbfReferenceFilterCfg,
  *,
  robot_position_xy_w: torch.Tensor,
  nominal_velocity_xy_w: torch.Tensor,
  closest_points_xy_w: torch.Tensor,
  surface_clearances_m: torch.Tensor,
  obstacle_velocities_xy_w: torch.Tensor,
  active: torch.Tensor,
) -> PlanarFilterResult:
  """Project nominal velocity onto nearby moving-obstacle CBF half-spaces.

  This is a small deterministic sequential projection, not a learned policy.
  Each constraint has the zeroing-CBF form

  ``n dot (v_robot - v_obstacle) >= -alpha * (clearance - safe_clearance)``.
  """
  obstacle_count = surface_clearances_m.shape[1]
  selected_count = min(cfg.nearest_obstacles, obstacle_count)
  masked_clearance = torch.where(
    active,
    surface_clearances_m,
    torch.full_like(surface_clearances_m, torch.inf),
  )
  selected_clearance, selected_ids = torch.topk(
    masked_clearance,
    k=selected_count,
    dim=1,
    largest=False,
    sorted=True,
  )
  gather_xy = selected_ids[..., None].expand(-1, -1, 2)
  selected_points = torch.gather(closest_points_xy_w, 1, gather_xy)
  selected_obstacle_velocity = torch.gather(obstacle_velocities_xy_w, 1, gather_xy)
  selected_active = torch.gather(active, 1, selected_ids) & (
    selected_clearance <= cfg.activation_clearance_m
  )

  separation = robot_position_xy_w[:, None, :] - selected_points
  separation_norm = torch.linalg.vector_norm(separation, dim=-1, keepdim=True)
  normals = separation / separation_norm.clamp_min(1e-6)
  selected_active &= separation_norm.squeeze(-1) > 1e-6
  required_outward_speed = (normals * selected_obstacle_velocity).sum(
    dim=-1
  ) - cfg.cbf_alpha * (selected_clearance - cfg.safe_clearance_m)

  velocity = nominal_velocity_xy_w.clone()
  for _ in range(cfg.projection_iterations):
    for obstacle_index in range(selected_count):
      normal = normals[:, obstacle_index]
      deficiency = required_outward_speed[:, obstacle_index] - (normal * velocity).sum(
        dim=-1
      )
      correction = torch.clamp(deficiency, min=0.0)
      correction = torch.where(selected_active[:, obstacle_index], correction, 0.0)
      velocity = velocity + correction[:, None] * normal

  intervention = _limit_vector_norm(
    velocity - nominal_velocity_xy_w,
    cfg.max_intervention_speed_mps,
  )
  velocity = _limit_vector_norm(
    nominal_velocity_xy_w + intervention,
    cfg.max_planar_speed_mps,
  )
  intervention = velocity - nominal_velocity_xy_w

  achieved_outward_speed = (normals * velocity[:, None, :]).sum(dim=-1)
  violation = torch.clamp(required_outward_speed - achieved_outward_speed, min=0.0)
  violation = torch.where(selected_active, violation, 0.0)
  max_violation = violation.max(dim=1).values
  minimum_clearance = masked_clearance.min(dim=1).values

  return PlanarFilterResult(
    velocity_w=velocity,
    intervention_w=intervention,
    minimum_clearance_m=minimum_clearance,
    maximum_cbf_violation_mps=max_violation,
  )


__all__ = [
  "LinkCbfReferenceFilterCfg",
  "LinkFilterResult",
  "PlanarCbfReferenceFilterCfg",
  "PlanarFilterResult",
  "arm_posture_velocity_candidates",
  "filter_link_velocities",
  "filter_planar_velocity",
  "gate_joint_recovery_during_posture",
  "joint_position_residual_limits",
  "planar_capsule_geometry",
  "project_preferred_joint_velocity_to_cbf",
  "safe_standing_joint_velocity",
  "select_lookahead_arm_posture_velocity",
  "update_posture_hold_time",
]
