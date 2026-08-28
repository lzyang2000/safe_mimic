"""Exact kinematic motion replay for reference-filtering demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

from safe_mimic.tasks.reference_filter import (
  LinkCbfReferenceFilterCfg,
  PlanarCbfReferenceFilterCfg,
  arm_posture_velocity_candidates,
  filter_link_velocities,
  filter_planar_velocity,
  gate_joint_recovery_during_posture,
  joint_position_residual_limits,
  planar_capsule_geometry,
  project_preferred_joint_velocity_to_cbf,
  select_lookahead_arm_posture_velocity,
  update_posture_hold_time,
)


class KinematicReplayMotionCommand(MotionCommand):
  """Overwrite the robot with the current reference frame before sensing.

  The environment retains its normal action interface, but policy actions and
  intermediate physics state cannot change the pose observed at the end of a
  control step. This makes the command useful for inspecting reference motion,
  moving humans, and body-mounted sensors without tracking-policy error.
  """

  cfg: KinematicReplayMotionCommandCfg

  def __init__(self, cfg: KinematicReplayMotionCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self._all_env_ids = torch.arange(
      self.num_envs, dtype=torch.long, device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Start exact replay at frame zero without reset-state randomization."""
    self.reset_to_frame(env_ids, 0)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    """Advance, write, and forward the exact reference for selected envs."""
    replay_env_ids = self._all_env_ids if env_ids is None else env_ids
    if replay_env_ids.numel() == 0:
      return

    self.time_steps[replay_env_ids] = torch.remainder(
      self.time_steps[replay_env_ids] + 1,
      self.motion.time_step_total,
    )
    self._write_reference_state_to_sim(
      replay_env_ids,
      self.body_pos_w[replay_env_ids, 0],
      self.body_quat_w[replay_env_ids, 0],
      self.body_lin_vel_w[replay_env_ids, 0],
      self.body_ang_vel_w[replay_env_ids, 0],
      self.joint_pos[replay_env_ids],
      self.joint_vel[replay_env_ids],
    )

    # Command updates run after ManagerBasedRlEnv's shared forward() and before
    # sim.sense(). Refresh derived body/site state so the LiDAR phase and viewer
    # use the frame just written above.
    self._env.sim.forward()
    self.update_relative_body_poses()


@dataclass(kw_only=True)
class KinematicReplayMotionCommandCfg(MotionCommandCfg):
  """Configuration that builds :class:`KinematicReplayMotionCommand`."""

  def build(self, env) -> KinematicReplayMotionCommand:
    return KinematicReplayMotionCommand(self, env)


class PlanarFilteredReplayMotionCommand(KinematicReplayMotionCommand):
  """Expose a stateful, privileged-CBF-filtered motion reference.

  In kinematic mode the filtered state is also written directly to MuJoCo for
  filter debugging.  In policy-tracking mode only the command properties are
  changed: physics and policy actions retain control of the robot while the
  actor observes and tracks the filtered reference.
  """

  cfg: PlanarFilteredReplayMotionCommandCfg

  def __init__(self, cfg: PlanarFilteredReplayMotionCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self._obstacle_entities = tuple(
      env.scene[name] for name in cfg.obstacle_entity_names
    )
    obstacle_count = sum(
      entity.data.geom_pos_w.shape[1] for entity in self._obstacle_entities
    )
    if obstacle_count < 1:
      raise ValueError("reference filter requires at least one obstacle geom")

    self._filtered_root_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
    self._filtered_root_velocity_xy_w = torch.zeros_like(self._filtered_root_xy_w)
    self._root_translation_residual_xy_w = torch.zeros_like(self._filtered_root_xy_w)
    self._filter_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._previous_obstacle_centers_w = torch.zeros(
      self.num_envs, obstacle_count, 3, device=self.device
    )
    self._obstacle_velocity_w = torch.zeros(
      self.num_envs, obstacle_count, 3, device=self.device
    )
    self._obstacle_sample_age_s = torch.zeros(
      self.num_envs, obstacle_count, device=self.device
    )
    self._obstacle_history_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    link_body_ids, link_body_names = self.robot.find_bodies(
      cfg.link_filter.body_names, preserve_order=True
    )
    if tuple(link_body_names) != cfg.link_filter.body_names:
      raise ValueError("link filter body order does not match its configuration")
    self._link_body_ids = torch.tensor(
      link_body_ids, dtype=torch.long, device=self.device
    )
    self._link_body_global_ids = self.robot.indexing.body_ids[self._link_body_ids].to(
      dtype=torch.long
    )
    self._joint_global_ids = self.robot.indexing.joint_ids.to(dtype=torch.long)
    self._joint_position_residual_limit_rad = joint_position_residual_limits(
      cfg.link_filter, tuple(self.robot.joint_names), self.device
    )
    self._link_joint_ancestry = self._build_link_joint_ancestry()
    self._joint_joint_ancestry, self._joint_rollout_order = (
      self._build_joint_joint_ancestry()
    )
    self._joint_descendant_link_ids = tuple(
      torch.nonzero(self._link_joint_ancestry[:, joint_id], as_tuple=False).flatten()
      for joint_id in range(self.robot.num_joints)
    )
    self._joint_descendant_joint_ids = tuple(
      torch.nonzero(self._joint_joint_ancestry[:, joint_id], as_tuple=False).flatten()
      for joint_id in range(self.robot.num_joints)
    )
    posture_joint_mask = torch.tensor(
      [
        any(token in joint_name for token in cfg.link_filter.posture_joint_name_tokens)
        for joint_name in self.robot.joint_names
      ],
      dtype=torch.bool,
      device=self.device,
    )
    self._left_posture_joint_mask = posture_joint_mask & torch.tensor(
      [joint_name.startswith("left_") for joint_name in self.robot.joint_names],
      dtype=torch.bool,
      device=self.device,
    )
    self._right_posture_joint_mask = posture_joint_mask & torch.tensor(
      [joint_name.startswith("right_") for joint_name in self.robot.joint_names],
      dtype=torch.bool,
      device=self.device,
    )
    self._posture_hold_remaining_s = torch.zeros(
      self.num_envs, 2, device=self.device
    )
    self._filtered_joint_pos = torch.zeros(
      self.num_envs, self.robot.num_joints, device=self.device
    )
    self._filtered_joint_vel = torch.zeros_like(self._filtered_joint_pos)
    self._joint_position_residual = torch.zeros_like(self._filtered_joint_pos)
    self._joint_filter_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["filter_minimum_clearance_m"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["filter_intervention_speed_mps"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["filter_reference_offset_m"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["filter_cbf_violation_mps"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["link_filter_minimum_clearance_m"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["link_filter_cbf_violation_mps"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["joint_filter_intervention_rps"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["joint_filter_standing_pull_rps"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["joint_filter_reference_residual_rad"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def _raw_body_pos_w(self) -> torch.Tensor:
    return MotionCommand.body_pos_w.fget(self)  # type: ignore[union-attr]

  def _raw_body_lin_vel_w(self) -> torch.Tensor:
    return MotionCommand.body_lin_vel_w.fget(self)  # type: ignore[union-attr]

  def _raw_joint_pos(self) -> torch.Tensor:
    return MotionCommand.joint_pos.fget(self)  # type: ignore[union-attr]

  def _raw_joint_vel(self) -> torch.Tensor:
    return MotionCommand.joint_vel.fget(self)  # type: ignore[union-attr]

  def _build_link_joint_ancestry(self) -> torch.Tensor:
    """Return which hinge joints can move each configured robot link."""
    model = self._env.sim.mj_model
    parent_ids = model.body_parentid
    joint_body_ids = model.jnt_bodyid
    joint_ids = self._joint_global_ids.detach().cpu().tolist()
    ancestry: list[list[bool]] = []
    for link_body_id in self._link_body_global_ids.detach().cpu().tolist():
      ancestor_bodies: set[int] = set()
      body_id = int(link_body_id)
      while body_id > 0:
        ancestor_bodies.add(body_id)
        body_id = int(parent_ids[body_id])
      ancestry.append(
        [int(joint_body_ids[joint_id]) in ancestor_bodies for joint_id in joint_ids]
      )
    return torch.tensor(ancestry, dtype=torch.float32, device=self.device)

  def _build_joint_joint_ancestry(self) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return descendant-joint masks and a root-to-leaf rollout order."""
    model = self._env.sim.mj_model
    parent_ids = model.body_parentid
    joint_body_ids = model.jnt_bodyid
    joint_ids = self._joint_global_ids.detach().cpu().tolist()

    body_depth: dict[int, int] = {0: 0}

    def depth(body_id: int) -> int:
      if body_id not in body_depth:
        body_depth[body_id] = depth(int(parent_ids[body_id])) + 1
      return body_depth[body_id]

    ancestry: list[list[bool]] = []
    for descendant_joint_id in joint_ids:
      descendant_body_id = int(joint_body_ids[descendant_joint_id])
      ancestor_bodies: set[int] = set()
      body_id = int(parent_ids[descendant_body_id])
      while body_id > 0:
        ancestor_bodies.add(body_id)
        body_id = int(parent_ids[body_id])
      ancestry.append(
        [
          int(joint_body_ids[ancestor_joint_id]) in ancestor_bodies
          or (
            int(joint_body_ids[ancestor_joint_id]) == descendant_body_id
            and ancestor_joint_id < descendant_joint_id
          )
          for ancestor_joint_id in joint_ids
        ]
      )

    order = tuple(
      sorted(
        range(len(joint_ids)),
        key=lambda local_id: (
          depth(int(joint_body_ids[joint_ids[local_id]])),
          joint_ids[local_id],
        ),
      )
    )
    return torch.tensor(ancestry, dtype=torch.bool, device=self.device), order

  @property
  def joint_pos(self) -> torch.Tensor:
    raw = self._raw_joint_pos()
    if not hasattr(self, "_joint_filter_initialized"):
      return raw
    return torch.where(
      self._joint_filter_initialized[:, None], self._filtered_joint_pos, raw
    )

  @property
  def joint_vel(self) -> torch.Tensor:
    raw = self._raw_joint_vel()
    if not hasattr(self, "_joint_filter_initialized"):
      return raw
    return torch.where(
      self._joint_filter_initialized[:, None], self._filtered_joint_vel, raw
    )

  @property
  def body_pos_w(self) -> torch.Tensor:
    raw = self._raw_body_pos_w()
    if not hasattr(self, "_filter_initialized"):
      return raw
    offset_xy = self._filtered_root_xy_w - raw[:, 0, :2]
    offset_xy = torch.where(self._filter_initialized[:, None], offset_xy, 0.0)
    filtered = raw.clone()
    filtered[..., :2] += offset_xy[:, None, :]
    return filtered

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    raw = self._raw_body_lin_vel_w()
    if not hasattr(self, "_filter_initialized"):
      return raw
    velocity_delta_xy = self._filtered_root_velocity_xy_w - raw[:, 0, :2]
    velocity_delta_xy = torch.where(
      self._filter_initialized[:, None], velocity_delta_xy, 0.0
    )
    filtered = raw.clone()
    filtered[..., :2] += velocity_delta_xy[:, None, :]
    return filtered

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    raw = MotionCommand.anchor_pos_w.fget(self)  # type: ignore[union-attr]
    if not hasattr(self, "_filter_initialized"):
      return raw
    raw_root_xy = self._raw_body_pos_w()[:, 0, :2]
    offset_xy = self._filtered_root_xy_w - raw_root_xy
    offset_xy = torch.where(self._filter_initialized[:, None], offset_xy, 0.0)
    filtered = raw.clone()
    filtered[:, :2] += offset_xy
    return filtered

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    raw = MotionCommand.anchor_lin_vel_w.fget(self)  # type: ignore[union-attr]
    if not hasattr(self, "_filter_initialized"):
      return raw
    raw_root_velocity_xy = self._raw_body_lin_vel_w()[:, 0, :2]
    velocity_delta_xy = self._filtered_root_velocity_xy_w - raw_root_velocity_xy
    velocity_delta_xy = torch.where(
      self._filter_initialized[:, None], velocity_delta_xy, 0.0
    )
    filtered = raw.clone()
    filtered[:, :2] += velocity_delta_xy
    return filtered

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.time_steps[env_ids] = 0
    raw_root_pos = self._raw_body_pos_w()[env_ids, 0]
    raw_root_velocity = self._raw_body_lin_vel_w()[env_ids, 0]
    self._filtered_root_xy_w[env_ids] = raw_root_pos[:, :2]
    self._filtered_root_velocity_xy_w[env_ids] = raw_root_velocity[:, :2]
    self._root_translation_residual_xy_w[env_ids] = 0.0
    self._filter_initialized[env_ids] = True
    self._filtered_joint_pos[env_ids] = self._raw_joint_pos()[env_ids]
    self._filtered_joint_vel[env_ids] = self._raw_joint_vel()[env_ids]
    self._joint_position_residual[env_ids] = 0.0
    self._posture_hold_remaining_s[env_ids] = 0.0
    self._joint_filter_initialized[env_ids] = True
    self._obstacle_history_initialized[env_ids] = False
    self.reset_to_frame(env_ids, 0)

  def _obstacle_tensors(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = torch.cat(
      [entity.data.geom_pos_w for entity in self._obstacle_entities], dim=1
    )
    quaternions = torch.cat(
      [entity.data.geom_quat_w for entity in self._obstacle_entities], dim=1
    )
    sizes = torch.cat(
      [
        self._env.sim.model.geom_size[:, entity.indexing.geom_ids.to(dtype=torch.long)]
        for entity in self._obstacle_entities
      ],
      dim=1,
    )
    return centers, quaternions, sizes

  def _estimate_obstacle_velocities(
    self,
    centers_w: torch.Tensor,
    active: torch.Tensor,
  ) -> torch.Tensor:
    cfg = self.cfg.planar_filter
    dt = self._env.step_dt
    initialized = self._obstacle_history_initialized
    new_envs = ~initialized
    if torch.any(new_envs):
      self._previous_obstacle_centers_w[new_envs] = centers_w[new_envs]
      self._obstacle_velocity_w[new_envs] = 0.0
      self._obstacle_sample_age_s[new_envs] = 0.0
      self._obstacle_history_initialized[new_envs] = True

    self._obstacle_sample_age_s += dt
    displacement = centers_w - self._previous_obstacle_centers_w
    changed = torch.linalg.vector_norm(displacement, dim=-1) > 1e-5
    robot_z = self._raw_body_pos_w()[:, None, 0, 2]
    previous_z = self._previous_obstacle_centers_w[..., 2]
    previous_active = torch.abs(previous_z - robot_z) <= 2.0 * cfg.vertical_gate_m
    measurable = changed & active & previous_active & initialized[:, None]
    elapsed = self._obstacle_sample_age_s.clamp_min(dt)
    measured_velocity = displacement / elapsed[..., None]
    measured_velocity = _limit_obstacle_velocity(
      measured_velocity, cfg.obstacle_velocity_limit_mps
    )
    smoothed = (
      cfg.obstacle_velocity_smoothing * self._obstacle_velocity_w
      + (1.0 - cfg.obstacle_velocity_smoothing) * measured_velocity
    )
    decay = torch.exp(
      torch.tensor(
        -cfg.obstacle_velocity_decay * dt,
        device=self.device,
      )
    )
    self._obstacle_velocity_w *= decay
    self._obstacle_velocity_w = torch.where(
      measurable[..., None], smoothed, self._obstacle_velocity_w
    )
    self._previous_obstacle_centers_w = torch.where(
      changed[..., None], centers_w, self._previous_obstacle_centers_w
    )
    self._obstacle_sample_age_s = torch.where(changed, 0.0, self._obstacle_sample_age_s)
    return self._obstacle_velocity_w

  def _link_linear_jacobian(self) -> torch.Tensor:
    """Compute batched world-frame linear Jacobians for filtered links."""
    joint_axes_w = self._env.sim.data.xaxis[:, self._joint_global_ids]
    joint_anchors_w = self._env.sim.data.xanchor[:, self._joint_global_ids]
    link_positions_w = self.robot.data.body_link_pos_w[:, self._link_body_ids]
    lever_arms_w = link_positions_w[:, :, None, :] - joint_anchors_w[:, None, :, :]
    jacobian = torch.cross(
      joint_axes_w[:, None, :, :].expand_as(lever_arms_w),
      lever_arms_w,
      dim=-1,
    )
    return jacobian * self._link_joint_ancestry[None, :, :, None]

  @staticmethod
  def _rotate_about_axis(
    vectors: torch.Tensor,
    axes: torch.Tensor,
    angles: torch.Tensor,
  ) -> torch.Tensor:
    """Apply batched Rodrigues rotations to one or more vectors."""
    axes = axes[:, :, None]
    cosine = torch.cos(angles)[:, :, None, None]
    sine = torch.sin(angles)[:, :, None, None]
    return (
      cosine * vectors
      + sine * torch.cross(axes.expand_as(vectors), vectors, dim=-1)
      + (1.0 - cosine) * axes * (axes * vectors).sum(dim=-1, keepdim=True)
    )

  def _rollout_candidate_link_positions(
    self,
    candidate_joint_velocities: torch.Tensor,
    link_positions_w: torch.Tensor,
  ) -> torch.Tensor:
    """Roll candidate poses through finite hinge rotations on batched CUDA data.

    The only Python loop follows the fixed robot joint tree. Environments,
    candidates, descendant links, and descendant joint frames remain tensor
    dimensions throughout; this does not invoke MuJoCo or transfer runtime data
    to the CPU.
    """
    cfg = self.cfg.link_filter
    candidate_joint_pos = self.joint_pos[:, None] + (
      cfg.posture_lookahead_s * candidate_joint_velocities
    )
    soft_limits = self.robot.data.soft_joint_pos_limits
    candidate_joint_pos = torch.clamp(
      candidate_joint_pos,
      soft_limits[:, None, :, 0],
      soft_limits[:, None, :, 1],
    )
    candidate_joint_delta = candidate_joint_pos - self.joint_pos[:, None]

    candidate_count = candidate_joint_velocities.shape[1]
    candidate_link_positions = (
      link_positions_w[:, None].expand(-1, candidate_count, -1, -1).clone()
    )
    joint_axes = (
      self._env.sim.data.xaxis[:, self._joint_global_ids]
      .unsqueeze(1)
      .expand(-1, candidate_count, -1, -1)
      .clone()
    )
    joint_anchors = (
      self._env.sim.data.xanchor[:, self._joint_global_ids]
      .unsqueeze(1)
      .expand(-1, candidate_count, -1, -1)
      .clone()
    )

    for joint_id in self._joint_rollout_order:
      axis = joint_axes[:, :, joint_id]
      anchor = joint_anchors[:, :, joint_id]
      angle = candidate_joint_delta[:, :, joint_id]

      descendant_link_ids = self._joint_descendant_link_ids[joint_id]
      if descendant_link_ids.numel() > 0:
        descendant_positions = candidate_link_positions[
          :, :, descendant_link_ids
        ]
        relative_positions = descendant_positions - anchor[:, :, None]
        candidate_link_positions[:, :, descendant_link_ids] = (
          anchor[:, :, None]
          + self._rotate_about_axis(relative_positions, axis, angle)
        )

      descendant_joint_ids = self._joint_descendant_joint_ids[joint_id]
      if descendant_joint_ids.numel() > 0:
        descendant_anchors = joint_anchors[:, :, descendant_joint_ids]
        relative_anchors = descendant_anchors - anchor[:, :, None]
        joint_anchors[:, :, descendant_joint_ids] = anchor[:, :, None] + (
          self._rotate_about_axis(relative_anchors, axis, angle)
        )
        descendant_axes = joint_axes[:, :, descendant_joint_ids]
        joint_axes[:, :, descendant_joint_ids] = self._rotate_about_axis(
          descendant_axes, axis, angle
        )

    return candidate_link_positions

  def _joint_velocity_avoidance_correction(
    self,
    centers_w: torch.Tensor,
    quaternions_w: torch.Tensor,
    sizes: torch.Tensor,
    obstacle_velocity_w: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Map link CBF corrections into realizable hinge-joint velocities."""
    cfg = self.cfg.link_filter
    link_positions_w = self.robot.data.body_link_pos_w[:, self._link_body_ids]
    link_velocities_w = self.robot.data.body_link_lin_vel_w[:, self._link_body_ids]
    result = filter_link_velocities(
      cfg,
      link_positions_w=link_positions_w,
      link_velocities_w=link_velocities_w,
      capsule_centers_w=centers_w,
      capsule_quaternions_w=quaternions_w,
      capsule_sizes=sizes,
      obstacle_velocities_w=obstacle_velocity_w,
    )
    jacobian = self._link_linear_jacobian()
    # Select a semantically useful short-horizon posture before applying the
    # instantaneous CBF constraints. Finite FK lets an orthogonal move such as
    # lowering a horizontal arm receive credit for future 3D clearance.
    candidates = arm_posture_velocity_candidates(
      cfg,
      joint_names=tuple(self.robot.joint_names),
      joint_pos=self.joint_pos,
      standing_joint_pos=self.robot.data.default_joint_pos,
    )
    candidate_link_positions = self._rollout_candidate_link_positions(
      candidates, link_positions_w
    )
    preferred_velocity = select_lookahead_arm_posture_velocity(
      cfg,
      link_names=cfg.body_names,
      candidate_joint_velocities=candidates,
      candidate_link_positions_w=candidate_link_positions,
      nearest_obstacle_ids=result.nearest_obstacle_ids,
      link_active=result.active,
      capsule_centers_w=centers_w,
      capsule_quaternions_w=quaternions_w,
      capsule_sizes=sizes,
      obstacle_velocities_w=obstacle_velocity_w,
    )
    selected_sides = torch.stack(
      (
        preferred_velocity[:, self._left_posture_joint_mask]
        .abs()
        .amax(dim=1)
        > 1e-6,
        preferred_velocity[:, self._right_posture_joint_mask]
        .abs()
        .amax(dim=1)
        > 1e-6,
      ),
      dim=1,
    )
    self._posture_hold_remaining_s = update_posture_hold_time(
      self._posture_hold_remaining_s,
      selected_sides,
      step_dt=self._env.step_dt,
      hold_s=cfg.posture_hold_s,
    )
    held_left = self._posture_hold_remaining_s[:, 0] > 0.0
    held_right = self._posture_hold_remaining_s[:, 1] > 0.0
    held_velocity = (
      held_left[:, None] * candidates[:, 2]
      + held_right[:, None] * candidates[:, 3]
    )
    held_joint_mask = (
      held_left[:, None] & self._left_posture_joint_mask[None]
    ) | (held_right[:, None] & self._right_posture_joint_mask[None])
    preferred_velocity = torch.where(
      held_joint_mask, held_velocity, preferred_velocity
    )
    self.metrics["joint_filter_standing_pull_rps"] = torch.linalg.vector_norm(
      preferred_velocity, dim=-1
    )
    required_joint_outward_speed = torch.linalg.vector_norm(
      result.link_velocity_correction_w, dim=-1
    )
    joint_correction = project_preferred_joint_velocity_to_cbf(
      cfg,
      preferred_joint_velocity=preferred_velocity,
      link_linear_jacobian=jacobian,
      link_normals_w=result.normals_w,
      required_joint_outward_speed_mps=required_joint_outward_speed,
      link_active=result.active,
    )

    achieved_link_correction = torch.einsum("nljc,nj->nlc", jacobian, joint_correction)
    achieved_link_velocity = link_velocities_w + achieved_link_correction
    achieved_outward_speed = (result.normals_w * achieved_link_velocity).sum(dim=-1)
    violation = torch.clamp(
      result.required_outward_speed_mps - achieved_outward_speed,
      min=0.0,
    )
    violation = torch.where(result.active, violation, 0.0)
    active_clearance = torch.where(
      result.active,
      result.minimum_clearance_m,
      torch.full_like(result.minimum_clearance_m, torch.inf),
    )
    self.metrics["link_filter_minimum_clearance_m"] = active_clearance.min(dim=1).values
    self.metrics["link_filter_cbf_violation_mps"] = violation.max(dim=1).values
    self.metrics["joint_filter_intervention_rps"] = torch.linalg.vector_norm(
      joint_correction, dim=-1
    )
    return joint_correction, preferred_velocity

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    replay_env_ids = self._all_env_ids if env_ids is None else env_ids
    if replay_env_ids.numel() == 0:
      return

    next_steps = self.time_steps[replay_env_ids] + 1
    wrapped = next_steps >= self.motion.time_step_total
    self.time_steps[replay_env_ids] = torch.remainder(
      next_steps, self.motion.time_step_total
    )
    raw_root_pos = self._raw_body_pos_w()[:, 0]
    raw_root_velocity = self._raw_body_lin_vel_w()[:, 0]
    raw_joint_pos = self._raw_joint_pos()
    raw_joint_vel = self._raw_joint_vel()
    if torch.any(wrapped):
      wrapped_ids = replay_env_ids[wrapped]
      self._filtered_root_xy_w[wrapped_ids] = raw_root_pos[wrapped_ids, :2]
      self._filtered_root_velocity_xy_w[wrapped_ids] = raw_root_velocity[
        wrapped_ids, :2
      ]
      self._root_translation_residual_xy_w[wrapped_ids] = 0.0
      self._filtered_joint_pos[wrapped_ids] = raw_joint_pos[wrapped_ids]
      self._filtered_joint_vel[wrapped_ids] = raw_joint_vel[wrapped_ids]
      self._joint_position_residual[wrapped_ids] = 0.0
      self._posture_hold_remaining_s[wrapped_ids] = 0.0
      self._obstacle_history_initialized[wrapped_ids] = False
    uninitialized = ~self._filter_initialized[replay_env_ids]
    if torch.any(uninitialized):
      init_ids = replay_env_ids[uninitialized]
      self._filtered_root_xy_w[init_ids] = raw_root_pos[init_ids, :2]
      self._filtered_root_velocity_xy_w[init_ids] = raw_root_velocity[init_ids, :2]
      self._root_translation_residual_xy_w[init_ids] = 0.0
      self._filter_initialized[init_ids] = True
    uninitialized_joints = ~self._joint_filter_initialized[replay_env_ids]
    if torch.any(uninitialized_joints):
      init_ids = replay_env_ids[uninitialized_joints]
      self._filtered_joint_pos[init_ids] = raw_joint_pos[init_ids]
      self._filtered_joint_vel[init_ids] = raw_joint_vel[init_ids]
      self._joint_position_residual[init_ids] = 0.0
      self._joint_filter_initialized[init_ids] = True

    self._filtered_root_xy_w[replay_env_ids] = (
      raw_root_pos[replay_env_ids, :2]
      + self._root_translation_residual_xy_w[replay_env_ids]
    )
    centers_w, quaternions_w, sizes = self._obstacle_tensors()
    filter_root_pos = raw_root_pos.clone()
    filter_root_pos[:, :2] = self._filtered_root_xy_w
    closest_xy, clearance, active = planar_capsule_geometry(
      filter_root_pos,
      centers_w,
      quaternions_w,
      sizes,
      robot_radius_m=self.cfg.planar_filter.robot_radius_m,
      vertical_gate_m=self.cfg.planar_filter.vertical_gate_m,
    )
    obstacle_velocity = self._estimate_obstacle_velocities(centers_w, active)

    position_error = raw_root_pos[:, :2] - self._filtered_root_xy_w
    recovery_velocity = _limit_obstacle_velocity(
      self.cfg.planar_filter.recovery_gain * position_error,
      self.cfg.planar_filter.max_recovery_speed_mps,
    )
    nominal_velocity = raw_root_velocity[:, :2] + recovery_velocity
    result = filter_planar_velocity(
      self.cfg.planar_filter,
      robot_position_xy_w=self._filtered_root_xy_w,
      nominal_velocity_xy_w=nominal_velocity,
      closest_points_xy_w=closest_xy,
      surface_clearances_m=clearance,
      obstacle_velocities_xy_w=obstacle_velocity,
      active=active,
    )
    self._filtered_root_velocity_xy_w[replay_env_ids] = result.velocity_w[
      replay_env_ids
    ]
    self._root_translation_residual_xy_w[replay_env_ids] += self._env.step_dt * (
      result.velocity_w[replay_env_ids] - raw_root_velocity[replay_env_ids, :2]
    )
    self._filtered_root_xy_w[replay_env_ids] = (
      raw_root_pos[replay_env_ids, :2]
      + self._root_translation_residual_xy_w[replay_env_ids]
    )

    link_cfg = self.cfg.link_filter
    joint_recovery_velocity = torch.clamp(
      -link_cfg.joint_recovery_gain * self._joint_position_residual,
      -link_cfg.max_joint_velocity_correction_rps,
      link_cfg.max_joint_velocity_correction_rps,
    )
    nominal_joint_velocity = raw_joint_vel + joint_recovery_velocity
    candidate_residual = (
      self._joint_position_residual + self._env.step_dt * joint_recovery_velocity
    )
    candidate_residual = torch.clamp(
      candidate_residual,
      -self._joint_position_residual_limit_rad,
      self._joint_position_residual_limit_rad,
    )
    candidate_joint_pos = raw_joint_pos + candidate_residual
    soft_limits = self.robot.data.soft_joint_pos_limits
    candidate_joint_pos = torch.clamp(
      candidate_joint_pos, soft_limits[..., 0], soft_limits[..., 1]
    )
    candidate_residual = candidate_joint_pos - raw_joint_pos
    self._filtered_joint_pos[replay_env_ids] = candidate_joint_pos[replay_env_ids]
    self._filtered_joint_vel[replay_env_ids] = nominal_joint_velocity[replay_env_ids]

    # Kinematic filter tuning evaluates FK at the nominal reference candidate.
    # In live policy mode the shared environment forward has already refreshed
    # the actual robot links and joint frames; do not overwrite that state.
    if self.cfg.write_reference_to_sim:
      self._write_reference_state_to_sim(
        replay_env_ids,
        self.body_pos_w[replay_env_ids, 0],
        self.body_quat_w[replay_env_ids, 0],
        self.body_lin_vel_w[replay_env_ids, 0],
        self.body_ang_vel_w[replay_env_ids, 0],
        self.joint_pos[replay_env_ids],
        self.joint_vel[replay_env_ids],
      )
      self._env.sim.forward()

    joint_avoidance_velocity, preferred_posture_velocity = (
      self._joint_velocity_avoidance_correction(
        centers_w,
        quaternions_w,
        sizes,
        obstacle_velocity,
      )
    )
    effective_joint_recovery_velocity = gate_joint_recovery_during_posture(
      joint_recovery_velocity,
      preferred_posture_velocity,
    )
    filtered_joint_velocity = (
      raw_joint_vel
      + effective_joint_recovery_velocity
      + joint_avoidance_velocity
    )
    filtered_residual = self._joint_position_residual + self._env.step_dt * (
      effective_joint_recovery_velocity + joint_avoidance_velocity
    )
    filtered_residual = torch.clamp(
      filtered_residual,
      -self._joint_position_residual_limit_rad,
      self._joint_position_residual_limit_rad,
    )
    filtered_joint_pos = torch.clamp(
      raw_joint_pos + filtered_residual,
      soft_limits[..., 0],
      soft_limits[..., 1],
    )
    filtered_residual = filtered_joint_pos - raw_joint_pos
    self._filtered_joint_pos[replay_env_ids] = filtered_joint_pos[replay_env_ids]
    self._filtered_joint_vel[replay_env_ids] = filtered_joint_velocity[replay_env_ids]
    self._joint_position_residual[replay_env_ids] = filtered_residual[replay_env_ids]
    self.metrics["joint_filter_reference_residual_rad"][replay_env_ids] = (
      torch.linalg.vector_norm(filtered_residual[replay_env_ids], dim=-1)
    )

    self.metrics["filter_minimum_clearance_m"][replay_env_ids] = (
      result.minimum_clearance_m[replay_env_ids]
    )
    self.metrics["filter_intervention_speed_mps"][replay_env_ids] = (
      torch.linalg.vector_norm(result.intervention_w[replay_env_ids], dim=-1)
    )
    self.metrics["filter_reference_offset_m"][replay_env_ids] = (
      torch.linalg.vector_norm(
        self._root_translation_residual_xy_w[replay_env_ids], dim=-1
      )
    )
    self.metrics["filter_cbf_violation_mps"][replay_env_ids] = (
      result.maximum_cbf_violation_mps[replay_env_ids]
    )

    if self.cfg.write_reference_to_sim:
      self._write_reference_state_to_sim(
        replay_env_ids,
        self.body_pos_w[replay_env_ids, 0],
        self.body_quat_w[replay_env_ids, 0],
        self.body_lin_vel_w[replay_env_ids, 0],
        self.body_ang_vel_w[replay_env_ids, 0],
        self.joint_pos[replay_env_ids],
        self.joint_vel[replay_env_ids],
      )
      self._env.sim.forward()
    self.update_relative_body_poses()


def _limit_obstacle_velocity(velocity: torch.Tensor, limit: float) -> torch.Tensor:
  speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
  return velocity * torch.clamp(limit / speed.clamp_min(1e-8), max=1.0)


@dataclass(kw_only=True)
class PlanarFilteredReplayMotionCommandCfg(KinematicReplayMotionCommandCfg):
  """Build replay with privileged planar and link/joint CBF filters."""

  obstacle_entity_names: tuple[str, ...]
  write_reference_to_sim: bool = True
  planar_filter: PlanarCbfReferenceFilterCfg = field(
    default_factory=PlanarCbfReferenceFilterCfg
  )
  link_filter: LinkCbfReferenceFilterCfg = field(
    default_factory=LinkCbfReferenceFilterCfg
  )

  def build(self, env) -> PlanarFilteredReplayMotionCommand:
    return PlanarFilteredReplayMotionCommand(self, env)


__all__ = [
  "KinematicReplayMotionCommand",
  "KinematicReplayMotionCommandCfg",
  "PlanarFilteredReplayMotionCommand",
  "PlanarFilteredReplayMotionCommandCfg",
]
