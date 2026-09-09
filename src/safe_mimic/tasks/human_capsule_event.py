"""Persistent pre-forward updater for composed human capsule obstacles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase

from safe_mimic.assets.soma_capsules import (
  HUMAN_INACTIVE_HEIGHT_M,
  HUMAN_RAY_BODY_NAME,
  human_capsule_body_name,
)
from safe_mimic.motions.composed_skeleton_bank import (
  OnlineComposedHumanSampler,
  SkeletonPathBank,
  _quat_apply,
  _rotate_xy,
)
from safe_mimic.motions.human_capsules import (
  DEFAULT_MAX_HUMAN_HEIGHT_M,
  DEFAULT_MIN_HUMAN_HEIGHT_M,
  SOMA_CAPSULE_SPECS,
  sample_body_scales_for_height,
)
from safe_mimic.motions.soma_mesh import (
  SomaViserSkin,
  load_soma_mesh_skin,
  prepare_soma_viser_skin,
)
from safe_mimic.tasks.encounter_sampling import (
  EncounterSampler,
  encounter_sampler_overrides,
  read_collision_terms,
)
from safe_mimic.tasks.human_target import robot_intersection_target_from_qpos

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


DEFAULT_APPROACH_SPEED_FLOOR_MPS = 0.3
DEFAULT_ROBOT_SPEED_HALF_LIFE_S = 1.0


def limit_intersection_delay_to_speed(
  *,
  delay_s: torch.Tensor,
  spawn_radius_m: torch.Tensor,
  max_speed_mps: torch.Tensor,
) -> torch.Tensor:
  """Stretch a scheduled intersection delay until the approach obeys a cap.

  The online composer imposes the approach speed through the schedule, not
  through the source clip: the human spawns on a ring of ``spawn_radius_m``
  around the robot and its entry walk is time-warped (``playback_speed``) so
  it arrives after ``delay_s``. The realized approach speed is therefore
  ``spawn_radius_m / delay_s``, the same definition the TTC encounter sampler
  uses. Selecting slower source clips cannot bound it, because every clip is
  re-warped to whatever the schedule demands.

  Keeping the radius fixed and stretching only the delay preserves the
  encounter's geometry (spawn ring, crossing angle, intersection point) and
  changes just how fast the human walks it. Returns a new tensor; inputs are
  not mutated.
  """
  required_delay_s = spawn_radius_m / max_speed_mps.clamp_min(1.0e-6)
  return torch.maximum(delay_s, required_delay_s)


def advance_robot_speed_cap(
  previous_peak_mps: torch.Tensor,
  planar_speed_mps: torch.Tensor,
  *,
  decay: float,
  floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Track the robot's recent peak planar speed and the cap derived from it.

  The peak rises instantly to a new maximum and otherwise decays, so a
  momentary stop mid-stride does not collapse the cap. The returned cap is
  floored: a standing robot would otherwise cap the human at zero and the
  encounter would never occur. Returns ``(peak, cap)`` as new tensors.
  """
  peak = torch.maximum(planar_speed_mps, previous_peak_mps * decay)
  return peak, peak.clamp_min(floor)


@requires_model_fields("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb")
class HumanCapsuleMotion(ManagerTermBase):
  """Compose future-crossing human paths and drive their mocap capsules.

  The pruned skeleton bank and transition graph live on the environment device.
  Every reset samples ``walk -> action -> walk``, performs PHP-style root
  alignment and velocity-aware skeleton-space inertialization, and places the
  action midpoint on the robot's live pose. Only current capsule poses are
  retained outside the source bank.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    params = cfg.params
    self.human_entity_name = str(params["human_entity"])
    self.robot_entity_name = str(params.get("robot_entity", "robot"))
    self.command_name = str(params.get("command_name", "motion"))
    fixed_spawn_radius = params.get("initial_spawn_radius_m")
    min_spawn_radius = params.get("min_initial_spawn_radius_m", fixed_spawn_radius)
    max_spawn_radius = params.get("max_initial_spawn_radius_m", fixed_spawn_radius)
    self.min_initial_spawn_radius_m = (
      None if min_spawn_radius is None else float(min_spawn_radius)
    )
    self.max_initial_spawn_radius_m = (
      None if max_spawn_radius is None else float(max_spawn_radius)
    )
    self.min_intersection_delay_s = float(params.get("min_intersection_delay_s", 1.0))
    self.max_intersection_delay_s = float(params.get("max_intersection_delay_s", 3.0))
    self.min_crossing_angle_rad = float(
      params.get("min_crossing_angle_rad", np.deg2rad(60.0))
    )
    self.max_crossing_angle_rad = float(
      params.get("max_crossing_angle_rad", np.deg2rad(120.0))
    )
    self.min_human_height_m = float(
      params.get("min_human_height_m", DEFAULT_MIN_HUMAN_HEIGHT_M)
    )
    self.max_human_height_m = float(
      params.get("max_human_height_m", DEFAULT_MAX_HUMAN_HEIGHT_M)
    )
    # Play-time encounter cap, toggled from the viser panel. Default off so
    # training and every recorded benchmark keep the sampled approach speeds.
    self.limit_approach_speed_to_robot = bool(
      params.get("limit_approach_speed_to_robot", False)
    )
    self.approach_speed_floor_mps = float(
      params.get("approach_speed_floor_mps", DEFAULT_APPROACH_SPEED_FLOOR_MPS)
    )
    self.robot_speed_half_life_s = float(
      params.get("robot_speed_half_life_s", DEFAULT_ROBOT_SPEED_HALF_LIFE_S)
    )
    self.show_mesh = bool(params.get("show_mesh", False))
    self.print_velocity = bool(params.get("print_velocity", False))
    self.velocity_print_interval_s = float(
      params.get("velocity_print_interval_s", 0.5)
    )
    self.use_shared_obstacle_free_mask = bool(
      params.get("use_shared_obstacle_free_mask", False)
    )
    mesh_skin_path = params.get("mesh_skin_path")
    self.mesh_skin_path = (
      Path(str(mesh_skin_path)) if mesh_skin_path is not None else None
    )
    if self.show_mesh and self.mesh_skin_path is None:
      raise ValueError("show_mesh requires mesh_skin_path")
    if self.velocity_print_interval_s <= 0.0:
      raise ValueError("human velocity print interval must be positive")
    if self.approach_speed_floor_mps <= 0.0:
      raise ValueError("approach speed floor must be positive")
    if self.robot_speed_half_life_s <= 0.0:
      raise ValueError("robot speed half-life must be positive")
    if not (0.0 <= self.min_intersection_delay_s <= self.max_intersection_delay_s):
      raise ValueError("invalid human intersection delay range")
    if not 0.0 < self.min_crossing_angle_rad <= self.max_crossing_angle_rad:
      raise ValueError("invalid human crossing angle range")
    if not 0.0 < self.min_human_height_m <= self.max_human_height_m:
      raise ValueError("invalid human height range")
    if (self.min_initial_spawn_radius_m is None) != (
      self.max_initial_spawn_radius_m is None
    ):
      raise ValueError("both initial human spawn-radius bounds are required")
    if self.min_initial_spawn_radius_m is not None and not (
      0.0 < self.min_initial_spawn_radius_m <= self.max_initial_spawn_radius_m
    ):
      raise ValueError("initial human spawn-radius range is invalid")
    self.encounter_sampling = str(params.get("encounter_sampling", "independent"))
    if self.encounter_sampling not in ("independent", "ttc"):
      raise ValueError("encounter_sampling must be 'independent' or 'ttc'")
    self._encounter_sampler: EncounterSampler | None = None
    if self.encounter_sampling == "ttc":
      if (
        self.min_initial_spawn_radius_m is None
        or self.max_initial_spawn_radius_m is None
      ):
        raise ValueError("ttc encounter sampling requires spawn-radius bounds")
      self._encounter_sampler = EncounterSampler(
        env.num_envs,
        env.device,
        spawn_radius_clamp_m=(
          self.min_initial_spawn_radius_m,
          self.max_initial_spawn_radius_m,
        ),
        **encounter_sampler_overrides(params),
      )

    self.bank = SkeletonPathBank(Path(params["skeleton_bank_path"]))
    self.sampler = OnlineComposedHumanSampler(
      self.bank,
      env.num_envs,
      env.device,
      transition_duration_s=float(params.get("transition_duration_s", 0.2)),
      update_hz=float(params.get("update_hz", 10.0)),
      inactive_height_m=HUMAN_INACTIVE_HEIGHT_M,
      retain_joint_poses=self.show_mesh,
    )
    self._viser_skin: SomaViserSkin | None = None
    self._viser_mesh_handles: dict[int, Any] = {}
    self._viser_velocity_label_handles: dict[int, Any] = {}
    self._viser_active_env: int | None = None
    self._viser_mesh_revisions: dict[int, int] = {}
    self._pose_revision = 0
    self._velocity_previous_root_w: torch.Tensor | None = None
    self._velocity_previous_time_s: torch.Tensor | None = None
    self._velocity_valid: torch.Tensor | None = None
    self._velocity_measurement_valid: torch.Tensor | None = None
    self._velocity_w: torch.Tensor | None = None
    self._velocity_closing_speed_mps: torch.Tensor | None = None
    self._velocity_distance_m: torch.Tensor | None = None
    self._velocity_last_print_s = -float("inf")
    self._robot_speed_peak_mps = torch.zeros(env.num_envs, device=env.device)
    if self.print_velocity:
      self._velocity_previous_root_w = torch.zeros(
        (env.num_envs, 3), device=env.device
      )
      self._velocity_previous_time_s = torch.zeros(
        env.num_envs, device=env.device
      )
      self._velocity_valid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      )
      self._velocity_measurement_valid = torch.zeros_like(self._velocity_valid)
      self._velocity_w = torch.zeros((env.num_envs, 3), device=env.device)
      self._velocity_closing_speed_mps = torch.zeros(
        env.num_envs, device=env.device
      )
      self._velocity_distance_m = torch.zeros(env.num_envs, device=env.device)
    transition_index = Path(params["transition_index_path"])
    with np.load(transition_index / "edges.npz", allow_pickle=False) as graph:
      self._action_path_ids = torch.as_tensor(
        np.asarray(graph["action_path_ids"]).copy(),
        dtype=torch.long,
        device=env.device,
      )
      self._entry_path_ids = torch.as_tensor(
        np.asarray(graph["action_entry_connector_ids"]).copy(),
        dtype=torch.long,
        device=env.device,
      )
      self._exit_path_ids = torch.as_tensor(
        np.asarray(graph["action_exit_connector_ids"]).copy(),
        dtype=torch.long,
        device=env.device,
      )
    if self._entry_path_ids.shape != self._exit_path_ids.shape:
      raise ValueError("entry and exit transition tables must have equal shape")
    if self._entry_path_ids.shape[0] != len(self._action_path_ids):
      raise ValueError("transition action and connector table lengths differ")
    graph_tensors = (
      self._action_path_ids,
      self._entry_path_ids,
      self._exit_path_ids,
    )
    entry_rows = self.sampler.source_to_row[self._entry_path_ids]
    entry_last_frames = self.sampler.frame_counts[entry_rows] - 1
    entry_start_xy = self.sampler.local_positions[
      entry_rows, 0, self.sampler.anchor_index, :2
    ].float()
    entry_end_xy = self.sampler.local_positions[
      entry_rows, entry_last_frames, self.sampler.anchor_index, :2
    ].float()
    entry_travel_m = torch.linalg.vector_norm(
      entry_end_xy - entry_start_xy, dim=-1
    ).reshape(-1)
    self._approach_edge_order = torch.argsort(entry_travel_m)
    self._approach_edge_travel_m = entry_travel_m[self._approach_edge_order]
    self.device_storage_bytes = self.sampler.device_storage_bytes + sum(
      tensor.numel() * tensor.element_size() for tensor in graph_tensors
    )

    entity = env.scene[self.human_entity_name]
    expected_names = tuple(spec.name for spec in SOMA_CAPSULE_SPECS)
    expected_bodies = tuple(human_capsule_body_name(name) for name in expected_names)
    self._direct_geom_poses = tuple(entity.body_names) == (HUMAN_RAY_BODY_NAME,)
    self._mocap_ids: torch.Tensor | None = None
    if not self._direct_geom_poses:
      if tuple(entity.body_names) != expected_bodies:
        raise ValueError("runtime human entity body order does not match capsules")
      body_ids = entity.indexing.body_ids.detach().cpu().numpy().astype(np.int64)
      mocap_ids = np.asarray(env.sim.mj_model.body_mocapid)[body_ids]
      if np.any(mocap_ids < 0) or len(np.unique(mocap_ids)) != len(expected_names):
        raise ValueError("every human capsule body must have a unique mocap id")
      self._mocap_ids = torch.as_tensor(
        mocap_ids, dtype=torch.long, device=env.device
      )
    self._geom_ids = entity.indexing.geom_ids.to(dtype=torch.long)
    if self._geom_ids.numel() != len(expected_names):
      raise ValueError("runtime human entity must have one geom per capsule")

  def _deactivate(self, env_ids: torch.Tensor) -> None:
    """Park selected humans below the scene until their next episode reset."""
    if len(env_ids) == 0:
      return
    if self._encounter_sampler is not None:
      # Human-less episodes must never be attributed to an encounter bin.
      self._encounter_sampler.clear_assignments(env_ids)
    poses = self.sampler._poses
    poses.centers_w[env_ids] = 0.0
    poses.centers_w[env_ids, :, 2] = HUMAN_INACTIVE_HEIGHT_M
    poses.radii_m[env_ids] = self.sampler.base_radii
    poses.half_lengths_m[env_ids] = torch.where(
      self.sampler.is_sphere,
      torch.zeros_like(self.sampler.base_radii),
      torch.full_like(self.sampler.base_radii, 0.05),
    )
    poses.active[env_ids] = False
    self.sampler.dirty[env_ids] = False
    self.sampler.scheduled[env_ids] = False
    self.sampler.next_update_times_s[env_ids] = torch.inf
    if self._velocity_valid is not None:
      self._velocity_valid[env_ids] = False
    if self._velocity_measurement_valid is not None:
      self._velocity_measurement_valid[env_ids] = False

  def _global_time_s(self) -> float:
    return float(self._env.common_step_counter) * self._env.step_dt

  def _all_env_ids(self) -> torch.Tensor:
    return torch.arange(self.num_envs, dtype=torch.long, device=self.device)

  def _robot_root_positions_from_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Read the newly written reset root without requiring sim.forward()."""

    robot = self._env.scene[self.robot_entity_name]
    qpos_env_ids = env_ids[:, None]
    return robot.data.data.qpos[qpos_env_ids, robot.indexing.free_joint_q_adr[:3]]

  def _sample_sequences(
    self,
    count: int,
    *,
    minimum_entry_travel_m: torch.Tensor | None = None,
  ) -> torch.Tensor:
    neighbor_count = self._entry_path_ids.shape[1]
    if minimum_entry_travel_m is None:
      action_rows = torch.randint(
        len(self._action_path_ids), (count,), device=self.device
      )
      neighbor_ranks = torch.randint(neighbor_count, (count,), device=self.device)
    else:
      minimum = minimum_entry_travel_m.to(self.device).reshape(count)
      first_eligible = torch.searchsorted(self._approach_edge_travel_m, minimum)
      available = len(self._approach_edge_order) - first_eligible
      # The longest compatible edge is still preferable to injecting root
      # translation if a requested distance exceeds the source bank.
      first_eligible = torch.where(
        available > 0,
        first_eligible,
        torch.full_like(first_eligible, len(self._approach_edge_order) - 1),
      )
      available = available.clamp_min(1)
      selected_rank = torch.floor(
        torch.rand(count, device=self.device) * available
      ).long()
      flat_edge = self._approach_edge_order[first_eligible + selected_rank]
      action_rows = torch.div(flat_edge, neighbor_count, rounding_mode="floor")
      neighbor_ranks = torch.remainder(flat_edge, neighbor_count)
    return torch.stack(
      (
        self._entry_path_ids[action_rows, neighbor_ranks],
        self._action_path_ids[action_rows],
        self._exit_path_ids[action_rows, neighbor_ranks],
      ),
      dim=-1,
    )

  def _place_initial_approach_on_ring(
    self,
    env_ids: torch.Tensor,
    *,
    intersection_delay_s: torch.Tensor,
    body_scale_xyz: torch.Tensor,
    spawn_positions_w: torch.Tensor,
  ) -> None:
    """Use natural source root motion for the boundary-to-action approach."""

    if self.min_initial_spawn_radius_m is None:
      return
    sampler = self.sampler
    count = len(env_ids)
    action_time = sampler.local_intersection_times_s[env_ids]
    target_at_action = sampler.root_anchor_w[env_ids]
    desired_vector = target_at_action[:, :2] - spawn_positions_w[:, :2]
    desired_distance = torch.linalg.vector_norm(desired_vector, dim=-1)

    # Locate the point on the entry walk whose remaining natural root travel
    # matches the boundary-to-action distance. A small coarse search followed
    # by bisection avoids the linear world-space offset that caused foot skate.
    sample_count = 9
    fractions = torch.linspace(0.0, 1.0, sample_count, device=self.device)
    sample_times = action_time[:, None] * fractions[None]
    expanded_env_ids = env_ids[:, None].expand(-1, sample_count).reshape(-1)
    sampled_positions = sampler._sample_chain(
      expanded_env_ids, sample_times.reshape(-1)
    )[0]
    sampled_roots = sampled_positions[:, sampler.anchor_index, :2].reshape(
      count, sample_count, 2
    )
    sampled_roots = sampled_roots * body_scale_xyz[:, None, :2]
    action_root_xy = sampled_roots[:, -1]
    remaining_distance = torch.linalg.vector_norm(
      action_root_xy[:, None] - sampled_roots, dim=-1
    )
    crossings = (remaining_distance[:, :-1] >= desired_distance[:, None]) & (
      remaining_distance[:, 1:] <= desired_distance[:, None]
    )
    has_crossing = crossings.any(dim=-1)
    interval = torch.argmax(crossings.to(torch.int64), dim=-1)
    closest = torch.argmin(
      torch.abs(remaining_distance - desired_distance[:, None]), dim=-1
    )
    lower_time = sample_times.gather(1, interval[:, None]).squeeze(1)
    upper_time = sample_times.gather(1, (interval + 1)[:, None]).squeeze(1)
    closest_time = sample_times.gather(1, closest[:, None]).squeeze(1)
    lower_time = torch.where(has_crossing, lower_time, closest_time)
    upper_time = torch.where(has_crossing, upper_time, closest_time)
    for _ in range(6):
      midpoint = 0.5 * (lower_time + upper_time)
      midpoint_positions = sampler._sample_chain(env_ids, midpoint)[0]
      midpoint_root = (
        midpoint_positions[:, sampler.anchor_index, :2] * body_scale_xyz[:, :2]
      )
      midpoint_distance = torch.linalg.vector_norm(
        action_root_xy - midpoint_root, dim=-1
      )
      move_lower = has_crossing & (midpoint_distance >= desired_distance)
      lower_time = torch.where(move_lower, midpoint, lower_time)
      upper_time = torch.where(has_crossing & ~move_lower, midpoint, upper_time)
    initial_time = 0.5 * (lower_time + upper_time)
    playback_speed = (action_time - initial_time) / intersection_delay_s.clamp_min(
      self._env.step_dt
    )
    sampler.playback_speed[env_ids] = playback_speed.clamp_min(0.05)

    initial_positions, initial_quaternions, _ = sampler._sample_chain(
      env_ids, initial_time
    )
    action_positions = sampler._sample_chain(env_ids, action_time)[0]
    initial_positions = initial_positions * body_scale_xyz[:, None]
    action_positions = action_positions * body_scale_xyz[:, None]
    source_vector = (
      action_positions[:, sampler.anchor_index, :2]
      - initial_positions[:, sampler.anchor_index, :2]
    )
    source_heading = torch.atan2(source_vector[:, 1], source_vector[:, 0])
    forward = _quat_apply(
      initial_quaternions[:, sampler.anchor_index],
      torch.tensor((0.0, 1.0, 0.0), device=self.device).expand(count, 3),
    )
    facing_heading = torch.atan2(forward[:, 1], forward[:, 0])
    source_heading = torch.where(
      torch.linalg.vector_norm(source_vector, dim=-1) < 0.03,
      facing_heading,
      source_heading,
    )
    desired_heading = torch.atan2(desired_vector[:, 1], desired_vector[:, 0])
    placement_yaw = desired_heading - source_heading
    placement_yaw = torch.atan2(torch.sin(placement_yaw), torch.cos(placement_yaw))

    initial_root = initial_positions[:, sampler.anchor_index]
    rotated_initial_root = _rotate_xy(initial_root, placement_yaw)
    foot_ground = initial_positions[:, sampler.foot_indices, 2].min(dim=-1).values
    ground = self._env.scene.env_origins[env_ids, 2]
    translation = torch.cat(
      (
        spawn_positions_w[:, :2] - rotated_initial_root[:, :2],
        (ground - foot_ground).unsqueeze(-1),
      ),
      dim=-1,
    )

    sampler.placement_yaw[env_ids] = placement_yaw
    sampler.translation_w[env_ids] = translation
    sampler.current_translation_w[env_ids] = translation
    sampler.dirty[env_ids] = True

  def _update_robot_speed_peak(self) -> None:
    """Advance the decayed peak of the robot's planar speed by one step."""
    robot = self._env.scene[self.robot_entity_name]
    planar_speed = torch.linalg.vector_norm(
      robot.data.root_link_lin_vel_w[:, :2], dim=-1
    )
    decay = 0.5 ** (self._env.step_dt / self.robot_speed_half_life_s)
    self._robot_speed_peak_mps, _ = advance_robot_speed_cap(
      self._robot_speed_peak_mps,
      planar_speed,
      decay=decay,
      floor=self.approach_speed_floor_mps,
    )

  def _speed_limited_delay_steps(
    self,
    env_ids: torch.Tensor,
    delay_steps: torch.Tensor,
    spawn_radius_m: torch.Tensor,
  ) -> torch.Tensor:
    """Stretch the scheduled delay so the human never outruns the robot."""
    if not self.limit_approach_speed_to_robot:
      return delay_steps
    cap = self._robot_speed_peak_mps[env_ids].clamp_min(self.approach_speed_floor_mps)
    limited_s = limit_intersection_delay_to_speed(
      delay_s=delay_steps.float() * self._env.step_dt,
      spawn_radius_m=spawn_radius_m,
      max_speed_mps=cap,
    )
    return torch.round(limited_s / self._env.step_dt).long().clamp_min(1)

  def _schedule(
    self,
    env_ids: torch.Tensor,
    global_time_s: float,
    collided_since_last: torch.Tensor | None = None,
  ) -> None:
    if env_ids.numel() == 0:
      return
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    if self._velocity_valid is not None:
      # A newly composed sequence may start elsewhere in world space. Do not
      # mistake that reset/reschedule discontinuity for physical velocity.
      self._velocity_valid[env_ids] = False
    if self._velocity_measurement_valid is not None:
      self._velocity_measurement_valid[env_ids] = False
    count = len(env_ids)

    # TTC-mode draws replace only the two independent uniforms; the sampler
    # runs up front so the independent path's RNG consumption is unchanged.
    # A missing flag tensor means a mid-episode reschedule: a collision would
    # have reset the environment instead, so no collision occurred.
    sampled = None
    if self._encounter_sampler is not None:
      if collided_since_last is None:
        collided_since_last = torch.zeros(count, dtype=torch.bool, device=self.device)
      sampled = self._encounter_sampler.sample(
        env_ids, global_time_s, collided_since_last
      )
    if sampled is None:
      requested_delay = torch.empty(count, device=self.device).uniform_(
        self.min_intersection_delay_s,
        self.max_intersection_delay_s,
      )
    else:
      requested_delay = sampled[0]
    requested_steps = torch.round(requested_delay / self._env.step_dt).long()
    delay_steps = requested_steps.clamp_min(1)
    actual_delay_s = delay_steps.float() * self._env.step_dt
    target_positions, target_yaw, target_heading = (
      robot_intersection_target_from_qpos(
        self._env,
        env_ids,
        self.robot_entity_name,
      )
    )

    angle = torch.empty(count, device=self.device).uniform_(
      self.min_crossing_angle_rad, self.max_crossing_angle_rad
    )
    side = torch.where(
      torch.rand(count, device=self.device) < 0.5,
      -torch.ones(count, device=self.device),
      torch.ones(count, device=self.device),
    )
    body_scale = sample_body_scales_for_height(
      count,
      self.device,
      min_height_m=self.min_human_height_m,
      max_height_m=self.max_human_height_m,
    )
    robot_positions = self._robot_root_positions_from_qpos(env_ids)
    spawn_angle = torch.rand(count, device=self.device) * (2.0 * torch.pi)
    if sampled is not None:
      spawn_radius = sampled[2]
    elif self.min_initial_spawn_radius_m is None:
      spawn_radius = torch.zeros(count, device=self.device)
    else:
      assert self.max_initial_spawn_radius_m is not None
      spawn_radius = torch.empty(count, device=self.device).uniform_(
        self.min_initial_spawn_radius_m, self.max_initial_spawn_radius_m
      )
    delay_steps = self._speed_limited_delay_steps(env_ids, delay_steps, spawn_radius)
    actual_delay_s = delay_steps.float() * self._env.step_dt
    spawn_positions = robot_positions.clone()
    spawn_positions[:, 0] += spawn_radius * torch.cos(spawn_angle)
    spawn_positions[:, 1] += spawn_radius * torch.sin(spawn_angle)
    minimum_entry_travel = spawn_radius / body_scale[:, :2].amin(dim=-1) + 0.5
    self.sampler.schedule_intersections(
      env_ids,
      sequence_source_ids=self._sample_sequences(
        count, minimum_entry_travel_m=minimum_entry_travel
      ),
      global_intersection_times_s=global_time_s + actual_delay_s,
      robot_positions_at_intersection_w=target_positions,
      robot_yaw_at_intersection=target_yaw,
      robot_path_heading_at_intersection=target_heading,
      # Intersect at the center of the annotated punch/kick segment. The two
      # walking connectors only provide a continuous approach and departure.
      action_phase=0.5,
      crossing_angle_rad=angle * side,
      ground_height_m=self._env.scene.env_origins[env_ids, 2],
      body_scale_xyz=body_scale,
      radius_scale=torch.empty(count, device=self.device).uniform_(0.92, 1.08),
      radius_margin_m=torch.empty(count, device=self.device).uniform_(0.0, 0.025),
    )
    self._place_initial_approach_on_ring(
      env_ids,
      intersection_delay_s=actual_delay_s,
      body_scale_xyz=body_scale,
      spawn_positions_w=spawn_positions,
    )

  def _write_updated_poses(
    self,
    global_time_s: float,
    *,
    force_ids: torch.Tensor | None = None,
  ) -> None:
    poses = self.sampler.sample_held(global_time_s)
    env_ids = self.sampler.last_updated_env_ids
    if force_ids is not None:
      env_ids = torch.unique(torch.cat((env_ids, force_ids)))
    if len(env_ids) == 0:
      return
    self._update_velocity_diagnostic(env_ids, global_time_s)
    env_grid, geom_grid = torch.meshgrid(env_ids, self._geom_ids, indexing="ij")
    if self._direct_geom_poses:
      self._env.sim.model.geom_pos[env_grid, geom_grid] = poses.centers_w[env_ids]
      self._env.sim.model.geom_quat[env_grid, geom_grid] = (
        poses.quaternions_wxyz[env_ids]
      )
    else:
      assert self._mocap_ids is not None
      env_grid, mocap_grid = torch.meshgrid(
        env_ids, self._mocap_ids, indexing="ij"
      )
      self._env.sim.data.mocap_pos[env_grid, mocap_grid] = poses.centers_w[env_ids]
      self._env.sim.data.mocap_quat[env_grid, mocap_grid] = (
        poses.quaternions_wxyz[env_ids]
      )

    radii = poses.radii_m[env_ids]
    half_lengths = poses.half_lengths_m[env_ids]
    model = self._env.sim.model
    model.geom_size[env_grid, geom_grid] = torch.stack(
      (radii, half_lengths, torch.zeros_like(radii)), dim=-1
    )
    model.geom_rbound[env_grid, geom_grid] = radii + half_lengths
    model.geom_aabb[env_grid, geom_grid, 0] = 0.0
    model.geom_aabb[env_grid, geom_grid, 1] = torch.stack(
      (radii, radii, radii + half_lengths), dim=-1
    )
    self._pose_revision += 1

  def _update_velocity_diagnostic(
    self,
    env_ids: torch.Tensor,
    global_time_s: float,
  ) -> None:
    """Measure velocity between actual held-pose updates and print env zero."""
    if not self.print_velocity:
      return
    assert self._velocity_previous_root_w is not None
    assert self._velocity_previous_time_s is not None
    assert self._velocity_valid is not None
    assert self._velocity_measurement_valid is not None
    assert self._velocity_w is not None
    assert self._velocity_closing_speed_mps is not None
    assert self._velocity_distance_m is not None

    roots_w = self.sampler._poses.root_positions_w[env_ids]  # noqa: SLF001
    active = self.sampler._poses.active[env_ids]  # noqa: SLF001
    elapsed_s = global_time_s - self._velocity_previous_time_s[env_ids]
    valid = self._velocity_valid[env_ids] & active & (elapsed_s > 1.0e-6)
    velocities_w = (
      roots_w - self._velocity_previous_root_w[env_ids]
    ) / elapsed_s[:, None].clamp_min(1.0e-6)
    robot_data = self._env.scene[self.robot_entity_name].data
    toward_robot_xy = robot_data.root_link_pos_w[env_ids, :2] - roots_w[:, :2]
    distance_xy = torch.linalg.vector_norm(toward_robot_xy, dim=-1)
    relative_velocity_xy = (
      velocities_w[:, :2] - robot_data.root_link_lin_vel_w[env_ids, :2]
    )
    closing_speed = torch.sum(
      relative_velocity_xy
      * toward_robot_xy
      / distance_xy[:, None].clamp_min(1.0e-6),
      dim=-1,
    )
    self._velocity_measurement_valid[env_ids] = valid
    self._velocity_w[env_ids] = torch.where(
      valid[:, None], velocities_w, torch.zeros_like(velocities_w)
    )
    self._velocity_closing_speed_mps[env_ids] = torch.where(
      valid, closing_speed, 0.0
    )
    self._velocity_distance_m[env_ids] = distance_xy

    env_zero_row = (env_ids == 0).nonzero().flatten()
    due_to_print = (
      global_time_s - self._velocity_last_print_s
      >= self.velocity_print_interval_s
    )
    if len(env_zero_row) and due_to_print:
      row = int(env_zero_row[0])
      if bool(valid[row]):
        velocity = velocities_w[row]
        speed_xy = torch.linalg.vector_norm(velocity[:2])
        vx, vy, vz, speed, closing, distance, playback = (
          float(value)
          for value in (
            velocity[0],
            velocity[1],
            velocity[2],
            speed_xy,
            closing_speed[row],
            distance_xy[row],
            self.sampler.playback_speed[0],
          )
        )
        print(
          "[primary-human] "
          f"v_w=({vx:+.2f}, {vy:+.2f}, {vz:+.2f}) m/s  "
          f"speed_xy={speed:.2f} m/s  closing={closing:+.2f} m/s  "
          f"distance={distance:.2f} m  playback={playback:.2f}x",
          flush=True,
        )
        self._velocity_last_print_s = global_time_s

    self._velocity_previous_root_w[env_ids] = roots_w
    self._velocity_previous_time_s[env_ids] = global_time_s
    self._velocity_valid[env_ids] = active

  def _velocity_label_text(self, env_index: int) -> str:
    assert self._velocity_w is not None
    assert self._velocity_closing_speed_mps is not None
    assert self._velocity_distance_m is not None
    velocity = self._velocity_w[env_index]
    speed_xy = torch.linalg.vector_norm(velocity[:2])
    vx, vy, vz, speed, closing, distance = (
      float(value)
      for value in (
        velocity[0],
        velocity[1],
        velocity[2],
        speed_xy,
        self._velocity_closing_speed_mps[env_index],
        self._velocity_distance_m[env_index],
      )
    )
    return (
      f"human v=({vx:+.2f}, {vy:+.2f}, {vz:+.2f}) m/s\n"
      f"speed={speed:.2f}  closing={closing:+.2f}  distance={distance:.2f} m"
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      resolved = self._all_env_ids()
    else:
      resolved = env_ids.to(device=self.device, dtype=torch.long)
    active = resolved
    inactive = torch.empty(0, dtype=torch.long, device=self.device)
    if self.use_shared_obstacle_free_mask:
      shared_mask = getattr(self._env, "_safe_mimic_obstacle_free_envs", None)
      if shared_mask is None:
        raise RuntimeError("shared obstacle-free mask was not initialized")
      inactive = resolved[shared_mask[resolved]]
      active = resolved[~shared_mask[resolved]]
    # Event resets run after terminations are computed and before the
    # termination manager resets, so the term buffers still describe the
    # episodes that just ended. Environments going human-less attribute their
    # pending outcome here instead of at a schedule they will not receive.
    active_collided: torch.Tensor | None = None
    if self._encounter_sampler is not None:
      collided = read_collision_terms(
        self._env.termination_manager, self.num_envs, self.device
      )
      self._encounter_sampler.observe_terminal(inactive, collided[inactive])
      active_collided = collided[active]
    now = self._global_time_s()
    self._deactivate(inactive)
    self._schedule(active, now, collided_since_last=active_collided)
    self._write_updated_poses(now, force_ids=inactive)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **_: object,
  ) -> None:
    del env, env_ids
    self._update_robot_speed_peak()
    now = self._global_time_s()
    expired = self.sampler.expired_env_ids(now)
    if len(expired):
      self._schedule(expired, now)
    self._write_updated_poses(now)

  def _load_viser_skin(self) -> SomaViserSkin:
    if self._viser_skin is None:
      assert self.mesh_skin_path is not None
      self._viser_skin = prepare_soma_viser_skin(
        load_soma_mesh_skin(self.mesh_skin_path), self.bank.joint_names
      )
    return self._viser_skin

  def debug_vis(self, visualizer: Any) -> None:
    """Render one posed SOMA mesh for the selected Viser environment."""
    if not self.show_mesh or not hasattr(visualizer, "server"):
      return
    env_index = int(visualizer.env_idx)
    if not 0 <= env_index < self.num_envs:
      return
    joint_positions = self.sampler.joint_positions_local_m
    joint_quaternions = self.sampler.joint_quaternions_local_wxyz
    if joint_positions is None or joint_quaternions is None:
      return

    if self._viser_active_env is not None and self._viser_active_env != env_index:
      previous = self._viser_mesh_handles.get(self._viser_active_env)
      if previous is not None:
        previous.visible = False
      previous_label = self._viser_velocity_label_handles.get(
        self._viser_active_env
      )
      if previous_label is not None:
        previous_label.visible = False
    self._viser_active_env = env_index

    skin = self._load_viser_skin()
    handle = self._viser_mesh_handles.get(env_index)
    scene_offset = np.asarray(
      getattr(visualizer, "_scene_offset", np.zeros(3)), dtype=np.float32
    )
    if handle is None or self._viser_mesh_revisions.get(env_index) != (
      self._pose_revision
    ):
      positions = joint_positions[env_index].detach().cpu().numpy()
      quaternions = joint_quaternions[env_index].detach().cpu().numpy()
      body_scale = self.sampler.body_scale_xyz[env_index].detach().cpu().numpy()
      yaw = float(self.sampler.placement_yaw[env_index])
      translation = self.sampler.current_translation_w[env_index].detach().cpu().numpy()
      vertices_w = skin.skin_vertices_mujoco(
        positions,
        quaternions,
        body_scale_xyz=body_scale,
        placement_yaw=yaw,
        translation_w=translation + scene_offset,
      )
      if handle is None:
        handle = visualizer.server.scene.add_mesh_simple(
          f"/safe_mimic/human_mesh/env_{env_index}",
          vertices_w,
          skin.triangles,
          color=(205, 158, 121),
          opacity=1.0,
          material="toon3",
          side="double",
          cast_shadow=False,
          receive_shadow=False,
        )
        self._viser_mesh_handles[env_index] = handle
      else:
        handle.vertices = vertices_w
      self._viser_mesh_revisions[env_index] = self._pose_revision
    human_active = bool(self.sampler._poses.active[env_index])
    handle.visible = human_active

    if self.print_velocity:
      assert self._velocity_measurement_valid is not None
      assert self._velocity_distance_m is not None
      label = self._viser_velocity_label_handles.get(env_index)
      root_w = (
        self.sampler._poses.root_positions_w[env_index].detach().cpu().numpy()
      )
      body_scale_z = float(self.sampler.body_scale_xyz[env_index, 2])
      label_position = root_w + scene_offset + np.array(
        (0.0, 0.0, 1.15 * body_scale_z), dtype=np.float32
      )
      label_text = self._velocity_label_text(env_index)
      if label is None:
        label = visualizer.server.scene.add_label(
          f"/safe_mimic/human_velocity/env_{env_index}",
          label_text,
          position=label_position,
          font_size_mode="screen",
          font_screen_scale=0.85,
          depth_test=False,
          anchor="bottom-center",
        )
        self._viser_velocity_label_handles[env_index] = label
      else:
        label.text = label_text
        label.position = label_position
      label.visible = human_active and bool(
        self._velocity_measurement_valid[env_index]
      )
