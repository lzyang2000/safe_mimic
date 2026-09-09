"""Fast GPU playback events for offline-compiled human trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase

from safe_mimic.assets.soma_capsules import (
  HUMAN_CROWD_CAPACITY,
  HUMAN_CROWD_RAY_BODY_NAME,
  HUMAN_INACTIVE_HEIGHT_M,
  HUMAN_RAY_BODY_NAME,
  crowd_capsule_body_name,
  human_capsule_body_name,
)
from safe_mimic.motions.annular_crowd import sample_annular_crowd
from safe_mimic.motions.human_capsules import (
  DEFAULT_MAX_HUMAN_HEIGHT_M,
  DEFAULT_MIN_HUMAN_HEIGHT_M,
  SOMA_CAPSULE_SPECS,
  SOMA_CROWD_PROXY_SPECS,
  sample_body_scales_for_height,
)
from safe_mimic.motions.packed_human_trajectory import (
  PackedHumanTrajectoryBank,
  PackedHumanTrajectorySampler,
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


class _PackedHumanEventBase(ManagerTermBase):
  """Common direct-geom/mocap binding and sparse pose writes."""

  sampler: PackedHumanTrajectorySampler
  _geom_ids: torch.Tensor
  _mocap_ids: torch.Tensor | None
  _direct_geom_poses: bool

  def _global_time_s(self) -> float:
    return float(self._env.common_step_counter) * self._env.step_dt

  def _all_env_ids(self) -> torch.Tensor:
    return torch.arange(self.num_envs, dtype=torch.long, device=self.device)

  def _write_primary_poses(
    self,
    now: float,
    *,
    force_ids: torch.Tensor | None = None,
    size_ids: torch.Tensor | None = None,
  ) -> None:
    poses = self.sampler.sample_held(now, force_ids=force_ids)
    env_ids = self.sampler.last_updated_ids
    if force_ids is not None and force_ids.numel():
      env_ids = torch.unique(torch.cat((env_ids, force_ids)))
    if env_ids.numel():
      env_grid, geom_grid = torch.meshgrid(env_ids, self._geom_ids, indexing="ij")
      if self._direct_geom_poses:
        model = self._env.sim.model
        model.geom_pos[env_grid, geom_grid] = poses.centers_w[env_ids]
        model.geom_quat[env_grid, geom_grid] = poses.quaternions_wxyz[env_ids]
      else:
        assert self._mocap_ids is not None
        env_grid, mocap_grid = torch.meshgrid(env_ids, self._mocap_ids, indexing="ij")
        self._env.sim.data.mocap_pos[env_grid, mocap_grid] = poses.centers_w[env_ids]
        self._env.sim.data.mocap_quat[env_grid, mocap_grid] = poses.quaternions_wxyz[
          env_ids
        ]
    resize_ids = env_ids
    if size_ids is not None and size_ids.numel():
      resize_ids = torch.unique(torch.cat((resize_ids, size_ids)))
    if resize_ids.numel():
      env_grid, geom_grid = torch.meshgrid(resize_ids, self._geom_ids, indexing="ij")
      self._write_sizes(
        env_grid,
        geom_grid,
        poses.radii_m[resize_ids],
        poses.half_lengths_m[resize_ids],
      )

  def _write_crowd_poses(
    self,
    now: float,
    capacity: int,
    *,
    force_ids: torch.Tensor | None = None,
    size_ids: torch.Tensor | None = None,
  ) -> None:
    poses = self.sampler.sample_held(now, force_ids=force_ids)
    agent_ids = self.sampler.last_updated_ids
    if force_ids is not None and force_ids.numel():
      agent_ids = torch.unique(torch.cat((agent_ids, force_ids)))
    if agent_ids.numel():
      env_ids = torch.div(agent_ids, capacity, rounding_mode="floor")
      crowd_ids = torch.remainder(agent_ids, capacity)
      geom_ids = self._geom_ids[crowd_ids]
      geom_env_grid = env_ids[:, None].expand_as(geom_ids)
      if self._direct_geom_poses:
        model = self._env.sim.model
        model.geom_pos[geom_env_grid, geom_ids] = poses.centers_w[agent_ids]
        model.geom_quat[geom_env_grid, geom_ids] = poses.quaternions_wxyz[agent_ids]
      else:
        assert self._mocap_ids is not None
        mocap_ids = self._mocap_ids[crowd_ids]
        mocap_env_grid = env_ids[:, None].expand_as(mocap_ids)
        self._env.sim.data.mocap_pos[mocap_env_grid, mocap_ids] = poses.centers_w[
          agent_ids
        ]
        self._env.sim.data.mocap_quat[mocap_env_grid, mocap_ids] = (
          poses.quaternions_wxyz[agent_ids]
        )
    resize_ids = agent_ids
    if size_ids is not None and size_ids.numel():
      resize_ids = torch.unique(torch.cat((resize_ids, size_ids)))
    if resize_ids.numel():
      env_ids = torch.div(resize_ids, capacity, rounding_mode="floor")
      crowd_ids = torch.remainder(resize_ids, capacity)
      geom_ids = self._geom_ids[crowd_ids]
      geom_env_grid = env_ids[:, None].expand_as(geom_ids)
      self._write_sizes(
        geom_env_grid,
        geom_ids,
        poses.radii_m[resize_ids],
        poses.half_lengths_m[resize_ids],
      )

  def _write_sizes(
    self,
    env_grid: torch.Tensor,
    geom_grid: torch.Tensor,
    radii: torch.Tensor,
    half_lengths: torch.Tensor,
  ) -> None:
    model = self._env.sim.model
    model.geom_size[env_grid, geom_grid] = torch.stack(
      (radii, half_lengths, torch.zeros_like(radii)), dim=-1
    )
    model.geom_rbound[env_grid, geom_grid] = radii + half_lengths
    model.geom_aabb[env_grid, geom_grid, 0] = 0.0
    model.geom_aabb[env_grid, geom_grid, 1] = torch.stack(
      (radii, radii, radii + half_lengths), dim=-1
    )


@requires_model_fields("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb")
class PackedHumanCapsuleMotion(_PackedHumanEventBase):
  """Place a prebuilt walk-action-walk path across the robot's live pose."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    params = cfg.params
    self.human_entity_name = str(params["human_entity"])
    self.robot_entity_name = str(params.get("robot_entity", "robot"))
    self.command_name = str(params.get("command_name", "motion"))
    self.min_initial_spawn_radius_m = float(
      params.get("min_initial_spawn_radius_m", 2.0)
    )
    self.max_initial_spawn_radius_m = float(
      params.get("max_initial_spawn_radius_m", 4.0)
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
    self.use_shared_obstacle_free_mask = bool(
      params.get("use_shared_obstacle_free_mask", False)
    )
    if not (0.0 < self.min_initial_spawn_radius_m <= self.max_initial_spawn_radius_m):
      raise ValueError("invalid initial human spawn-radius range")
    if not (0.0 <= self.min_intersection_delay_s <= self.max_intersection_delay_s):
      raise ValueError("invalid human intersection delay range")
    if not 0.0 < self.min_crossing_angle_rad <= self.max_crossing_angle_rad:
      raise ValueError("invalid human crossing angle range")
    if not 0.0 < self.min_human_height_m <= self.max_human_height_m:
      raise ValueError("invalid human height range")
    self.encounter_sampling = str(params.get("encounter_sampling", "independent"))
    if self.encounter_sampling not in ("independent", "ttc"):
      raise ValueError("encounter_sampling must be 'independent' or 'ttc'")
    self._encounter_sampler: EncounterSampler | None = None
    if self.encounter_sampling == "ttc":
      self._encounter_sampler = EncounterSampler(
        env.num_envs,
        env.device,
        spawn_radius_clamp_m=(
          self.min_initial_spawn_radius_m,
          self.max_initial_spawn_radius_m,
        ),
        **encounter_sampler_overrides(params),
      )

    self.bank = PackedHumanTrajectoryBank(params["packed_bank_path"])
    if self.bank.config.get("kind") != "primary":
      raise ValueError("primary event requires a primary packed trajectory bank")
    self.sampler = PackedHumanTrajectorySampler(
      self.bank,
      env.num_envs,
      env.device,
      update_hz=float(params.get("update_hz", 10.0)),
      inactive_height_m=HUMAN_INACTIVE_HEIGHT_M,
    )
    self._approach_path_order = torch.argsort(self.sampler.entry_travel_m)
    self._approach_path_travel_m = self.sampler.entry_travel_m[
      self._approach_path_order
    ]
    self.device_storage_bytes = self.sampler.device_storage_bytes

    entity = env.scene[self.human_entity_name]
    expected_names = tuple(spec.name for spec in SOMA_CAPSULE_SPECS)
    bank_names = tuple(spec.name for spec in self.bank.capsule_specs)
    if bank_names != expected_names:
      raise ValueError("packed primary capsule order does not match the entity")
    expected_bodies = tuple(human_capsule_body_name(name) for name in expected_names)
    self._direct_geom_poses = tuple(entity.body_names) == (HUMAN_RAY_BODY_NAME,)
    self._mocap_ids = None
    if not self._direct_geom_poses:
      if tuple(entity.body_names) != expected_bodies:
        raise ValueError("runtime primary-human body order does not match capsules")
      body_ids = entity.indexing.body_ids.detach().cpu().numpy().astype(np.int64)
      mocap_ids = np.asarray(env.sim.mj_model.body_mocapid)[body_ids]
      if np.any(mocap_ids < 0) or len(np.unique(mocap_ids)) != len(expected_names):
        raise ValueError("every primary-human capsule must have a unique mocap id")
      self._mocap_ids = torch.as_tensor(mocap_ids, dtype=torch.long, device=env.device)
    self._geom_ids = entity.indexing.geom_ids.to(dtype=torch.long)
    if self._geom_ids.numel() != len(expected_names):
      raise ValueError("runtime primary human must have one geom per capsule")

  def _sample_paths(self, minimum_travel_m: torch.Tensor) -> torch.Tensor:
    first_eligible = torch.searchsorted(self._approach_path_travel_m, minimum_travel_m)
    available = len(self.bank) - first_eligible
    first_eligible = torch.where(
      available > 0,
      first_eligible,
      torch.full_like(first_eligible, len(self.bank) - 1),
    )
    available = available.clamp_min(1)
    rank = torch.floor(torch.rand_like(minimum_travel_m) * available).long()
    return self._approach_path_order[first_eligible + rank]

  def _schedule(
    self,
    env_ids: torch.Tensor,
    now: float,
    collided_since_last: torch.Tensor | None = None,
  ) -> None:
    if env_ids.numel() == 0:
      return
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    count = len(env_ids)

    # TTC-mode draws replace only the two independent uniforms; the sampler
    # runs up front so the independent path's RNG consumption is unchanged.
    # A missing flag tensor means a mid-episode reschedule: a collision would
    # have reset the environment instead, so no collision occurred.
    sampled = None
    if self._encounter_sampler is not None:
      if collided_since_last is None:
        collided_since_last = torch.zeros(count, dtype=torch.bool, device=self.device)
      sampled = self._encounter_sampler.sample(env_ids, now, collided_since_last)
    if sampled is None:
      requested_delay = torch.empty(count, device=self.device).uniform_(
        self.min_intersection_delay_s,
        self.max_intersection_delay_s,
      )
    else:
      requested_delay = sampled[0]
    delay_steps = torch.round(requested_delay / self._env.step_dt).long().clamp_min(1)
    actual_delay_s = delay_steps.float() * self._env.step_dt
    target_positions, _, target_heading = robot_intersection_target_from_qpos(
      self._env,
      env_ids,
      self.robot_entity_name,
    )

    body_scale = sample_body_scales_for_height(
      count,
      self.device,
      min_height_m=self.min_human_height_m,
      max_height_m=self.max_human_height_m,
    )
    if sampled is None:
      spawn_radius = torch.empty(count, device=self.device).uniform_(
        self.min_initial_spawn_radius_m,
        self.max_initial_spawn_radius_m,
      )
    else:
      spawn_radius = sampled[2]
    source_distance = spawn_radius / body_scale[:, :2].mean(dim=-1)
    path_ids = self._sample_paths(source_distance + 0.1)
    action_times = self.sampler.action_times_s(path_ids)
    entry_times = self.sampler.entry_times_for_distance(path_ids, source_distance)
    playback_speed = (action_times - entry_times) / actual_delay_s

    crossing_angle = torch.empty(count, device=self.device).uniform_(
      self.min_crossing_angle_rad,
      self.max_crossing_angle_rad,
    )
    crossing_side = torch.where(
      torch.rand(count, device=self.device) < 0.5,
      -torch.ones(count, device=self.device),
      torch.ones(count, device=self.device),
    )
    self.sampler.schedule_intersections(
      env_ids,
      path_ids=path_ids,
      global_intersection_times_s=now + actual_delay_s,
      local_intersection_times_s=action_times,
      target_positions_w=target_positions,
      target_heading=target_heading,
      crossing_angle_rad=crossing_angle * crossing_side,
      ground_height_m=self._env.scene.env_origins[env_ids, 2],
      body_scale_xyz=body_scale,
      radius_scale=torch.empty(count, device=self.device).uniform_(0.92, 1.08),
      radius_margin_m=torch.empty(count, device=self.device).uniform_(0.0, 0.025),
      playback_speed=playback_speed.clamp_min(0.05),
      heading_start_times_s=entry_times,
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
    self.sampler.deactivate(inactive)
    now = self._global_time_s()
    self._schedule(active, now, collided_since_last=active_collided)
    self._write_primary_poses(
      now,
      force_ids=resolved,
      size_ids=resolved,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **_: object,
  ) -> None:
    del env, env_ids
    now = self._global_time_s()
    expired = self.sampler.expired_ids(now)
    self._schedule(expired, now)
    self._write_primary_poses(
      now,
      force_ids=expired,
      size_ids=expired,
    )


@requires_model_fields("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb")
class PackedHumanCapsuleCrowdMotion(_PackedHumanEventBase):
  """Animate a variable-density crowd from a packed stationary-motion bank."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    params = cfg.params
    self.human_entity_name = str(params["human_entity"])
    self.robot_entity_name = str(params.get("robot_entity", "robot"))
    self.capacity = int(params.get("capacity", HUMAN_CROWD_CAPACITY))
    self.min_count = int(params.get("min_count", 0))
    self.max_count = int(params.get("max_count", self.capacity))
    self.min_radius_m = float(params.get("min_radius_m", 2.0))
    self.max_radius_m = float(params.get("max_radius_m", 4.0))
    spacing = params.get("target_arc_spacing_m")
    self.target_arc_spacing_m = None if spacing is None else float(spacing)
    self.randomize_density = bool(params.get("randomize_density", True))
    self.obstacle_free_probability = float(params.get("obstacle_free_probability", 0.0))
    self.radial_jitter_m = float(params.get("radial_jitter_m", 0.0))
    self.min_shape_exponent = float(params.get("min_shape_exponent", 2.0))
    self.max_shape_exponent = float(params.get("max_shape_exponent", 2.0))
    self.angular_jitter_fraction = float(params.get("angular_jitter_fraction", 0.0))
    self.inward_facing_probability = float(params.get("inward_facing_probability", 1.0))
    self.inward_facing_jitter_rad = float(params.get("inward_facing_jitter_rad", 0.0))
    self.facing_yaw_offset_rad = float(params.get("facing_yaw_offset_rad", 0.0))
    self.min_playback_speed = float(params.get("min_playback_speed", 0.8))
    self.max_playback_speed = float(params.get("max_playback_speed", 1.2))
    self.min_human_height_m = float(
      params.get("min_human_height_m", DEFAULT_MIN_HUMAN_HEIGHT_M)
    )
    self.max_human_height_m = float(
      params.get("max_human_height_m", DEFAULT_MAX_HUMAN_HEIGHT_M)
    )
    if self.capacity != HUMAN_CROWD_CAPACITY:
      raise ValueError("event capacity must match the compiled crowd asset")
    if not 0 <= self.min_count <= self.max_count <= self.capacity:
      raise ValueError("invalid crowd density range")
    if not 0.0 <= self.obstacle_free_probability <= 1.0:
      raise ValueError("obstacle-free probability must be in [0, 1]")
    if not 0.0 < self.min_playback_speed <= self.max_playback_speed:
      raise ValueError("invalid crowd playback-speed range")
    if not 0.0 < self.min_human_height_m <= self.max_human_height_m:
      raise ValueError("invalid crowd human-height range")

    self.bank = PackedHumanTrajectoryBank(params["packed_bank_path"])
    if self.bank.config.get("kind") != "crowd":
      raise ValueError("crowd event requires a crowd packed trajectory bank")
    self.agent_count = env.num_envs * self.capacity
    self.sampler = PackedHumanTrajectorySampler(
      self.bank,
      self.agent_count,
      env.device,
      update_hz=float(params.get("update_hz", 5.0)),
      inactive_height_m=HUMAN_INACTIVE_HEIGHT_M,
      loop=True,
      lock_root_xy=True,
    )
    self.device_storage_bytes = self.sampler.device_storage_bytes

    entity = env.scene[self.human_entity_name]
    capsule_names = tuple(spec.name for spec in SOMA_CROWD_PROXY_SPECS)
    bank_names = tuple(spec.name for spec in self.bank.capsule_specs)
    if bank_names != capsule_names:
      raise ValueError("packed crowd capsule order does not match the entity")
    expected_bodies = tuple(
      crowd_capsule_body_name(crowd_index, capsule_name)
      for crowd_index in range(self.capacity)
      for capsule_name in capsule_names
    )
    expected_geom_count = self.capacity * len(capsule_names)
    self._direct_geom_poses = tuple(entity.body_names) == (HUMAN_CROWD_RAY_BODY_NAME,)
    self._mocap_ids = None
    if not self._direct_geom_poses:
      if tuple(entity.body_names) != expected_bodies:
        raise ValueError("runtime crowd body order does not match its capsules")
      body_ids = entity.indexing.body_ids.detach().cpu().numpy().astype(np.int64)
      mocap_ids = np.asarray(env.sim.mj_model.body_mocapid)[body_ids]
      if np.any(mocap_ids < 0) or len(np.unique(mocap_ids)) != expected_geom_count:
        raise ValueError("every crowd capsule body must have a unique mocap id")
      self._mocap_ids = torch.as_tensor(
        mocap_ids.reshape(self.capacity, -1),
        dtype=torch.long,
        device=env.device,
      )
    self._geom_ids = entity.indexing.geom_ids.to(dtype=torch.long).reshape(
      self.capacity, -1
    )
    if self._geom_ids.numel() != expected_geom_count:
      raise ValueError("runtime crowd must have one geom per capsule")

    shared_mask = getattr(env, "_safe_mimic_obstacle_free_envs", None)
    if shared_mask is None:
      shared_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      env._safe_mimic_obstacle_free_envs = shared_mask
    self._obstacle_free_envs: torch.Tensor = shared_mask

  def _robot_root_positions_from_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
    robot = self._env.scene[self.robot_entity_name]
    return robot.data.data.qpos[env_ids[:, None], robot.indexing.free_joint_q_adr[:3]]

  def _agent_ids_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
    slots = torch.arange(self.capacity, device=self.device)
    return (env_ids[:, None] * self.capacity + slots[None]).reshape(-1)

  def _schedule_resets(self, env_ids: torch.Tensor, now: float) -> torch.Tensor:
    count = len(env_ids)
    placement = sample_annular_crowd(
      count,
      self.capacity,
      self.device,
      min_count=self.min_count,
      max_count=self.max_count,
      min_radius_m=self.min_radius_m,
      max_radius_m=self.max_radius_m,
      target_arc_spacing_m=self.target_arc_spacing_m,
      randomize_density=self.randomize_density,
      radial_jitter_m=self.radial_jitter_m,
      min_shape_exponent=self.min_shape_exponent,
      max_shape_exponent=self.max_shape_exponent,
      angular_jitter_fraction=self.angular_jitter_fraction,
      inward_facing_probability=self.inward_facing_probability,
      inward_facing_jitter_rad=self.inward_facing_jitter_rad,
    )
    self._obstacle_free_envs[env_ids] = (
      torch.rand(count, device=self.device) < self.obstacle_free_probability
    )
    all_agent_ids = self._agent_ids_for_envs(env_ids)
    enabled = (placement.active & ~self._obstacle_free_envs[env_ids, None]).reshape(-1)
    active_ids = all_agent_ids[enabled]
    inactive_ids = all_agent_ids[~enabled]
    self.sampler.deactivate(inactive_ids)
    if active_ids.numel() == 0:
      return all_agent_ids

    robot_positions = self._robot_root_positions_from_qpos(env_ids)
    slot_positions = robot_positions[:, None].expand(-1, self.capacity, -1).clone()
    slot_positions[..., :2] += placement.offsets_xy_m
    slot_positions[..., 2] = self._env.scene.env_origins[env_ids, None, 2]
    slot_positions = slot_positions.reshape(-1, 3)[enabled]
    facing_yaw = placement.facing_yaw_rad + self.facing_yaw_offset_rad
    facing_yaw = torch.atan2(torch.sin(facing_yaw), torch.cos(facing_yaw))
    facing_yaw = facing_yaw.reshape(-1)[enabled]

    active_count = len(active_ids)
    path_ids = torch.randint(len(self.bank), (active_count,), device=self.device)
    duration_s = (self.sampler.frame_counts[path_ids] - 1).float() / self.bank.fps
    phase_times_s = torch.rand(active_count, device=self.device) * duration_s
    body_scale = sample_body_scales_for_height(
      active_count,
      self.device,
      min_height_m=self.min_human_height_m,
      max_height_m=self.max_human_height_m,
    )
    self.sampler.schedule_intersections(
      active_ids,
      path_ids=path_ids,
      global_intersection_times_s=torch.full((active_count,), now, device=self.device),
      local_intersection_times_s=phase_times_s,
      target_positions_w=slot_positions,
      target_heading=facing_yaw,
      crossing_angle_rad=0.0,
      ground_height_m=slot_positions[:, 2],
      body_scale_xyz=body_scale,
      radius_scale=torch.empty(active_count, device=self.device).uniform_(0.92, 1.08),
      radius_margin_m=torch.empty(active_count, device=self.device).uniform_(
        0.0, 0.025
      ),
      playback_speed=torch.empty(active_count, device=self.device).uniform_(
        self.min_playback_speed, self.max_playback_speed
      ),
      align_to_facing=True,
    )
    return all_agent_ids

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      resolved = self._all_env_ids()
    else:
      resolved = env_ids.to(device=self.device, dtype=torch.long)
    now = self._global_time_s()
    force_ids = self._schedule_resets(resolved, now)
    self._write_crowd_poses(
      now,
      self.capacity,
      force_ids=force_ids,
      size_ids=force_ids,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **_: object,
  ) -> None:
    del env, env_ids
    self._write_crowd_poses(self._global_time_s(), self.capacity)


__all__ = [
  "PackedHumanCapsuleCrowdMotion",
  "PackedHumanCapsuleMotion",
]
