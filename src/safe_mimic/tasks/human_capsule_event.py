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

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


def _yaw_from_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quaternion_wxyz.unbind(dim=-1)
  return torch.atan2(
    2.0 * (w * z + x * y),
    1.0 - 2.0 * (y * y + z * z),
  )


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
class HumanCapsuleMotion(ManagerTermBase):
  """Compose future-crossing human paths and drive their mocap capsules.

  The pruned skeleton bank and transition graph live on the environment device.
  Every reset samples ``walk -> action -> walk``, performs PHP-style root
  alignment and velocity-aware skeleton-space inertialization, and places the
  action midpoint on the robot reference path. Only current capsule poses are
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
    self.show_mesh = bool(params.get("show_mesh", False))
    mesh_skin_path = params.get("mesh_skin_path")
    self.mesh_skin_path = (
      Path(str(mesh_skin_path)) if mesh_skin_path is not None else None
    )
    if self.show_mesh and self.mesh_skin_path is None:
      raise ValueError("show_mesh requires mesh_skin_path")
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
    self._viser_active_env: int | None = None
    self._viser_mesh_revisions: dict[int, int] = {}
    self._pose_revision = 0
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
    if tuple(entity.body_names) != expected_bodies:
      raise ValueError("runtime human entity body order does not match capsules")
    body_ids = entity.indexing.body_ids.detach().cpu().numpy().astype(np.int64)
    mocap_ids = np.asarray(env.sim.mj_model.body_mocapid)[body_ids]
    if np.any(mocap_ids < 0) or len(np.unique(mocap_ids)) != len(expected_names):
      raise ValueError("every human capsule body must have a unique mocap id")
    self._mocap_ids = torch.as_tensor(mocap_ids, dtype=torch.long, device=env.device)
    self._geom_ids = entity.indexing.geom_ids.to(dtype=torch.long)
    if self._geom_ids.numel() != len(expected_names):
      raise ValueError("runtime human entity must have one geom per capsule")

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

  def _schedule(self, env_ids: torch.Tensor, global_time_s: float) -> None:
    if env_ids.numel() == 0:
      return
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    count = len(env_ids)
    command: Any = self._env.command_manager.get_term(self.command_name)
    motion = command.motion
    current_frames = command.time_steps[env_ids]
    last_frame = int(motion.time_step_total) - 1

    requested_delay = torch.empty(count, device=self.device).uniform_(
      self.min_intersection_delay_s,
      self.max_intersection_delay_s,
    )
    requested_steps = torch.round(requested_delay / self._env.step_dt).long()
    delay_steps = requested_steps.clamp_min(1)
    target_frames = torch.clamp(current_frames + delay_steps, max=last_frame)
    actual_delay_s = delay_steps.float() * self._env.step_dt

    anchor_index = int(command.motion_anchor_body_index)
    target_positions = motion.body_pos_w[target_frames, anchor_index].clone()
    target_positions += self._env.scene.env_origins[env_ids]
    target_quaternions = motion.body_quat_w[target_frames, anchor_index]
    target_yaw = _yaw_from_quaternion(target_quaternions)
    before_frames = torch.clamp(target_frames - 2, min=0)
    after_frames = torch.clamp(target_frames + 2, max=last_frame)
    before = motion.body_pos_w[before_frames, anchor_index, :2]
    after = motion.body_pos_w[after_frames, anchor_index, :2]
    displacement = after - before
    target_heading = torch.atan2(displacement[:, 1], displacement[:, 0])
    target_heading = torch.where(
      torch.linalg.vector_norm(displacement, dim=-1) < 0.03,
      target_yaw,
      target_heading,
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
    if self.min_initial_spawn_radius_m is None:
      spawn_radius = torch.zeros(count, device=self.device)
    else:
      assert self.max_initial_spawn_radius_m is not None
      spawn_radius = torch.empty(count, device=self.device).uniform_(
        self.min_initial_spawn_radius_m, self.max_initial_spawn_radius_m
      )
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

  def _write_updated_poses(self, global_time_s: float) -> None:
    poses = self.sampler.sample_held(global_time_s)
    env_ids = self.sampler.last_updated_env_ids
    if len(env_ids) == 0:
      return
    env_grid, mocap_grid = torch.meshgrid(env_ids, self._mocap_ids, indexing="ij")
    self._env.sim.data.mocap_pos[env_grid, mocap_grid] = poses.centers_w[env_ids]
    self._env.sim.data.mocap_quat[env_grid, mocap_grid] = poses.quaternions_wxyz[
      env_ids
    ]

    env_grid, geom_grid = torch.meshgrid(env_ids, self._geom_ids, indexing="ij")
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

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      resolved = self._all_env_ids()
    else:
      resolved = env_ids.to(device=self.device, dtype=torch.long)
    now = self._global_time_s()
    self._schedule(resolved, now)
    self._write_updated_poses(now)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **_: object,
  ) -> None:
    del env, env_ids
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
    self._viser_active_env = env_index

    skin = self._load_viser_skin()
    handle = self._viser_mesh_handles.get(env_index)
    if handle is None or self._viser_mesh_revisions.get(env_index) != (
      self._pose_revision
    ):
      positions = joint_positions[env_index].detach().cpu().numpy()
      quaternions = joint_quaternions[env_index].detach().cpu().numpy()
      body_scale = self.sampler.body_scale_xyz[env_index].detach().cpu().numpy()
      yaw = float(self.sampler.placement_yaw[env_index])
      translation = self.sampler.current_translation_w[env_index].detach().cpu().numpy()
      scene_offset = np.asarray(
        getattr(visualizer, "_scene_offset", np.zeros(3)), dtype=np.float32
      )
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
    handle.visible = bool(self.sampler._poses.active[env_index])
