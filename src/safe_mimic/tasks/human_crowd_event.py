"""Persistent 10 Hz updater for an animated annular capsule crowd."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase

from safe_mimic.assets.soma_capsules import (
  HUMAN_CROWD_CAPACITY,
  HUMAN_CROWD_RAY_BODY_NAME,
  HUMAN_INACTIVE_HEIGHT_M,
  crowd_capsule_body_name,
)
from safe_mimic.motions.annular_crowd import sample_annular_crowd
from safe_mimic.motions.composed_skeleton_bank import (
  OnlineComposedHumanSampler,
  SkeletonPathBank,
)
from safe_mimic.motions.human_capsules import (
  DEFAULT_MAX_HUMAN_HEIGHT_M,
  DEFAULT_MIN_HUMAN_HEIGHT_M,
  SOMA_CROWD_PROXY_SPECS,
  sample_body_scales_for_height,
)
from safe_mimic.motions.soma_mesh import (
  SomaViserSkin,
  cluster_soma_viser_skin,
  load_soma_mesh_skin,
  prepare_soma_viser_skin,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


@requires_model_fields("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb")
class HumanCapsuleCrowdMotion(ManagerTermBase):
  """Drive a variable-density crowd from a stationary arm-action graph.

  Each enabled person owns a fixed slot on an annular boundary sampled at reset.
  reset. The pelvis XY is locked to that slot while the complete skeleton keeps
  its arm, torso, and leg articulation. Consecutive actions use the same
  velocity-aware skeleton inertialization as the crossing-human composer.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    params = cfg.params
    self.human_entity_name = str(params["human_entity"])
    self.robot_entity_name = str(params.get("robot_entity", "robot"))
    self.capacity = int(params.get("capacity", HUMAN_CROWD_CAPACITY))
    self.min_count = int(params.get("min_count", 4))
    self.max_count = int(params.get("max_count", self.capacity))
    self.min_radius_m = float(params.get("min_radius_m", 3.0))
    self.max_radius_m = float(params.get("max_radius_m", 6.0))
    target_arc_spacing = params.get("target_arc_spacing_m")
    self.target_arc_spacing_m = (
      None if target_arc_spacing is None else float(target_arc_spacing)
    )
    self.randomize_density = bool(params.get("randomize_density", False))
    self.radial_jitter_m = float(params.get("radial_jitter_m", 0.0))
    self.min_shape_exponent = float(params.get("min_shape_exponent", 2.0))
    self.max_shape_exponent = float(params.get("max_shape_exponent", 2.0))
    self.angular_jitter_fraction = float(params.get("angular_jitter_fraction", 0.25))
    self.inward_facing_probability = float(params.get("inward_facing_probability", 0.7))
    self.inward_facing_jitter_rad = float(
      params.get("inward_facing_jitter_rad", torch.pi / 4.0)
    )
    self.facing_yaw_offset_rad = float(params.get("facing_yaw_offset_rad", 0.0))
    self.min_playback_speed = float(params.get("min_playback_speed", 0.8))
    self.max_playback_speed = float(params.get("max_playback_speed", 1.2))
    self.min_human_height_m = float(
      params.get("min_human_height_m", DEFAULT_MIN_HUMAN_HEIGHT_M)
    )
    self.max_human_height_m = float(
      params.get("max_human_height_m", DEFAULT_MAX_HUMAN_HEIGHT_M)
    )
    self.show_mesh = bool(params.get("show_mesh", False))
    self.motion_update_hz = float(params.get("update_hz", 10.0))
    self.mesh_update_hz = float(params.get("mesh_update_hz", self.motion_update_hz))
    self.mesh_voxel_size_m = float(params.get("mesh_voxel_size_m", 0.0))
    if self.capacity != HUMAN_CROWD_CAPACITY:
      raise ValueError("event capacity must match the compiled crowd asset")
    if not 0 <= self.min_count <= self.max_count <= self.capacity:
      raise ValueError("invalid crowd density range")
    if not 0.0 < self.min_playback_speed <= self.max_playback_speed:
      raise ValueError("invalid crowd playback-speed range")
    if not 0.0 < self.min_human_height_m <= self.max_human_height_m:
      raise ValueError("invalid crowd human-height range")
    if not 0.0 < self.mesh_update_hz <= self.motion_update_hz:
      raise ValueError("mesh update rate must lie in (0, motion update rate]")
    if self.mesh_voxel_size_m < 0.0:
      raise ValueError("mesh voxel size must be non-negative")

    mesh_skin_path = params.get("mesh_skin_path")
    self.mesh_skin_path = (
      Path(str(mesh_skin_path)) if mesh_skin_path is not None else None
    )
    if self.show_mesh and self.mesh_skin_path is None:
      raise ValueError("show_mesh requires mesh_skin_path")

    self.bank = SkeletonPathBank(Path(params["skeleton_bank_path"]))
    self.agent_count = env.num_envs * self.capacity
    self.sampler = OnlineComposedHumanSampler(
      self.bank,
      self.agent_count,
      env.device,
      transition_duration_s=float(params.get("transition_duration_s", 0.2)),
      update_hz=self.motion_update_hz,
      inactive_height_m=HUMAN_INACTIVE_HEIGHT_M,
      retain_joint_poses=self.show_mesh,
      lock_root_xy=True,
      capsule_specs=SOMA_CROWD_PROXY_SPECS,
    )

    transition_index = Path(params["transition_index_path"])
    with np.load(transition_index / "edges.npz", allow_pickle=False) as graph:
      self._action_path_ids = torch.as_tensor(
        np.asarray(graph["action_path_ids"]).copy(),
        dtype=torch.long,
        device=env.device,
      )
      self._next_path_ids = torch.as_tensor(
        np.asarray(graph["stationary_next_ids"]).copy(),
        dtype=torch.long,
        device=env.device,
      )
    if self._next_path_ids.ndim != 2 or self._next_path_ids.shape[0] != len(
      self._action_path_ids
    ):
      raise ValueError("stationary action graph has incompatible shapes")
    lookup_size = int(self.bank.config["source_path_count"])
    self._action_to_row = torch.full(
      (lookup_size,), -1, dtype=torch.long, device=env.device
    )
    self._action_to_row[self._action_path_ids] = torch.arange(
      len(self._action_path_ids), device=env.device
    )
    if torch.any(self._action_to_row[self._next_path_ids] < 0):
      raise ValueError("stationary graph points outside its action set")

    entity = env.scene[self.human_entity_name]
    capsule_names = tuple(spec.name for spec in SOMA_CROWD_PROXY_SPECS)
    expected_bodies = tuple(
      crowd_capsule_body_name(crowd_index, capsule_name)
      for crowd_index in range(self.capacity)
      for capsule_name in capsule_names
    )
    expected_geom_count = self.capacity * len(capsule_names)
    self._direct_geom_poses = tuple(entity.body_names) == (HUMAN_CROWD_RAY_BODY_NAME,)
    self._mocap_ids: torch.Tensor | None = None
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

    self._enabled = torch.zeros(self.agent_count, dtype=torch.bool, device=self.device)
    self._slot_positions_w = torch.zeros((self.agent_count, 3), device=self.device)
    self._slot_facing_yaw = torch.zeros(self.agent_count, device=self.device)
    self._last_path_ids = torch.zeros(
      self.agent_count, dtype=torch.long, device=self.device
    )
    graph_tensors = (
      self._action_path_ids,
      self._next_path_ids,
      self._action_to_row,
    )
    self.device_storage_bytes = self.sampler.device_storage_bytes + sum(
      tensor.numel() * tensor.element_size() for tensor in graph_tensors
    )

    self._viser_skin: SomaViserSkin | None = None
    self._viser_mesh_handles: dict[tuple[int, int], Any] = {}
    self._viser_active_env: int | None = None
    self._viser_mesh_revisions: dict[tuple[int, int], int] = {}
    self._pose_revision = 0
    self._mesh_update_stride = max(
      1, round(self.motion_update_hz / self.mesh_update_hz)
    )

  def _global_time_s(self) -> float:
    return float(self._env.common_step_counter) * self._env.step_dt

  def _all_env_ids(self) -> torch.Tensor:
    return torch.arange(self.num_envs, dtype=torch.long, device=self.device)

  def _robot_root_positions_from_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Read the newly written reset root without requiring sim.forward()."""

    robot = self._env.scene[self.robot_entity_name]
    qpos_env_ids = env_ids[:, None]
    return robot.data.data.qpos[qpos_env_ids, robot.indexing.free_joint_q_adr[:3]]

  def _agent_ids_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
    slots = torch.arange(self.capacity, device=self.device)
    return (env_ids[:, None] * self.capacity + slots[None]).reshape(-1)

  def _sample_sequences(
    self,
    count: int,
    *,
    first_path_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if first_path_ids is None:
      first_rows = torch.randint(
        len(self._action_path_ids), (count,), device=self.device
      )
      first = self._action_path_ids[first_rows]
    else:
      first = first_path_ids.to(device=self.device, dtype=torch.long)
      first_rows = self._action_to_row[first]
      if torch.any(first_rows < 0):
        raise ValueError("continuation starts outside stationary action graph")
    rank = torch.randint(self._next_path_ids.shape[1], (count,), device=self.device)
    second = self._next_path_ids[first_rows, rank]
    second_rows = self._action_to_row[second]
    rank = torch.randint(self._next_path_ids.shape[1], (count,), device=self.device)
    third = self._next_path_ids[second_rows, rank]
    return torch.stack((first, second, third), dim=-1)

  def _deactivate(self, agent_ids: torch.Tensor) -> None:
    if len(agent_ids) == 0:
      return
    poses = self.sampler._poses
    poses.centers_w[agent_ids] = 0.0
    poses.centers_w[agent_ids, :, 2] = HUMAN_INACTIVE_HEIGHT_M
    # Keep valid primitive dimensions while the member is parked below the
    # world. Zero-length capsules make Viser's trimesh conversion emit NaN
    # bounds, which corrupts the websocket JSON and leaves the page white.
    poses.radii_m[agent_ids] = self.sampler.base_radii
    poses.half_lengths_m[agent_ids] = torch.where(
      self.sampler.is_sphere,
      torch.zeros_like(self.sampler.base_radii),
      torch.full_like(self.sampler.base_radii, 0.05),
    )
    poses.active[agent_ids] = False
    self.sampler.dirty[agent_ids] = False
    self.sampler.next_update_times_s[agent_ids] = torch.inf

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
    all_agent_ids = self._agent_ids_for_envs(env_ids)
    enabled = placement.active.reshape(-1)
    self._enabled[all_agent_ids] = enabled

    robot_positions = self._robot_root_positions_from_qpos(env_ids)
    slot_positions = robot_positions[:, None].expand(-1, self.capacity, -1).clone()
    slot_positions[..., :2] += placement.offsets_xy_m
    slot_positions[..., 2] = self._env.scene.env_origins[env_ids, None, 2]
    self._slot_positions_w[all_agent_ids] = slot_positions.reshape(-1, 3)
    facing_yaw = placement.facing_yaw_rad + self.facing_yaw_offset_rad
    facing_yaw = torch.atan2(torch.sin(facing_yaw), torch.cos(facing_yaw))
    self._slot_facing_yaw[all_agent_ids] = facing_yaw.reshape(-1)

    active_ids = all_agent_ids[enabled]
    inactive_ids = all_agent_ids[~enabled]
    self._deactivate(inactive_ids)
    if len(active_ids):
      active_count = len(active_ids)
      sequence = self._sample_sequences(active_count)
      body_scale = sample_body_scales_for_height(
        active_count,
        self.device,
        min_height_m=self.min_human_height_m,
        max_height_m=self.max_human_height_m,
      )
      playback_speed = torch.empty(active_count, device=self.device).uniform_(
        self.min_playback_speed, self.max_playback_speed
      )
      self.sampler.schedule_intersections(
        active_ids,
        sequence_source_ids=sequence,
        global_intersection_times_s=torch.full(
          (active_count,), now, device=self.device
        ),
        robot_positions_at_intersection_w=self._slot_positions_w[active_ids],
        robot_yaw_at_intersection=torch.zeros(active_count, device=self.device),
        robot_path_heading_at_intersection=self._slot_facing_yaw[active_ids],
        action_phase=torch.rand(active_count, device=self.device),
        crossing_angle_rad=0.0,
        ground_height_m=self._slot_positions_w[active_ids, 2],
        body_scale_xyz=body_scale,
        radius_scale=torch.empty(active_count, device=self.device).uniform_(0.92, 1.08),
        radius_margin_m=torch.empty(active_count, device=self.device).uniform_(
          0.0, 0.025
        ),
        playback_speed=playback_speed,
        align_to_facing=True,
      )
      self._last_path_ids[active_ids] = sequence[:, 2]
    return all_agent_ids

  def _schedule_continuations(self, agent_ids: torch.Tensor, now: float) -> None:
    if len(agent_ids) == 0:
      return
    sequence = self._sample_sequences(
      len(agent_ids), first_path_ids=self._last_path_ids[agent_ids]
    )
    # The old triplet's third segment becomes the new triplet's first. Its
    # accumulated source alignment is folded into the placement yaw so the
    # boundary pose remains continuous in world space.
    placement_yaw = (
      self.sampler.placement_yaw[agent_ids] + (self.sampler.alignment_yaw[agent_ids, 2])
    )
    alignment_translation = (
      self.sampler.alignment_translation[agent_ids, 2]
      * self.sampler.body_scale_xyz[agent_ids]
    )
    old_yaw = self.sampler.placement_yaw[agent_ids]
    cosine = torch.cos(old_yaw)
    sine = torch.sin(old_yaw)
    rotated_alignment = alignment_translation.clone()
    rotated_alignment[:, 0] = (
      cosine * alignment_translation[:, 0] - sine * alignment_translation[:, 1]
    )
    rotated_alignment[:, 1] = (
      sine * alignment_translation[:, 0] + cosine * alignment_translation[:, 1]
    )
    translation = self.sampler.translation_w[agent_ids] + rotated_alignment
    self.sampler.schedule_intersections(
      agent_ids,
      sequence_source_ids=sequence,
      global_intersection_times_s=torch.full(
        (len(agent_ids),), now, device=self.device
      ),
      robot_positions_at_intersection_w=self._slot_positions_w[agent_ids],
      robot_yaw_at_intersection=torch.zeros(len(agent_ids), device=self.device),
      robot_path_heading_at_intersection=torch.zeros(
        len(agent_ids), device=self.device
      ),
      action_phase=0.0,
      crossing_angle_rad=0.0,
      ground_height_m=self._slot_positions_w[agent_ids, 2],
      body_scale_xyz=self.sampler.body_scale_xyz[agent_ids],
      radius_scale=self.sampler.radius_scale[agent_ids],
      radius_margin_m=self.sampler.radius_margin_m[agent_ids],
      playback_speed=self.sampler.playback_speed[agent_ids],
      placement_yaw_override=placement_yaw,
      translation_override_w=translation,
    )
    self._last_path_ids[agent_ids] = sequence[:, 2]

  def _write_poses(self, now: float, *, force_ids: torch.Tensor | None = None) -> None:
    poses = self.sampler.sample_held(now)
    agent_ids = self.sampler.last_updated_env_ids
    if force_ids is not None:
      agent_ids = torch.unique(torch.cat((agent_ids, force_ids)))
    if len(agent_ids) == 0:
      return
    env_ids = torch.div(agent_ids, self.capacity, rounding_mode="floor")
    crowd_ids = torch.remainder(agent_ids, self.capacity)
    geom_ids = self._geom_ids[crowd_ids]
    geom_env_grid = env_ids[:, None].expand_as(geom_ids)
    if self._direct_geom_poses:
      model = self._env.sim.model
      model.geom_pos[geom_env_grid, geom_ids] = poses.centers_w[agent_ids]
      model.geom_quat[geom_env_grid, geom_ids] = poses.quaternions_wxyz[agent_ids]
    else:
      assert self._mocap_ids is not None
      mocap_ids = self._mocap_ids[crowd_ids]
      env_grid = env_ids[:, None].expand_as(mocap_ids)
      self._env.sim.data.mocap_pos[env_grid, mocap_ids] = poses.centers_w[agent_ids]
      self._env.sim.data.mocap_quat[env_grid, mocap_ids] = poses.quaternions_wxyz[
        agent_ids
      ]

    radii = poses.radii_m[agent_ids]
    half_lengths = poses.half_lengths_m[agent_ids]
    model = self._env.sim.model
    model.geom_size[geom_env_grid, geom_ids] = torch.stack(
      (radii, half_lengths, torch.zeros_like(radii)), dim=-1
    )
    model.geom_rbound[geom_env_grid, geom_ids] = radii + half_lengths
    model.geom_aabb[geom_env_grid, geom_ids, 0] = 0.0
    model.geom_aabb[geom_env_grid, geom_ids, 1] = torch.stack(
      (radii, radii, radii + half_lengths), dim=-1
    )
    self._pose_revision += 1

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      resolved = self._all_env_ids()
    else:
      resolved = env_ids.to(device=self.device, dtype=torch.long)
    now = self._global_time_s()
    force_ids = self._schedule_resets(resolved, now)
    self._write_poses(now, force_ids=force_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **_: object,
  ) -> None:
    del env, env_ids
    now = self._global_time_s()
    expired = self.sampler.expired_env_ids(now)
    expired = expired[self._enabled[expired]]
    self._schedule_continuations(expired, now)
    self._write_poses(now)

  def _load_viser_skin(self) -> SomaViserSkin:
    if self._viser_skin is None:
      assert self.mesh_skin_path is not None
      self._viser_skin = prepare_soma_viser_skin(
        load_soma_mesh_skin(self.mesh_skin_path), self.bank.joint_names
      )
      if self.mesh_voxel_size_m > 0.0:
        self._viser_skin = cluster_soma_viser_skin(
          self._viser_skin, self.mesh_voxel_size_m
        )
    return self._viser_skin

  def debug_vis(self, visualizer: Any) -> None:
    """Render posed meshes for enabled members in the selected environment."""

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
      for crowd_index in range(self.capacity):
        previous = self._viser_mesh_handles.get((self._viser_active_env, crowd_index))
        if previous is not None:
          previous.visible = False
    self._viser_active_env = env_index
    skin = self._load_viser_skin()
    scene_offset = np.asarray(
      getattr(visualizer, "_scene_offset", np.zeros(3)), dtype=np.float32
    )
    for crowd_index in range(self.capacity):
      key = (env_index, crowd_index)
      agent_index = env_index * self.capacity + crowd_index
      handle = self._viser_mesh_handles.get(key)
      active = bool(self.sampler._poses.active[agent_index]) and bool(
        self._enabled[agent_index]
      )
      if not active:
        if handle is not None:
          handle.visible = False
        continue
      revision_due = self._pose_revision % self._mesh_update_stride == 0
      if handle is None or (
        revision_due and self._viser_mesh_revisions.get(key) != self._pose_revision
      ):
        positions = joint_positions[agent_index].detach().cpu().numpy()
        quaternions = joint_quaternions[agent_index].detach().cpu().numpy()
        body_scale = self.sampler.body_scale_xyz[agent_index].detach().cpu().numpy()
        yaw = float(self.sampler.placement_yaw[agent_index])
        translation = (
          self.sampler.current_translation_w[agent_index].detach().cpu().numpy()
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
            f"/safe_mimic/human_crowd/env_{env_index}/person_{crowd_index}",
            vertices_w,
            skin.triangles,
            color=(205, 158, 121),
            opacity=1.0,
            material="toon3",
            side="double",
            cast_shadow=False,
            receive_shadow=False,
          )
          self._viser_mesh_handles[key] = handle
        else:
          handle.vertices = vertices_w
        self._viser_mesh_revisions[key] = self._pose_revision
      handle.visible = True
