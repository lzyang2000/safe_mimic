"""Exact kinematic motion replay for reference-filtering demonstrations."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from safe_mimic.motions.packed_npz_motion_lib import (
  PackedNpzMotionLib,
  load_packed_npz_manifest,
)
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


def _planar_reference_alignment(
  reference_anchor_quat_w: torch.Tensor,
  robot_root_pos_w: torch.Tensor,
  robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return yaw rotation and root XY that align reference to the robot."""
  yaw_delta_w = yaw_quat(
    quat_mul(robot_anchor_quat_w, quat_inv(reference_anchor_quat_w))
  )
  return yaw_delta_w, robot_root_pos_w[..., :2]


def hinge_chain_body_positions(
  root_pos_w: torch.Tensor,
  root_quat_w: torch.Tensor,
  *,
  parent_chain_index: tuple[int, ...],
  parent_root_slot: tuple[int, ...],
  body_pos_l: torch.Tensor,
  body_quat_l: torch.Tensor,
  joint_pos_l: torch.Tensor,
  joint_axis_l: torch.Tensor,
  joint_angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compose world poses of a hinge chain from batched root frames.

  Bodies are visited in the given order, which must be topological: entry
  ``b`` attaches to chain entry ``parent_chain_index[b]`` when that value is
  non-negative, otherwise to root slot ``parent_root_slot[b]`` of
  ``root_pos_w``/``root_quat_w`` (shape ``(N, num_roots, 3/4)``). Each body
  carries exactly one hinge whose local anchor/axis follow MuJoCo's
  convention (``jnt_pos``/``jnt_axis`` expressed in the owning body's frame):
  the world frame of body ``b`` at angle ``q`` keeps the anchor fixed while
  rotating the local frame, matching ``mj_kinematics``. Angles are relative
  to the model's zero configuration (``qpos0 == 0`` is asserted upstream).
  Returns world body positions ``(N, B, 3)`` and wxyz quaternions
  ``(N, B, 4)``.
  """
  body_count = len(parent_chain_index)
  positions: list[torch.Tensor] = []
  quaternions: list[torch.Tensor] = []
  batch = root_pos_w.shape[0]
  for body_index in range(body_count):
    chain_parent = parent_chain_index[body_index]
    if chain_parent >= body_index:
      raise ValueError("hinge chain bodies must be topologically ordered")
    if chain_parent >= 0:
      parent_pos = positions[chain_parent]
      parent_quat = quaternions[chain_parent]
    else:
      parent_pos = root_pos_w[:, parent_root_slot[body_index]]
      parent_quat = root_quat_w[:, parent_root_slot[body_index]]
    offset_l = body_pos_l[body_index].expand(batch, 3)
    frame_quat_l = body_quat_l[body_index].expand(batch, 4)
    anchor_l = joint_pos_l[body_index].expand(batch, 3)
    axis_l = joint_axis_l[body_index].expand(batch, 3)
    pre_pos = parent_pos + quat_apply(parent_quat, offset_l)
    pre_quat = quat_mul(parent_quat, frame_quat_l)
    hinge_quat = quat_from_angle_axis(joint_angles[:, body_index], axis_l)
    post_quat = quat_mul(pre_quat, hinge_quat)
    anchor_w = pre_pos + quat_apply(pre_quat, anchor_l)
    post_pos = anchor_w - quat_apply(post_quat, anchor_l)
    positions.append(post_pos)
    quaternions.append(post_quat)
  return torch.stack(positions, dim=1), torch.stack(quaternions, dim=1)


def _apply_planar_reference_alignment(
  positions_w: torch.Tensor,
  reference_root_pos_w: torch.Tensor,
  aligned_root_xy_w: torch.Tensor,
  yaw_delta_w: torch.Tensor,
) -> torch.Tensor:
  """Yaw-rotate a body cloud about its root and translate its root in XY."""
  relative_positions = positions_w - reference_root_pos_w[:, None]
  yaw = yaw_delta_w[:, None].expand(-1, positions_w.shape[1], -1)
  aligned_relative_positions = quat_apply(yaw, relative_positions)
  aligned_positions = positions_w.clone()
  aligned_positions[..., :2] = (
    aligned_root_xy_w[:, None] + aligned_relative_positions[..., :2]
  )
  return aligned_positions


def _validate_propagation_flags(*, arm: bool, whole_body: bool) -> tuple[bool, bool]:
  """Return ``(propagate_targets, anchor_target_corrected)`` for the cfg flags."""
  if arm and whole_body:
    raise ValueError(
      "choose arm-only or whole-body propagation, not both "
      "(propagate_arm_corrections_to_body_targets vs "
      "propagate_joint_corrections_to_body_targets)"
    )
  return (arm or whole_body, whole_body)


MOTION_MANIFEST_SUFFIXES = (".yaml", ".yml", ".jsonl")


def is_motion_manifest(path: str | Path) -> bool:
  """True when ``path`` names a clip-library manifest rather than one NPZ."""
  return Path(path).suffix.lower() in MOTION_MANIFEST_SUFFIXES


class LibraryMotionLoader:
  """``MotionLoader``-compatible view over a packed clip library.

  Frames of every clip sit back to back in flat tensors, so the replay
  command keeps indexing ``body_pos_w[time_steps]`` with a GLOBAL frame index;
  ``time_step_total`` is the total frame count. The clip tables tell the
  command where each env's current clip starts and ends so it can sample a
  start frame inside a clip and chain to a fresh clip when one runs out.
  """

  def __init__(self, library: PackedNpzMotionLib) -> None:
    self._library = library
    self.joint_pos = library.all_joint_pos
    self.joint_vel = library.all_joint_vel
    self.body_pos_w = library.all_body_pos_w
    self.body_quat_w = library.all_body_quat_w
    self.body_lin_vel_w = library.all_body_lin_vel_w
    self.body_ang_vel_w = library.all_body_ang_vel_w
    self.time_step_total = int(library.total_frames)
    self.clip_start_idx = library.motion_start_idx.to(dtype=torch.long)
    self.clip_num_frames = library.motion_num_frames.to(dtype=torch.long)
    self.clip_weights = library.motion_weights

  def num_clips(self) -> int:
    return int(self._library.num_motions())

  def sample_clips(self, count: int) -> torch.Tensor:
    """Sample clip ids with replacement according to manifest weights."""
    return self._library.sample_motions(count)


class KinematicReplayMotionCommand(MotionCommand):
  """Overwrite the robot with the current reference frame before sensing.

  The environment retains its normal action interface, but policy actions and
  intermediate physics state cannot change the pose observed at the end of a
  control step. This makes the command useful for inspecting reference motion,
  moving humans, and body-mounted sensors without tracking-policy error.
  """

  cfg: KinematicReplayMotionCommandCfg

  def __init__(self, cfg: KinematicReplayMotionCommandCfg, env) -> None:
    library: PackedNpzMotionLib | None = None
    loader_cfg = cfg
    if is_motion_manifest(cfg.motion_file):
      # ``MotionCommand.__init__`` hard-codes the single-NPZ ``MotionLoader``.
      # Hand it the library's first clip so its storage-independent setup
      # (metrics, buffers) runs unchanged, then swap in the flat library view.
      sources = load_packed_npz_manifest(cfg.motion_file, splits=cfg.manifest_splits)
      loader_cfg = copy.copy(cfg)
      loader_cfg.motion_file = str(sources[0].path)
      robot = env.scene[cfg.entity_name]
      body_indexes = robot.find_bodies(cfg.body_names, preserve_order=True)[0]
      library = PackedNpzMotionLib(
        cfg.motion_file, body_indexes, device=env.device, splits=cfg.manifest_splits
      )
    super().__init__(loader_cfg, env)
    self.cfg = cfg
    if library is not None:
      self.motion = LibraryMotionLoader(library)
    self._all_env_ids = torch.arange(
      self.num_envs, dtype=torch.long, device=self.device
    )
    # Per-env bounds of the clip currently being replayed (global frame
    # indices). A single NPZ is one clip spanning every frame.
    self._clip_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._clip_end = torch.full(
      (self.num_envs,),
      self.motion.time_step_total,
      dtype=torch.long,
      device=self.device,
    )

  def _has_clip_library(self) -> bool:
    return isinstance(self.motion, LibraryMotionLoader)

  def _assign_clips(self, env_ids: torch.Tensor, *, at_start: bool) -> None:
    """Draw a clip per env, record its bounds, and place ``time_steps`` in it."""
    assert isinstance(self.motion, LibraryMotionLoader)
    clips = self.motion.sample_clips(len(env_ids)).to(device=env_ids.device)
    start = self.motion.clip_start_idx.to(env_ids.device)[clips]
    frames = self.motion.clip_num_frames.to(env_ids.device)[clips]
    self._clip_start[env_ids] = start
    self._clip_end[env_ids] = start + frames
    if at_start:
      offset = torch.zeros_like(start)
    else:
      offset = torch.floor(
        torch.rand(len(env_ids), device=env_ids.device) * frames
      ).long()
    self.time_steps[env_ids] = start + offset

  def _advance_time_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Step ``time_steps`` by one frame; return the mask of envs whose clip ended.

    With a clip library an ended clip CHAINS to a freshly sampled clip's first
    frame (the live alignment re-glues the reference to the robot's current
    pose, so the robot simply continues from where it is). With a single NPZ
    the frame index wraps to zero, exactly the historical modulo behaviour.
    """
    next_steps = self.time_steps[env_ids] + 1
    clip_end = getattr(self, "_clip_end", None)
    if clip_end is None:
      end = torch.full_like(next_steps, self.motion.time_step_total)
    else:
      end = clip_end[env_ids]
    wrapped = next_steps >= end
    self.time_steps[env_ids] = next_steps
    if bool(wrapped.any()):
      wrapped_ids = env_ids[wrapped]
      if self._has_clip_library():
        self._assign_clips(wrapped_ids, at_start=True)
      else:
        clip_start = getattr(self, "_clip_start", None)
        self.time_steps[wrapped_ids] = (
          0 if clip_start is None else clip_start[wrapped_ids]
        )
    return wrapped

  def _sample_start_time_steps(self, env_ids: torch.Tensor) -> None:
    """Sample per-env start frames according to ``cfg.sampling_mode``.

    ``start`` reproduces the historical exact frame-zero replay bit for bit.
    ``uniform`` reuses the upstream uniform sampler (a random frame in
    ``[0, time_step_total)`` per env). ``adaptive`` is refused explicitly: its
    failure-bin bookkeeping lives in upstream reset/update paths that these
    replay commands override, so it is unverified here and must not fall back
    silently.
    """
    mode = self.cfg.sampling_mode
    if mode not in ("start", "uniform"):
      raise NotImplementedError(
        f"sampling_mode {mode!r} is not supported by kinematic replay commands: "
        "adaptive bin bookkeeping is unverified through their overridden "
        "reset/update paths; use 'start' or 'uniform'"
      )
    if self._has_clip_library():
      # Library: pick a clip (manifest weights), then its first frame or a
      # uniform frame inside it. Upstream's uniform sampler assumes one clip.
      self._assign_clips(env_ids, at_start=mode == "start")
      return
    if mode == "start":
      self.time_steps[env_ids] = 0
    elif mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      raise NotImplementedError(
        f"sampling_mode {mode!r} is not supported by kinematic replay commands: "
        "adaptive bin bookkeeping is unverified through their overridden "
        "reset/update paths; use 'start' or 'uniform'"
      )

  def _write_current_frame_to_sim(self, env_ids: torch.Tensor) -> None:
    """Write the exact reference state at each env's current frame.

    This is ``reset_to_frame`` without its scalar frame assignment, so envs
    can start at different (already sampled) frames.
    """
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Start exact replay at the sampled frame without reset-state randomization."""
    self._sample_start_time_steps(env_ids)
    self._write_current_frame_to_sim(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    """Advance, write, and forward the exact reference for selected envs."""
    replay_env_ids = self._all_env_ids if env_ids is None else env_ids
    if replay_env_ids.numel() == 0:
      return

    self._advance_time_steps(replay_env_ids)
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
  """Configuration that builds :class:`KinematicReplayMotionCommand`.

  ``motion_file`` may be one tracker NPZ (historical) or a clip-library
  manifest (``.yaml`` / ``.jsonl``); ``manifest_splits`` selects manifest
  splits, ``None`` meaning every clip.
  """

  manifest_splits: tuple[str, ...] | None = None

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
    self._obstacle_geom_counts = tuple(
      entity.data.geom_pos_w.shape[1] for entity in self._obstacle_entities
    )
    obstacle_count = sum(self._obstacle_geom_counts)
    if obstacle_count < 1:
      raise ValueError("reference filter requires at least one obstacle geom")
    entity_slices: dict[str, slice] = {}
    slice_offset = 0
    for entity_name, entity_geom_count in zip(
      cfg.obstacle_entity_names, self._obstacle_geom_counts, strict=True
    ):
      entity_slices[entity_name] = slice(slice_offset, slice_offset + entity_geom_count)
      slice_offset += entity_geom_count
    self._obstacle_entity_slices = entity_slices
    if cfg.link_filter_capsules_per_group or cfg.link_filter_nearest_groups:
      if not (
        len(cfg.link_filter_capsules_per_group)
        == len(cfg.link_filter_nearest_groups)
        == len(self._obstacle_entities)
      ):
        raise ValueError(
          "link-filter grouping must match obstacle_entity_names"
        )
      self._link_filter_capsules_per_group = cfg.link_filter_capsules_per_group
      self._link_filter_nearest_groups = cfg.link_filter_nearest_groups
    else:
      self._link_filter_capsules_per_group = (None,) * len(
        self._obstacle_entities
      )
      self._link_filter_nearest_groups = (None,) * len(self._obstacle_entities)
    for geom_count, capsules_per_group, nearest_groups in zip(
      self._obstacle_geom_counts,
      self._link_filter_capsules_per_group,
      self._link_filter_nearest_groups,
      strict=True,
    ):
      if (capsules_per_group is None) != (nearest_groups is None):
        raise ValueError(
          "capsules_per_group and nearest_groups must both be set or both be None"
        )
      if capsules_per_group is None:
        continue
      if capsules_per_group < 1 or nearest_groups < 1:
        raise ValueError("link-filter grouping values must be positive")
      if geom_count % capsules_per_group != 0:
        raise ValueError("obstacle geom count is not divisible by its group size")

    self._filtered_root_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
    self._filtered_root_velocity_xy_w = torch.zeros_like(self._filtered_root_xy_w)
    self._root_translation_residual_xy_w = torch.zeros_like(self._filtered_root_xy_w)
    self._filter_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._reference_alignment_yaw_w = torch.zeros(self.num_envs, 4, device=self.device)
    self._reference_alignment_yaw_w[:, 0] = 1.0
    self._reference_alignment_root_xy_w = torch.zeros(
      self.num_envs, 2, device=self.device
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
    self._obstacle_velocity_decay = torch.exp(
      torch.tensor(
        -cfg.planar_filter.obstacle_velocity_decay * env.step_dt,
        device=self.device,
      )
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
    posture_joint_flags = [
      any(token in joint_name for token in cfg.link_filter.posture_joint_name_tokens)
      for joint_name in self.robot.joint_names
    ]
    if not any(posture_joint_flags):
      raise ValueError("lookahead posture selection requires posture joints")
    self._posture_joint_mask = torch.tensor(
      posture_joint_flags,
      dtype=torch.bool,
      device=self.device,
    )
    self._left_posture_joint_mask = self._posture_joint_mask & torch.tensor(
      [joint_name.startswith("left_") for joint_name in self.robot.joint_names],
      dtype=torch.bool,
      device=self.device,
    )
    self._right_posture_joint_mask = self._posture_joint_mask & torch.tensor(
      [joint_name.startswith("right_") for joint_name in self.robot.joint_names],
      dtype=torch.bool,
      device=self.device,
    )
    self._posture_rollout_order = tuple(
      joint_id
      for joint_id in self._joint_rollout_order
      if posture_joint_flags[joint_id]
    )
    arm_link_flags = [
      any(token in link_name for token in cfg.link_filter.arm_joint_name_tokens)
      for link_name in cfg.link_filter.body_names
    ]
    if not any(arm_link_flags):
      raise ValueError("lookahead posture selection requires arm links")
    self._arm_link_mask = torch.tensor(
      arm_link_flags,
      dtype=torch.bool,
      device=self.device,
    )
    self._posture_hold_remaining_s = torch.zeros(self.num_envs, 2, device=self.device)
    self._filtered_joint_pos = torch.zeros(
      self.num_envs, self.robot.num_joints, device=self.device
    )
    self._filtered_joint_vel = torch.zeros_like(self._filtered_joint_pos)
    self._joint_position_residual = torch.zeros_like(self._filtered_joint_pos)
    self._joint_filter_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._propagate_targets, self._anchor_target_corrected = (
      _validate_propagation_flags(
        arm=bool(cfg.propagate_arm_corrections_to_body_targets),
        whole_body=bool(cfg.propagate_joint_corrections_to_body_targets),
      )
    )
    if self._propagate_targets:
      if self._anchor_target_corrected:
        # Whole-body mode: every hinge, chain rooted at the tracked pelvis.
        joint_local_ids = list(range(self.robot.num_joints))
      else:
        tokens = self.cfg.link_filter.arm_joint_name_tokens
        joint_local_ids = [
          joint_id
          for joint_id, joint_name in enumerate(self.robot.joint_names)
          if any(token in joint_name for token in tokens)
        ]
      self._init_target_propagation(
        joint_local_ids, allow_anchor_descendant=self._anchor_target_corrected
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

  def _motion_body_pos_w(self) -> torch.Tensor:
    return MotionCommand.body_pos_w.fget(self)  # type: ignore[union-attr]

  def _motion_body_quat_w(self) -> torch.Tensor:
    return MotionCommand.body_quat_w.fget(self)  # type: ignore[union-attr]

  def _motion_body_lin_vel_w(self) -> torch.Tensor:
    return MotionCommand.body_lin_vel_w.fget(self)  # type: ignore[union-attr]

  def _motion_body_ang_vel_w(self) -> torch.Tensor:
    return MotionCommand.body_ang_vel_w.fget(self)  # type: ignore[union-attr]

  def _update_reference_alignment(self, env_ids: torch.Tensor) -> None:
    """Align the current source frame to live robot XY and yaw."""
    if not self.cfg.align_reference_to_robot_each_step or env_ids.numel() == 0:
      return
    reference_anchor_quat_w = self._motion_body_quat_w()[
      :, self.motion_anchor_body_index
    ]
    yaw_delta_w, root_xy_w = _planar_reference_alignment(
      reference_anchor_quat_w[env_ids],
      self.robot_body_pos_w[env_ids, 0],
      self.robot_anchor_quat_w[env_ids],
    )
    self._reference_alignment_yaw_w[env_ids] = yaw_delta_w
    self._reference_alignment_root_xy_w[env_ids] = root_xy_w

  def _raw_body_pos_w(self) -> torch.Tensor:
    raw = self._motion_body_pos_w()
    if not self.cfg.align_reference_to_robot_each_step:
      return raw
    raw_root = raw[:, 0]
    return _apply_planar_reference_alignment(
      raw,
      raw_root,
      self._reference_alignment_root_xy_w,
      self._reference_alignment_yaw_w,
    )

  def _raw_body_quat_w(self) -> torch.Tensor:
    raw = self._motion_body_quat_w()
    if not self.cfg.align_reference_to_robot_each_step:
      return raw
    yaw = self._reference_alignment_yaw_w[:, None].expand(-1, raw.shape[1], -1)
    return quat_mul(yaw, raw)

  def _raw_body_lin_vel_w(self) -> torch.Tensor:
    raw = self._motion_body_lin_vel_w()
    if not self.cfg.align_reference_to_robot_each_step:
      return raw
    yaw = self._reference_alignment_yaw_w[:, None].expand(-1, raw.shape[1], -1)
    return quat_apply(yaw, raw)

  def _raw_body_ang_vel_w(self) -> torch.Tensor:
    raw = self._motion_body_ang_vel_w()
    if not self.cfg.align_reference_to_robot_each_step:
      return raw
    yaw = self._reference_alignment_yaw_w[:, None].expand(-1, raw.shape[1], -1)
    return quat_apply(yaw, raw)

  def _raw_joint_pos(self) -> torch.Tensor:
    return MotionCommand.joint_pos.fget(self)  # type: ignore[union-attr]

  def _raw_joint_vel(self) -> torch.Tensor:
    return MotionCommand.joint_vel.fget(self)  # type: ignore[union-attr]

  @property
  def command(self) -> torch.Tensor:
    """Expose either the filtered teacher or the nominal actor command."""
    if self.cfg.expose_filtered_command:
      return torch.cat((self.joint_pos, self.joint_vel), dim=1)
    return torch.cat((self._raw_joint_pos(), self._raw_joint_vel()), dim=1)

  @property
  def filtered_root_velocity_xy_w(self) -> torch.Tensor:
    """Privileged planar velocity selected by the reference filter."""
    return self._filtered_root_velocity_xy_w

  @property
  def filtered_joint_pos(self) -> torch.Tensor:
    """Privileged joint-position teacher after link-level filtering."""
    return self.joint_pos

  @property
  def filtered_joint_vel(self) -> torch.Tensor:
    """Privileged joint-velocity teacher after link-level filtering."""
    return self.joint_vel

  @property
  def obstacle_velocities_w(self) -> torch.Tensor:
    """Privileged smoothed world-frame velocity of every obstacle geom.

    Rows follow the same concatenated entity/geom order as the filter's
    obstacle tensors; use :attr:`obstacle_entity_slices` to select one entity.
    """
    return self._obstacle_velocity_w

  @property
  def obstacle_entity_slices(self) -> dict[str, slice]:
    """Map each obstacle entity name to its block in the obstacle tensors."""
    return dict(self._obstacle_entity_slices)

  def _init_target_propagation(
    self, joint_local_ids: list[int], *, allow_anchor_descendant: bool
  ) -> None:
    """Precompute the hinge chains that correct the body-target cloud.

    The tracked cloud holds only some bodies, while the hinges attach to
    intermediate bodies that are not tracked. The corrected targets are
    therefore produced by composing full chain forward kinematics from a
    tracked ancestor (the chain root) using model constants, once at the raw
    reference angles and once at the filtered angles; the difference displaces
    the tracked bodies. Arm-only mode passes the shoulder/elbow/wrist hinges
    (chains rooted at the tracked torso); whole-body mode passes every hinge
    (chains rooted at the tracked pelvis) and allows the motion anchor to be a
    corrected descendant. Only single-hinge-per-body chains rooted at a
    tracked body are supported; anything else raises at construction.
    """
    model = self._env.sim.mj_model
    parent_ids = model.body_parentid
    joint_body_ids = model.jnt_bodyid
    if not joint_local_ids:
      raise ValueError("target propagation requires at least one hinge joint")
    joint_global_ids = self._joint_global_ids.detach().cpu().tolist()
    body_to_joint: dict[int, int] = {}
    for joint_local_id in joint_local_ids:
      body_id = int(joint_body_ids[joint_global_ids[joint_local_id]])
      if body_id in body_to_joint:
        raise ValueError("target propagation supports one hinge per chain body")
      body_to_joint[body_id] = joint_local_id

    body_depth: dict[int, int] = {0: 0}

    def depth(body_id: int) -> int:
      if body_id not in body_depth:
        body_depth[body_id] = depth(int(parent_ids[body_id])) + 1
      return body_depth[body_id]

    chain_body_ids = sorted(
      body_to_joint, key=lambda body_id: (depth(body_id), body_id)
    )
    chain_index = {body_id: index for index, body_id in enumerate(chain_body_ids)}
    cloud_global_ids = (
      self.robot.indexing.body_ids[self.body_indexes].detach().cpu().tolist()
    )
    cloud_index_by_global = {
      int(body_id): cloud_id for cloud_id, body_id in enumerate(cloud_global_ids)
    }
    root_cloud_ids: list[int] = []
    root_slot_by_cloud: dict[int, int] = {}
    parent_chain_index: list[int] = []
    parent_root_slot: list[int] = []
    for body_id in chain_body_ids:
      parent_body = int(parent_ids[body_id])
      if parent_body in chain_index:
        parent_chain_index.append(chain_index[parent_body])
        parent_root_slot.append(0)
        continue
      cloud_id = cloud_index_by_global.get(parent_body)
      if cloud_id is None:
        raise ValueError("chains must attach to a tracked body or another chain body")
      if cloud_id not in root_slot_by_cloud:
        root_slot_by_cloud[cloud_id] = len(root_cloud_ids)
        root_cloud_ids.append(cloud_id)
      parent_chain_index.append(-1)
      parent_root_slot.append(root_slot_by_cloud[cloud_id])

    def to_device(values, columns: int) -> torch.Tensor:
      stacked = torch.tensor(values, dtype=torch.float32, device=self.device)
      return stacked.reshape(len(chain_body_ids), columns)

    self._arm_prop_parent_chain_index = tuple(parent_chain_index)
    self._arm_prop_parent_root_slot = tuple(parent_root_slot)
    self._arm_prop_root_cloud_ids = torch.tensor(
      root_cloud_ids, dtype=torch.long, device=self.device
    )
    self._arm_prop_body_pos_l = to_device(
      [model.body_pos[body_id].tolist() for body_id in chain_body_ids], 3
    )
    self._arm_prop_body_quat_l = to_device(
      [model.body_quat[body_id].tolist() for body_id in chain_body_ids], 4
    )
    self._arm_prop_joint_pos_l = to_device(
      [
        model.jnt_pos[joint_global_ids[body_to_joint[body_id]]].tolist()
        for body_id in chain_body_ids
      ],
      3,
    )
    self._arm_prop_joint_axis_l = to_device(
      [
        model.jnt_axis[joint_global_ids[body_to_joint[body_id]]].tolist()
        for body_id in chain_body_ids
      ],
      3,
    )
    self._arm_prop_chain_joint_ids = torch.tensor(
      [body_to_joint[body_id] for body_id in chain_body_ids],
      dtype=torch.long,
      device=self.device,
    )
    for body_id in chain_body_ids:
      joint_global_id = joint_global_ids[body_to_joint[body_id]]
      qpos_address = int(model.jnt_qposadr[joint_global_id])
      if abs(float(model.qpos0[qpos_address])) > 0.0:
        # The chain FK treats joint angles as rotations from the model's zero
        # configuration; a nonzero reference would rotate the displacement
        # without cancelling between the raw and filtered evaluations.
        raise ValueError(
          "target propagation requires qpos0 == 0 for every chain hinge"
        )

    descendant_cloud_ids: list[int] = []
    descendant_chain_ids: list[int] = []
    for cloud_id, body_id in enumerate(cloud_global_ids):
      ancestor = int(body_id)
      is_descendant = False
      while ancestor > 0:
        if ancestor in chain_index:
          is_descendant = True
          break
        ancestor = int(parent_ids[ancestor])
      if not is_descendant:
        continue
      if int(body_id) not in chain_index:
        raise ValueError("tracked bodies below the chain must own a chain hinge")
      descendant_cloud_ids.append(cloud_id)
      descendant_chain_ids.append(chain_index[int(body_id)])
    if (
      self.motion_anchor_body_index in descendant_cloud_ids
      and not allow_anchor_descendant
    ):
      raise ValueError(
        "the motion anchor cannot be a chain descendant unless whole-body "
        "propagation is enabled"
      )
    if 0 in descendant_cloud_ids:
      raise ValueError("the root body cannot be a chain descendant")
    self._arm_prop_cloud_ids = torch.tensor(
      descendant_cloud_ids, dtype=torch.long, device=self.device
    )
    self._arm_prop_cloud_chain_ids = torch.tensor(
      descendant_chain_ids, dtype=torch.long, device=self.device
    )
    # Historical name: in whole-body mode these caches cover every tracked
    # body incl. the anchor, not just the arm bodies.
    self._arm_body_target_offset_w = torch.zeros(
      self.num_envs, len(self.cfg.body_names), 3, device=self.device
    )
    self._arm_body_target_quat_delta_w = torch.zeros(
      self.num_envs, len(self.cfg.body_names), 4, device=self.device
    )
    self._arm_body_target_quat_delta_w[..., 0] = 1.0
    # Linear-velocity correction = finite difference of the position
    # displacement between consecutive command updates. Rows flagged for a
    # hold (fresh reset, motion wrap, first initialization) report zero for
    # one step so a displacement jump is never divided by dt.
    self._arm_body_target_lin_vel_w = torch.zeros(
      self.num_envs, len(self.cfg.body_names), 3, device=self.device
    )
    self._arm_prop_velocity_hold = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._arm_prop_step_dt = float(self._env.step_dt)

  def _update_arm_body_target_offsets(self, raw_joint_pos: torch.Tensor) -> None:
    """Refresh cached arm body-target pose corrections from the joint filter.

    Positions receive the FK displacement; orientations receive the world
    quaternion delta between the filtered and raw chain poses; linear
    velocities receive the finite difference of the position displacement
    across consecutive updates (angular velocity targets stay raw: that
    reward's 3.14 rad/s std makes the residual's effect negligible).
    Properties never recompute; they apply these caches.
    """
    previous_offset_w = self._arm_body_target_offset_w.clone()
    delta = torch.where(
      self._joint_filter_initialized[:, None],
      self._filtered_joint_pos - raw_joint_pos,
      torch.zeros_like(raw_joint_pos),
    )
    arm_angles_delta = delta[:, self._arm_prop_chain_joint_ids]
    self._arm_body_target_offset_w.zero_()
    self._arm_body_target_quat_delta_w.zero_()
    self._arm_body_target_quat_delta_w[..., 0] = 1.0
    if bool((arm_angles_delta.abs() > 1e-4).any()):
      self._compute_arm_body_target_offsets(raw_joint_pos, arm_angles_delta)
    finite_difference_w = (
      self._arm_body_target_offset_w - previous_offset_w
    ) / self._arm_prop_step_dt
    self._arm_body_target_lin_vel_w = torch.where(
      self._arm_prop_velocity_hold[:, None, None],
      torch.zeros_like(finite_difference_w),
      finite_difference_w,
    )
    self._arm_prop_velocity_hold.zero_()

  def _compute_arm_body_target_offsets(
    self, raw_joint_pos: torch.Tensor, arm_angles_delta: torch.Tensor
  ) -> None:
    """Run the raw and filtered chain FK and store pose corrections."""
    cloud_pos_w = self._raw_body_pos_w()
    cloud_quat_w = self._raw_body_quat_w()
    root_pos_w = cloud_pos_w[:, self._arm_prop_root_cloud_ids]
    root_quat_w = cloud_quat_w[:, self._arm_prop_root_cloud_ids]
    raw_angles = raw_joint_pos[:, self._arm_prop_chain_joint_ids]
    raw_positions, raw_quaternions = hinge_chain_body_positions(
      root_pos_w,
      root_quat_w,
      parent_chain_index=self._arm_prop_parent_chain_index,
      parent_root_slot=self._arm_prop_parent_root_slot,
      body_pos_l=self._arm_prop_body_pos_l,
      body_quat_l=self._arm_prop_body_quat_l,
      joint_pos_l=self._arm_prop_joint_pos_l,
      joint_axis_l=self._arm_prop_joint_axis_l,
      joint_angles=raw_angles,
    )
    adjusted_positions, adjusted_quaternions = hinge_chain_body_positions(
      root_pos_w,
      root_quat_w,
      parent_chain_index=self._arm_prop_parent_chain_index,
      parent_root_slot=self._arm_prop_parent_root_slot,
      body_pos_l=self._arm_prop_body_pos_l,
      body_quat_l=self._arm_prop_body_quat_l,
      joint_pos_l=self._arm_prop_joint_pos_l,
      joint_axis_l=self._arm_prop_joint_axis_l,
      joint_angles=raw_angles + arm_angles_delta,
    )
    self._arm_body_target_offset_w[:, self._arm_prop_cloud_ids] = (
      adjusted_positions[:, self._arm_prop_cloud_chain_ids]
      - raw_positions[:, self._arm_prop_cloud_chain_ids]
    )
    # World-frame left delta: tolerant of any constant right-side convention
    # offset between the tracked cloud quaternions and the model-constant FK.
    self._arm_body_target_quat_delta_w[:, self._arm_prop_cloud_ids] = quat_mul(
      adjusted_quaternions[:, self._arm_prop_cloud_chain_ids],
      quat_inv(raw_quaternions[:, self._arm_prop_cloud_chain_ids]),
    )

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
  def body_quat_w(self) -> torch.Tensor:
    raw = self._raw_body_quat_w()
    if not getattr(self, "_propagate_targets", False):
      return raw
    corrected = raw.clone()
    corrected[:, self._arm_prop_cloud_ids] = quat_mul(
      self._arm_body_target_quat_delta_w[:, self._arm_prop_cloud_ids],
      raw[:, self._arm_prop_cloud_ids],
    )
    return corrected

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    # Angular-velocity targets stay raw even when arm corrections propagate:
    # the tracking reward's 3.14 rad/s std makes the residual's rate negligible.
    return self._raw_body_ang_vel_w()

  @property
  def body_pos_w(self) -> torch.Tensor:
    raw = self._raw_body_pos_w()
    if not hasattr(self, "_filter_initialized"):
      return raw
    offset_xy = self._filtered_root_xy_w - raw[:, 0, :2]
    offset_xy = torch.where(self._filter_initialized[:, None], offset_xy, 0.0)
    filtered = raw.clone()
    filtered[..., :2] += offset_xy[:, None, :]
    if getattr(self, "_propagate_targets", False):
      filtered[:, self._arm_prop_cloud_ids] += self._arm_body_target_offset_w[
        :, self._arm_prop_cloud_ids
      ]
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
    if getattr(self, "_propagate_targets", False):
      filtered[:, self._arm_prop_cloud_ids] += self._arm_body_target_lin_vel_w[
        :, self._arm_prop_cloud_ids
      ]
    return filtered

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    raw = self._raw_body_pos_w()[:, self.motion_anchor_body_index]
    if not hasattr(self, "_filter_initialized"):
      return raw
    raw_root_xy = self._raw_body_pos_w()[:, 0, :2]
    offset_xy = self._filtered_root_xy_w - raw_root_xy
    offset_xy = torch.where(self._filter_initialized[:, None], offset_xy, 0.0)
    filtered = raw.clone()
    filtered[:, :2] += offset_xy
    if getattr(self, "_anchor_target_corrected", False):
      # Whole-body propagation: the anchor sits below the waist hinges.
      filtered += self._arm_body_target_offset_w[:, self.motion_anchor_body_index]
    return filtered

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    raw = self._raw_body_lin_vel_w()[:, self.motion_anchor_body_index]
    if not hasattr(self, "_filter_initialized"):
      return raw
    raw_root_velocity_xy = self._raw_body_lin_vel_w()[:, 0, :2]
    velocity_delta_xy = self._filtered_root_velocity_xy_w - raw_root_velocity_xy
    velocity_delta_xy = torch.where(
      self._filter_initialized[:, None], velocity_delta_xy, 0.0
    )
    filtered = raw.clone()
    filtered[:, :2] += velocity_delta_xy
    if getattr(self, "_anchor_target_corrected", False):
      filtered += self._arm_body_target_lin_vel_w[
        :, self.motion_anchor_body_index
      ]
    return filtered

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    raw = self._raw_body_quat_w()[:, self.motion_anchor_body_index]
    if not getattr(self, "_anchor_target_corrected", False):
      return raw
    return quat_mul(
      self._arm_body_target_quat_delta_w[:, self.motion_anchor_body_index], raw
    )

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._raw_body_ang_vel_w()[:, self.motion_anchor_body_index]

  # ---------------------------------------------------------------------------
  # Play-only ghost override: draw the ADJUSTER's command instead of (or next
  # to) the privileged teacher reference. The adjuster's residuals live inside
  # the actor network, so a play script must hand them in each step.
  # ---------------------------------------------------------------------------
  ADJUSTER_GHOST_COLOR = (0.35, 0.55, 0.95, 0.5)

  def set_ghost_override(
    self,
    joint_residual: torch.Tensor | None,
    *,
    show_teacher: bool = False,
  ) -> None:
    """Pose the debug ghost from ``raw joint command + joint_residual``.

    ``joint_residual`` is ``(num_envs, num_joints)`` in robot joint order (the
    adjuster's injected joint correction); ``None`` restores the teacher ghost.
    The ghost root follows the live-aligned RAW root (the adjuster has no
    pelvis channel). With ``show_teacher`` the teacher ghost is drawn as well
    (its configured color) so both references can be compared.
    """
    if joint_residual is None:
      self._ghost_override_residual = None
      return
    self._ghost_override_residual = joint_residual.detach()
    self._ghost_override_show_teacher = bool(show_teacher)

  def _adjuster_ghost_model(self):
    if getattr(self, "_adjuster_ghost_model_cache", None) is None:
      model = copy.deepcopy(self._env.sim.mj_model)
      color = np.array(self.ADJUSTER_GHOST_COLOR, dtype=np.float32)
      for gi in range(model.ngeom):
        if model.geom_contype[gi] != 0 or model.geom_conaffinity[gi] != 0:
          model.geom_rgba[gi, 3] = 0
        else:
          model.geom_rgba[gi] = color
      self._adjuster_ghost_model_cache = model
    return self._adjuster_ghost_model_cache

  def adjuster_ghost_qpos(self, env_index: int) -> np.ndarray:
    """Full ``qpos`` for the adjuster ghost of one environment."""
    residual = self._ghost_override_residual
    if residual is None:
      raise RuntimeError("no adjuster ghost override is set")
    entity = self._env.scene[self.cfg.entity_name]
    indexing = entity.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()
    qpos = np.zeros(self._env.sim.mj_model.nq)
    qpos[free_joint_q_adr[0:3]] = self._raw_body_pos_w()[env_index, 0].cpu().numpy()
    qpos[free_joint_q_adr[3:7]] = self._raw_body_quat_w()[env_index, 0].cpu().numpy()
    joints = self._raw_joint_pos()[env_index] + residual[env_index]
    qpos[joint_q_adr] = joints.cpu().numpy()
    return qpos

  def _debug_vis_impl(self, visualizer) -> None:
    residual = getattr(self, "_ghost_override_residual", None)
    if residual is None or self.cfg.viz.mode != "ghost":
      super()._debug_vis_impl(visualizer)
      return
    if getattr(self, "_ghost_override_show_teacher", False):
      super()._debug_vis_impl(visualizer)
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    model = self._adjuster_ghost_model()
    for batch in env_indices:
      visualizer.add_ghost_mesh(
        self.adjuster_ghost_qpos(batch),
        model=model,
        label=f"adjuster_ghost_{batch}",
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Sample the start frame first: every raw accessor below indexes the
    # motion by ``time_steps``, so alignment and filter state initialize at
    # the sampled frame. Reset-state randomization is deliberately not
    # applied (exact reference write, as before).
    self._sample_start_time_steps(env_ids)
    self._update_reference_alignment(env_ids)
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
    if getattr(self, "_propagate_targets", False):
      self._arm_body_target_offset_w[env_ids] = 0.0
      self._arm_body_target_quat_delta_w[env_ids] = 0.0
      self._arm_body_target_quat_delta_w[env_ids, :, 0] = 1.0
      self._arm_body_target_lin_vel_w[env_ids] = 0.0
      self._arm_prop_velocity_hold[env_ids] = True
    self._write_current_frame_to_sim(env_ids)

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

  def _select_link_filter_obstacles(
    self,
    centers_w: torch.Tensor,
    quaternions_w: torch.Tensor,
    sizes: torch.Tensor,
    obstacle_velocity_w: torch.Tensor,
    surface_clearance_m: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep only nearby grouped obstacles for the expensive per-link filter.

    Planar filtering still evaluates the complete obstacle set. Group ranking
    reuses its capsule-surface clearances, so large or horizontal capsules are
    ranked by actual planar proximity rather than center distance. Ungrouped
    entities, such as the articulated primary human, remain complete.
    """

    selected_centers = []
    selected_quaternions = []
    selected_sizes = []
    selected_velocities = []
    offset = 0
    for geom_count, capsules_per_group, nearest_groups in zip(
      self._obstacle_geom_counts,
      self._link_filter_capsules_per_group,
      self._link_filter_nearest_groups,
      strict=True,
    ):
      entity_slice = slice(offset, offset + geom_count)
      entity_centers = centers_w[:, entity_slice]
      entity_quaternions = quaternions_w[:, entity_slice]
      entity_sizes = sizes[:, entity_slice]
      entity_velocities = obstacle_velocity_w[:, entity_slice]
      if capsules_per_group is not None and nearest_groups is not None:
        group_count = geom_count // capsules_per_group
        selected_group_count = min(nearest_groups, group_count)
        entity_clearance = surface_clearance_m[:, entity_slice].reshape(
          self.num_envs, group_count, capsules_per_group
        )
        entity_active = active[:, entity_slice].reshape(
          self.num_envs, group_count, capsules_per_group
        )
        group_clearance = torch.where(
          entity_active,
          entity_clearance,
          torch.full_like(entity_clearance, torch.inf),
        ).amin(dim=-1)
        group_ids = torch.topk(
          group_clearance,
          k=selected_group_count,
          dim=1,
          largest=False,
          sorted=False,
        ).indices
        capsule_offsets = torch.arange(
          capsules_per_group, device=self.device, dtype=torch.long
        )
        capsule_ids = (
          group_ids[..., None] * capsules_per_group + capsule_offsets
        ).reshape(self.num_envs, -1)
        gather_xyz = capsule_ids[..., None].expand(-1, -1, 3)
        entity_centers = torch.gather(entity_centers, 1, gather_xyz)
        entity_quaternions = torch.gather(
          entity_quaternions,
          1,
          capsule_ids[..., None].expand(-1, -1, 4),
        )
        entity_sizes = torch.gather(entity_sizes, 1, gather_xyz)
        entity_velocities = torch.gather(entity_velocities, 1, gather_xyz)
      selected_centers.append(entity_centers)
      selected_quaternions.append(entity_quaternions)
      selected_sizes.append(entity_sizes)
      selected_velocities.append(entity_velocities)
      offset += geom_count
    return (
      torch.cat(selected_centers, dim=1),
      torch.cat(selected_quaternions, dim=1),
      torch.cat(selected_sizes, dim=1),
      torch.cat(selected_velocities, dim=1),
    )

  def _estimate_obstacle_velocities(
    self,
    centers_w: torch.Tensor,
    active: torch.Tensor,
  ) -> torch.Tensor:
    cfg = self.cfg.planar_filter
    dt = self._env.step_dt
    initialized = self._obstacle_history_initialized
    new_envs = ~initialized
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
    self._obstacle_velocity_w *= self._obstacle_velocity_decay
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

    # Every candidate is identically zero outside the posture-joint mask.
    # Skipping those zero-angle Rodrigues updates preserves the exact finite-FK
    # candidate poses while avoiding needless sequential CUDA launches.
    for joint_id in self._posture_rollout_order:
      axis = joint_axes[:, :, joint_id]
      anchor = joint_anchors[:, :, joint_id]
      angle = candidate_joint_delta[:, :, joint_id]

      descendant_link_ids = self._joint_descendant_link_ids[joint_id]
      if descendant_link_ids.numel() > 0:
        descendant_positions = candidate_link_positions[:, :, descendant_link_ids]
        relative_positions = descendant_positions - anchor[:, :, None]
        candidate_link_positions[:, :, descendant_link_ids] = anchor[
          :, :, None
        ] + self._rotate_about_axis(relative_positions, axis, angle)

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
      posture_joint_mask=self._posture_joint_mask,
      left_joint_mask=self._left_posture_joint_mask,
      right_joint_mask=self._right_posture_joint_mask,
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
      arm_link_mask=self._arm_link_mask,
    )
    selected_sides = torch.stack(
      (
        preferred_velocity[:, self._left_posture_joint_mask].abs().amax(dim=1) > 1e-6,
        preferred_velocity[:, self._right_posture_joint_mask].abs().amax(dim=1) > 1e-6,
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
      held_left[:, None] * candidates[:, 2] + held_right[:, None] * candidates[:, 3]
    )
    held_joint_mask = (held_left[:, None] & self._left_posture_joint_mask[None]) | (
      held_right[:, None] & self._right_posture_joint_mask[None]
    )
    preferred_velocity = torch.where(held_joint_mask, held_velocity, preferred_velocity)
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

    wrapped = self._advance_time_steps(replay_env_ids)
    self._update_reference_alignment(replay_env_ids)
    raw_root_pos = self._raw_body_pos_w()[:, 0]
    raw_root_velocity = self._raw_body_lin_vel_w()[:, 0]
    raw_joint_pos = self._raw_joint_pos()
    raw_joint_vel = self._raw_joint_vel()
    wrapped_ids = replay_env_ids[wrapped]
    self._filtered_root_xy_w[wrapped_ids] = raw_root_pos[wrapped_ids, :2]
    self._filtered_root_velocity_xy_w[wrapped_ids] = raw_root_velocity[wrapped_ids, :2]
    self._root_translation_residual_xy_w[wrapped_ids] = 0.0
    self._filtered_joint_pos[wrapped_ids] = raw_joint_pos[wrapped_ids]
    self._filtered_joint_vel[wrapped_ids] = raw_joint_vel[wrapped_ids]
    self._joint_position_residual[wrapped_ids] = 0.0
    self._posture_hold_remaining_s[wrapped_ids] = 0.0
    self._obstacle_history_initialized[wrapped_ids] = False
    uninitialized = ~self._filter_initialized[replay_env_ids]
    init_ids = replay_env_ids[uninitialized]
    self._filtered_root_xy_w[init_ids] = raw_root_pos[init_ids, :2]
    self._filtered_root_velocity_xy_w[init_ids] = raw_root_velocity[init_ids, :2]
    self._root_translation_residual_xy_w[init_ids] = 0.0
    self._filter_initialized[init_ids] = True
    uninitialized_joints = ~self._joint_filter_initialized[replay_env_ids]
    init_joint_ids = replay_env_ids[uninitialized_joints]
    self._filtered_joint_pos[init_joint_ids] = raw_joint_pos[init_joint_ids]
    self._filtered_joint_vel[init_joint_ids] = raw_joint_vel[init_joint_ids]
    self._joint_position_residual[init_joint_ids] = 0.0
    self._joint_filter_initialized[init_joint_ids] = True
    if self._propagate_targets:
      # A wrap or first initialization resets the filtered joints to raw; the
      # displacement jump must not be read as a velocity.
      self._arm_prop_velocity_hold[wrapped_ids] = True
      self._arm_prop_velocity_hold[init_joint_ids] = True

    if self.cfg.disable_filters:
      self._pass_through_raw_reference(
        replay_env_ids, raw_root_pos, raw_root_velocity, raw_joint_pos, raw_joint_vel
      )
      self._finish_update(replay_env_ids)
      return

    if not self.cfg.closed_loop_root_target:
      # Open loop: re-anchor the target to the live-aligned raw root each step.
      # Closed loop keeps the persistent world-frame target instead.
      self._filtered_root_xy_w[replay_env_ids] = (
        raw_root_pos[replay_env_ids, :2]
        + self._root_translation_residual_xy_w[replay_env_ids]
      )
    pre_filter_root_xy_w = self._filtered_root_xy_w.clone()
    centers_w, quaternions_w, sizes = self._obstacle_tensors()
    filter_root_pos = raw_root_pos.clone()
    if not self.cfg.planar_filter_at_robot_root:
      # Legacy: clearance and CBF are evaluated at the (possibly leading)
      # target. With the flag the live-aligned raw root, i.e. the robot,
      # is the evaluation point, so the escape velocity keeps flowing while
      # the ROBOT is still in danger even after the target has escaped.
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
      robot_position_xy_w=filter_root_pos[:, :2],
      nominal_velocity_xy_w=nominal_velocity,
      closest_points_xy_w=closest_xy,
      surface_clearances_m=clearance,
      obstacle_velocities_xy_w=obstacle_velocity,
      active=active,
    )
    self._filtered_root_velocity_xy_w[replay_env_ids] = result.velocity_w[
      replay_env_ids
    ]
    filtered_xy_w, residual_xy_w = advance_filtered_root_xy(
      pre_filter_xy_w=pre_filter_root_xy_w,
      raw_root_xy_w=raw_root_pos[:, :2],
      raw_root_velocity_xy_w=raw_root_velocity[:, :2],
      filtered_velocity_xy_w=result.velocity_w,
      translation_residual_xy_w=self._root_translation_residual_xy_w,
      step_dt=self._env.step_dt,
      closed_loop=self.cfg.closed_loop_root_target,
      max_lead_m=self.cfg.max_root_lead_m,
    )
    self._filtered_root_xy_w[replay_env_ids] = filtered_xy_w[replay_env_ids]
    self._root_translation_residual_xy_w[replay_env_ids] = residual_xy_w[
      replay_env_ids
    ]

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

    (
      link_centers_w,
      link_quaternions_w,
      link_sizes,
      link_obstacle_velocity,
    ) = self._select_link_filter_obstacles(
      centers_w,
      quaternions_w,
      sizes,
      obstacle_velocity,
      clearance,
      active,
    )
    joint_avoidance_velocity, preferred_posture_velocity = (
      self._joint_velocity_avoidance_correction(
        link_centers_w,
        link_quaternions_w,
        link_sizes,
        link_obstacle_velocity,
      )
    )
    effective_joint_recovery_velocity = gate_joint_recovery_during_posture(
      joint_recovery_velocity,
      preferred_posture_velocity,
    )
    filtered_joint_velocity = (
      raw_joint_vel + effective_joint_recovery_velocity + joint_avoidance_velocity
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
    if self._propagate_targets:
      self._update_arm_body_target_offsets(raw_joint_pos)
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
    self._finish_update(replay_env_ids)

  def _pass_through_raw_reference(
    self,
    replay_env_ids: torch.Tensor,
    raw_root_pos: torch.Tensor,
    raw_root_velocity: torch.Tensor,
    raw_joint_pos: torch.Tensor,
    raw_joint_vel: torch.Tensor,
  ) -> None:
    """``disable_filters``: the reference IS the live-aligned raw motion.

    Neither CBF runs, so every filtered quantity equals its raw counterpart,
    every residual (and therefore the teacher correction the co-adjust head
    is supervised towards) is zero, and the arm body-target offsets are
    zero. Only the clearance metric is still evaluated, at the robot root,
    so the no-avoidance baseline logs how close the humans actually got.
    """
    self._filtered_root_xy_w[replay_env_ids] = raw_root_pos[replay_env_ids, :2]
    self._filtered_root_velocity_xy_w[replay_env_ids] = raw_root_velocity[
      replay_env_ids, :2
    ]
    self._root_translation_residual_xy_w[replay_env_ids] = 0.0
    self._filtered_joint_pos[replay_env_ids] = raw_joint_pos[replay_env_ids]
    self._filtered_joint_vel[replay_env_ids] = raw_joint_vel[replay_env_ids]
    self._joint_position_residual[replay_env_ids] = 0.0
    self._posture_hold_remaining_s[replay_env_ids] = 0.0
    if self._propagate_targets:
      # Filtered joints equal raw joints, so this zeroes the cached offsets
      # (and their finite-difference velocities) through the normal path.
      self._update_arm_body_target_offsets(raw_joint_pos)
    centers_w, quaternions_w, sizes = self._obstacle_tensors()
    _, clearance, active = planar_capsule_geometry(
      raw_root_pos,
      centers_w,
      quaternions_w,
      sizes,
      robot_radius_m=self.cfg.planar_filter.robot_radius_m,
      vertical_gate_m=self.cfg.planar_filter.vertical_gate_m,
    )
    gated = torch.where(active, clearance, torch.full_like(clearance, math.inf))
    self.metrics["filter_minimum_clearance_m"][replay_env_ids] = gated.amin(dim=-1)[
      replay_env_ids
    ]
    for name in (
      "joint_filter_reference_residual_rad",
      "filter_intervention_speed_mps",
      "filter_reference_offset_m",
      "filter_cbf_violation_mps",
    ):
      self.metrics[name][replay_env_ids] = 0.0

  def _finish_update(self, replay_env_ids: torch.Tensor) -> None:
    """Common tail: optional ghost write-to-sim, then relative body poses."""
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


def advance_filtered_root_xy(
  *,
  pre_filter_xy_w: torch.Tensor,
  raw_root_xy_w: torch.Tensor,
  raw_root_velocity_xy_w: torch.Tensor,
  filtered_velocity_xy_w: torch.Tensor,
  translation_residual_xy_w: torch.Tensor,
  step_dt: float,
  closed_loop: bool,
  max_lead_m: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Advance the filtered root XY target by one command update.

  Open loop (legacy): the translation residual integrates the difference
  between the filtered and raw root velocities and the target is re-anchored
  to the live-aligned raw root: ``residual += dt * (v_filt - v_raw)``,
  ``filtered = raw + residual``. The target therefore runs ahead of the robot
  by the residual whether or not the robot follows.

  Closed loop: the target integrates the filtered velocity in the world frame
  from the position the filter was evaluated at, ``filtered = pre + dt *
  v_filt``, and the residual is the resulting offset from the raw root,
  ``residual = filtered - raw``. A robot that follows the filtered velocity
  sees a constant residual; one that does not accumulates a root position
  error. The planar filter's recovery term (``recovery_gain``, capped at
  ``max_recovery_speed_mps``) pulls the target back toward the raw root, so
  the offset stays bounded only while the reference planar speed stays under
  ``max_recovery_speed_mps``; otherwise it keeps growing (no tracking
  termination watches the planar offset: ``anchor_pos`` is z-only).

  ``max_lead_m`` (leash): after integration the offset from the raw root is
  clamped to that norm along its own direction, in both modes. Under live
  alignment the raw root IS the robot, so the leash keeps the target inside
  the region where the nominal root-position reward (std 0.3 m) still has
  gradient; joint@14k measured a dead term (value 0, gradient 0) on 19 % of
  frames and 61 % of collisions once the target led by more than 0.6 m.
  ``None`` keeps the unbounded legacy behaviour. Returns ``(filtered_xy_w,
  residual_xy_w)`` as new tensors; inputs are not mutated.
  """
  if closed_loop:
    filtered_xy_w = pre_filter_xy_w + step_dt * filtered_velocity_xy_w
    residual_xy_w = filtered_xy_w - raw_root_xy_w
  else:
    residual_xy_w = translation_residual_xy_w + step_dt * (
      filtered_velocity_xy_w - raw_root_velocity_xy_w
    )
  if max_lead_m is not None:
    lead = torch.linalg.vector_norm(residual_xy_w, dim=-1, keepdim=True)
    scale = torch.clamp(max_lead_m / lead.clamp_min(1.0e-9), max=1.0)
    residual_xy_w = residual_xy_w * scale
  return raw_root_xy_w + residual_xy_w, residual_xy_w


@dataclass(kw_only=True)
class PlanarFilteredReplayMotionCommandCfg(KinematicReplayMotionCommandCfg):
  """Build replay with privileged planar and link/joint CBF filters."""

  obstacle_entity_names: tuple[str, ...]
  link_filter_capsules_per_group: tuple[int | None, ...] = ()
  link_filter_nearest_groups: tuple[int | None, ...] = ()
  write_reference_to_sim: bool = True
  align_reference_to_robot_each_step: bool = False
  expose_filtered_command: bool = True
  # Propagate the link filter's ARM joint corrections into the task-space
  # body targets via reference-frame FK, so body pose rewards and the
  # ee_body_pos termination stop opposing limb corrections. Positions,
  # orientations, and linear velocities (finite-differenced displacement)
  # are corrected; angular velocity targets stay raw.
  propagate_arm_corrections_to_body_targets: bool = False
  # Propagate EVERY joint correction (legs, waist, arms) into the body
  # targets via FK rooted at the pelvis, anchor (torso) included, so the
  # nominal tracking rewards and the ee_body_pos termination evaluate the
  # fully filtered reference. Mutually exclusive with the arm-only flag.
  propagate_joint_corrections_to_body_targets: bool = False
  # Integrate the filtered root position in the world frame instead of
  # re-anchoring it to the live robot root each step, so the nominal root
  # position reward measures whether the robot actually executed the planar
  # escape. The recovery term still bounds the offset.
  closed_loop_root_target: bool = False
  # Leash: clamp the filtered root target's planar offset from the raw
  # (robot-aligned) root to this norm so the nominal root-position reward
  # keeps gradient. ``None`` leaves the offset unbounded (legacy).
  max_root_lead_m: float | None = None
  # Evaluate the planar clearance and CBF at the robot-aligned raw root
  # instead of at the filtered target, so the escape velocity tracks the
  # robot's actual danger rather than the target's.
  planar_filter_at_robot_root: bool = False
  # Nominal-reference baseline: skip BOTH CBF filters so the reference is the
  # live-aligned raw motion, every residual/teacher correction is zero and
  # the arm body-target offsets vanish. Rewards, terminations and
  # observations keep reading the same (now unfiltered) properties.
  disable_filters: bool = False
  planar_filter: PlanarCbfReferenceFilterCfg = field(
    default_factory=PlanarCbfReferenceFilterCfg
  )
  link_filter: LinkCbfReferenceFilterCfg = field(
    default_factory=LinkCbfReferenceFilterCfg
  )

  def __post_init__(self) -> None:
    if self.max_root_lead_m is not None and self.max_root_lead_m <= 0.0:
      raise ValueError("max_root_lead_m must be positive or None")

  def build(self, env) -> PlanarFilteredReplayMotionCommand:
    return PlanarFilteredReplayMotionCommand(self, env)


__all__ = [
  "KinematicReplayMotionCommand",
  "KinematicReplayMotionCommandCfg",
  "LibraryMotionLoader",
  "is_motion_manifest",
  "PlanarFilteredReplayMotionCommand",
  "PlanarFilteredReplayMotionCommandCfg",
  "advance_filtered_root_xy",
  "hinge_chain_body_positions",
]
