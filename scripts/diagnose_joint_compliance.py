"""Measure joint-filter compliance of an auxiliary checkpoint, per arm side.

The handoff's five signals on active avoidance frames:

1. teacher joint residual versus predicted joint residual (MAE, cosine);
2. mechanical path: for legacy checkpoints, the predicted residual versus the
   residual actually added to the action; for co-adjust checkpoints (command
   injection instead of an action residual), the injected correction's norm;
3. predicted correction versus achieved joint motion over 0.1 s and 0.5 s;
4. fraction of active frames where arm motion improves actual link clearance;
5. state-based compliance: where the robot's current joint pose sits along
   the raw->filtered reference segment (a STATE offset, not a motion command,
   so it is meaningful even when the policy holds still at the corrected
   pose);
6. planar (root) state compliance: on frames where the CBF planar-velocity
   filter meaningfully intervenes, how much of the commanded escape velocity
   (relative to the nominal reference) the robot's root actually realizes.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from safe_mimic.evaluation import ENCOUNTER_PRESETS, apply_encounter_preset
from safe_mimic.rl import PerceptiveLidarActor
from safe_mimic.tasks import (
  LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
  UNIFIED_ROOT_LEAD_M,
  mdp,
)
from safe_mimic.tasks.env_cfg import (
  DEFAULT_G1_BALLET_MANIFEST,
  DEFAULT_G1_BALLET_MIRROR_MANIFEST,
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)

BLIND_NOMINAL_TASK_ID = (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID
)
BLIND_NOHUMANS_TASK_ID = (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID
)
MOVES_TASK_ID = LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID

SIDES = ("left", "right")
ARM_JOINT_TOKENS = ("shoulder", "elbow", "wrist")
HORIZON_STEPS = (5, 25)
RING_LENGTH = 26
CLEARANCE_GATE_M = 0.8
PLANAR_ACTIVE_MPS = 0.1
PLANAR_STATE_ACTIVE_MPS = 0.1
STABLE_WINDOW_STEPS = 5
STABLE_THRESHOLD_RAD = 0.02


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1.0e-8)


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  return torch.where(mask, values, torch.zeros_like(values)).sum()


def _mean(total: torch.Tensor, count: torch.Tensor) -> float | None:
  count_value = int(count)
  if count_value == 0:
    return None
  return float(total) / count_value


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--task-id",
    choices=(
      LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID,
    ),
    default=LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
    help="rl-cfg/runner task id used to load the checkpoint's actor config",
  )
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--seed", type=int, default=31)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--min-radius-m", type=float, default=0.75)
  parser.add_argument("--max-radius-m", type=float, default=4.0)
  parser.add_argument("--min-intercept-s", type=float, default=0.5)
  parser.add_argument("--max-intercept-s", type=float, default=4.0)
  parser.add_argument("--total-steps", type=int, default=600)
  parser.add_argument("--active-threshold-rad", type=float, default=0.05)
  parser.add_argument(
    "--expose-filtered-command",
    action="store_true",
    help=(
      "Expose the filter-adjusted joint targets in the actor's command "
      "observation (tests reference-adjustment obedience)."
    ),
  )
  parser.add_argument(
    "--oracle-injection",
    action="store_true",
    help=(
      "Compute actions via policy.action_with_avoidance_override using the "
      "ground-truth avoidance teacher instead of the policy's own predicted "
      "correction (in co-adjust mode this injects the teacher's joint "
      "corrections into the command slice and uses the teacher planar as "
      "compass), isolating tracking compliance from prediction error."
    ),
  )
  parser.add_argument(
    "--random-start",
    action="store_true",
    help=(
      "Sample the motion command's start frame uniformly instead of pinning "
      "frame 0 (breaks comparability with existing frame-0 artifacts)."
    ),
  )
  parser.add_argument(
    "--encounter-preset",
    choices=tuple(ENCOUNTER_PRESETS),
    default="standard",
    help="encounter distribution: 'standard' = the frozen 0.75-4 m / 0.5-4 s envelope; "
    "'slow' = the slow training regime (spawn >= 1.8 m, approach <= 0.75 m/s, 10 s "
    "episodes), uniform over bins. Applied after the range flags above.",
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"checkpoint not found: {args.checkpoint}")
  if not args.motion_file.is_file():
    parser.error(f"motion file not found: {args.motion_file}")
  if args.num_envs < 1:
    parser.error("--num-envs must be positive")
  if args.total_steps < 1:
    parser.error("--total-steps must be positive")
  configure_torch_backends()

  if args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID:
    # The unified task's reward set and both filtered-reference flags must
    # match training exactly; build it from its own env-cfg builder instead
    # of the auxiliary builder plus FKC/FKC2 patches below.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(play=True)
  elif args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID:
    # Same unified builder plus the active-joint reward term added on top of
    # it during training; must match exactly like the unified branch above.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True, active_joint_reward=True
    )
  elif args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID:
    # Unified-Joint plus the leashed root target and robot-evaluated CBF.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
    )
  elif args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID:
    # Leash plus slow-regime encounters and the filter-gated ee termination.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      slow_regime=True,
      filter_gated_ee_termination=True,
    )
  elif args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID:
    # Leash-Slow with dense encounters and the strict stock ee termination.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      slow_regime=True,
      dense_encounters=True,
    )
  elif args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID:
    # Leash (canonical setup) with the whole ballet library as the reference
    # (pass the manifest as --motion-file).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    )
  elif args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID:
    # Leash-Ballet plus the slow encounter regime (eval overrides the spawn
    # and TTC ranges anyway; the flag keeps the cfg identical to training).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      slow_regime=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    )
  elif args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID:
    # Leash-Ballet with the lag-aware ee_body_pos termination.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      lag_aware_ee_termination=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    )
  elif (
    args.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID
  ):
    # Blind baseline: Leash-Ballet with the actor's LiDAR term reading "no returns".
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      blind_actor=True,
    )
  elif args.task_id == BLIND_NOMINAL_TASK_ID:
    # Blind + nominal baseline: blind actor AND both CBF filters off (raw reference).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      blind_actor=True,
      nominal_reference=True,
    )
  elif args.task_id == BLIND_NOHUMANS_TASK_ID:
    # Blind + nominal + no humans in training (play cfg keeps the populated scene).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      blind_actor=True,
      nominal_reference=True,
      training_humans=False,
    )
  elif args.task_id == MOVES_TASK_ID:
    # Escape moves: mirrored ballet library, travelling steps as the escape.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=True,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MIRROR_MANIFEST),
      escape_moves=True,
    )
  else:
    cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=True)
  # Pin frame-0 starts for comparability with existing artifacts.
  cfg.commands["motion"].sampling_mode = "uniform" if args.random_start else "start"
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.episode_length_s = 6.0
  cfg.commands["motion"].motion_file = str(args.motion_file)
  if args.expose_filtered_command:
    # Feed the actor the privileged CBF-filtered reference instead of raw.
    cfg.commands["motion"].expose_filtered_command = True
  if args.task_id in (
    LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  ):
    # Match training semantics: the FKC task's body position/orientation
    # targets follow the filtered joint corrections, so task-space metrics
    # reported here are consistent with what the checkpoint was trained on.
    cfg.commands["motion"].propagate_arm_corrections_to_body_targets = True
  if args.task_id == LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID:
    # Match training semantics: FKC2 also adds the active-correction reward
    # term. It only affects reported reward_per_step here, but keeping the
    # eval cfg's reward set matched to training avoids a semantic mismatch.
    cfg.rewards["active_correction_joint_tracking"] = RewardTermCfg(
      func=mdp.active_correction_joint_tracking_exp,
      weight=1.5,
      params={
        "command_name": "motion",
        "std": 0.2,
        "activation_threshold_rad": 0.05,
      },
    )
  cfg.events.pop("push_robot", None)
  crowd_event = cfg.events[HUMAN_MOTION_EVENT_NAME]
  crowd_event.params["obstacle_free_probability"] = 0.0
  crowd_event.params["min_count"] = 0
  crowd_event.params["max_count"] = 0
  crowd_event.params["randomize_density"] = False
  primary_event = cfg.events[PRIMARY_HUMAN_EVENT_NAME]
  primary_event.params["min_initial_spawn_radius_m"] = args.min_radius_m
  primary_event.params["max_initial_spawn_radius_m"] = args.max_radius_m
  primary_event.params["min_intersection_delay_s"] = args.min_intercept_s
  primary_event.params["max_intersection_delay_s"] = args.max_intercept_s
  if args.encounter_preset != "standard":
    apply_encounter_preset(cfg, args.encounter_preset)
  primary_event.params["show_mesh"] = False
  primary_event.params["print_velocity"] = False
  crowd_event.params["show_mesh"] = False
  for sensor in cfg.scene.sensors or ():
    sensor.debug_vis = False
  for group in cfg.observations.values():
    group.enable_corruption = False
  cfg.observations["lidar"].terms["directional_range_rate"].params["noise_cfg"] = None

  raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task_id)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task_id)
  if runner_cls is None:
    raise RuntimeError("auxiliary avoidance task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)
  if not isinstance(policy, PerceptiveLidarActor):
    raise RuntimeError("joint-compliance diagnosis requires the auxiliary actor")
  command_adjustment_mode = bool(
    getattr(policy, "adjust_command_with_joint_prediction", False)
  )
  if not command_adjustment_mode and policy.avoidance_joint_action_residual_gain <= 0.0:
    raise RuntimeError("checkpoint config has no explicit joint residual path")
  # Signal 2's mechanism differs by checkpoint family: legacy checkpoints add
  # an explicit residual to the normalized action; co-adjust checkpoints
  # instead inject the correction into the command slice the tracking net
  # observes (residual gain pinned to 0 by registration for that family).
  mode = "command_adjustment" if command_adjustment_mode else "action_residual"
  # Checkpoint load restores the long-running training step counter. Reschedule
  # the human once against that clock so time-based trajectories are valid.
  observations, _ = env.reset()
  if int(observations["avoidance_teacher"].shape[-1]) != 31:
    raise RuntimeError("avoidance teacher group must provide 31 corrections")

  device = args.device
  num_envs = args.num_envs
  robot = raw_env.scene["robot"]
  human = raw_env.scene[PRIMARY_HUMAN_ENTITY_NAME]
  human_geom_ids = human.indexing.geom_ids.to(dtype=torch.long)
  command = raw_env.command_manager.get_term("motion")
  link_names = tuple(command.cfg.link_filter.body_names)
  link_radius_m = float(command.cfg.link_filter.link_radius_m)

  side_link_names = {
    side: tuple(
      name
      for name in link_names
      if name.startswith(f"{side}_")
      and any(token in name for token in ARM_JOINT_TOKENS)
    )
    for side in SIDES
  }
  for side in SIDES:
    if len(side_link_names[side]) != 3:
      raise RuntimeError(
        f"expected 3 filtered {side} arm links, got {side_link_names[side]}"
      )
  arm_link_names = side_link_names["left"] + side_link_names["right"]
  arm_link_ids, resolved_names = robot.find_bodies(arm_link_names, preserve_order=True)
  if tuple(resolved_names) != arm_link_names:
    raise RuntimeError("robot arm link order mismatch")
  arm_link_ids_t = torch.tensor(arm_link_ids, dtype=torch.long, device=device)

  joint_names = tuple(robot.joint_names)
  joint_count = len(joint_names)
  side_joint_masks = {
    side: torch.tensor(
      [
        name.startswith(f"{side}_") and any(token in name for token in ARM_JOINT_TOKENS)
        for name in joint_names
      ],
      dtype=torch.bool,
      device=device,
    )
    for side in SIDES
  }
  for side in SIDES:
    if int(side_joint_masks[side].count_nonzero()) != 7:
      raise RuntimeError(f"expected 7 {side} arm joints in the action order")

  sim_model = raw_env.sim.model

  def arm_clearances() -> dict[str, torch.Tensor]:
    clearances = mdp.capsule_link_surface_clearances(
      robot.data.body_link_pos_w[:, arm_link_ids_t],
      human.data.geom_pos_w,
      human.data.geom_quat_w,
      sim_model.geom_size[:, human_geom_ids],
      link_radius_m=link_radius_m,
    ).amin(dim=-1)
    return {
      "left": clearances[:, :3].amin(dim=-1),
      "right": clearances[:, 3:].amin(dim=-1),
    }

  def zero() -> torch.Tensor:
    return torch.zeros((), device=device)

  accumulators = {
    side: {
      "active_frames": zero(),
      "prediction_mae": zero(),
      "prediction_cosine": zero(),
      "added_norm_l2": zero(),
      "injected_norm_l2": zero(),
      "state_t_sum": zero(),
      "state_t_ge_half_frames": zero(),
      "state_dist_to_filtered_l2": zero(),
      "state_dist_to_raw_l2": zero(),
      "state_stable_frames": zero(),
      "state_stable_t_sum": zero(),
      "state_stable_t_ge_half_frames": zero(),
      "state_stable_dist_to_filtered_l2": zero(),
      "state_stable_dist_to_raw_l2": zero(),
      **{
        f"h{h}_{name}": zero()
        for h in HORIZON_STEPS
        for name in (
          "valid_windows",
          "cosine_vs_teacher",
          "cosine_vs_pred",
          "projection_on_teacher",
          "dq_l2",
        )
      },
      "clearance_gated": zero(),
      "clearance_improved": zero(),
    }
    for side in SIDES
  }
  planar_frames = zero()
  planar_cosine_sum = zero()
  planar_state_active_frames = zero()
  planar_state_escape_ratio_sum = zero()
  planar_state_escape_ratio_ge_half_frames = zero()
  planar_state_cos_escape_sum = zero()
  planar_state_tracking_err_filt_sum = zero()
  planar_state_tracking_err_raw_sum = zero()

  gain = float(policy.avoidance_joint_action_residual_gain)
  action_mask = policy.avoidance_joint_action_mask
  action_scales = policy.avoidance_joint_action_scales
  max_added_deviation: float | None = 0.0 if mode == "action_residual" else None

  q_ring = torch.zeros((RING_LENGTH, num_envs, joint_count), device=device)
  teacher_ring = torch.zeros_like(q_ring)
  pred_ring = torch.zeros_like(q_ring)
  active_ring = {
    side: torch.zeros((RING_LENGTH, num_envs), dtype=torch.bool, device=device)
    for side in SIDES
  }
  clearance_ring = {
    side: torch.zeros((RING_LENGTH, num_envs), device=device) for side in SIDES
  }
  done_count_ring = torch.zeros(
    (RING_LENGTH, num_envs), dtype=torch.long, device=device
  )
  done_count = torch.zeros(num_envs, dtype=torch.long, device=device)

  with torch.inference_mode():
    for step in range(args.total_steps):
      teacher = observations["avoidance_teacher"]
      teacher_joints = teacher[:, 2:]
      prediction = policy.predict_avoidance(observations)
      pred_joints = prediction[:, 2:]
      joint_pos = robot.data.joint_pos.clone()
      raw_joint_pos = command._raw_joint_pos()  # noqa: SLF001
      clearance = arm_clearances()

      added_residual = gain * action_mask * pred_joints / action_scales
      if mode == "action_residual" and step == 0:
        reference = policy._joint_action_residual(prediction)  # noqa: SLF001
        max_added_deviation = float((added_residual - reference).abs().max())

      teacher_planar = teacher[:, :2]
      planar_active = (
        torch.linalg.vector_norm(teacher_planar, dim=-1) >= PLANAR_ACTIVE_MPS
      )
      planar_frames += planar_active.count_nonzero()
      planar_cosine_sum += _masked_sum(
        _cosine(prediction[:, :2], teacher_planar), planar_active
      )

      # Signal 6: planar (root) state compliance. On frames where the CBF
      # planar-velocity filter meaningfully diverges from the nominal
      # reference, how much of that commanded escape (relative to the raw
      # reference) does the robot's actual root velocity realize?
      v_filt = command.filtered_root_velocity_xy_w
      v_raw = command._raw_body_lin_vel_w()[:, 0, :2]  # noqa: SLF001
      v_robot = robot.data.root_link_lin_vel_w[:, :2]
      d = v_filt - v_raw
      planar_state_active = torch.linalg.vector_norm(d, dim=-1) >= (
        PLANAR_STATE_ACTIVE_MPS
      )
      planar_state_active_frames += planar_state_active.count_nonzero()
      escape_vec = v_robot - v_raw
      escape_ratio = (escape_vec * d).sum(dim=-1) / d.square().sum(dim=-1).clamp_min(
        1.0e-8
      )
      planar_state_escape_ratio_sum += _masked_sum(escape_ratio, planar_state_active)
      planar_state_escape_ratio_ge_half_frames += _masked_sum(
        (escape_ratio >= 0.5).to(escape_ratio.dtype), planar_state_active
      )
      planar_state_cos_escape_sum += _masked_sum(
        _cosine(escape_vec, d), planar_state_active
      )
      planar_state_tracking_err_filt_sum += _masked_sum(
        torch.linalg.vector_norm(v_robot - v_filt, dim=-1), planar_state_active
      )
      planar_state_tracking_err_raw_sum += _masked_sum(
        torch.linalg.vector_norm(v_robot - v_raw, dim=-1), planar_state_active
      )

      # Signal 5b support: a side's teacher joint residual counts as STABLE
      # when it has not moved much over the previous STABLE_WINDOW_STEPS
      # steps, i.e. no consecutive-step change within that window exceeds
      # STABLE_THRESHOLD_RAD. This isolates genuine tracking/state compliance
      # from apparent noncompliance caused by the mimic simply lagging a
      # correction that is still actively ramping. Reuses teacher_ring (the
      # same ring-buffer/done-masking pattern as the horizon windows above)
      # so a stability window never spans an environment reset.
      if step >= STABLE_WINDOW_STEPS:
        stable_base = step - STABLE_WINDOW_STEPS
        stable_window_valid = done_count == done_count_ring[stable_base % RING_LENGTH]
        window_samples = [teacher_joints] + [
          teacher_ring[(step - lag) % RING_LENGTH]
          for lag in range(1, STABLE_WINDOW_STEPS + 1)
        ]
        max_abs_change = torch.stack(
          [
            (window_samples[i] - window_samples[i + 1]).abs()
            for i in range(STABLE_WINDOW_STEPS)
          ],
          dim=0,
        ).amax(dim=0)
        stable_signal = {
          side: (
            max_abs_change[:, side_joint_masks[side]].amax(dim=-1)
            < STABLE_THRESHOLD_RAD
          )
          & stable_window_valid
          for side in SIDES
        }
      else:
        stable_signal = {
          side: torch.zeros(num_envs, dtype=torch.bool, device=device) for side in SIDES
        }

      active = {}
      for side in SIDES:
        joint_mask = side_joint_masks[side]
        teacher_side = teacher_joints[:, joint_mask]
        pred_side = pred_joints[:, joint_mask]
        side_active = teacher_side.abs().amax(dim=-1) >= args.active_threshold_rad
        active[side] = side_active
        acc = accumulators[side]
        acc["active_frames"] += side_active.count_nonzero()
        acc["prediction_mae"] += _masked_sum(
          (pred_side - teacher_side).abs().mean(dim=-1), side_active
        )
        acc["prediction_cosine"] += _masked_sum(
          _cosine(pred_side, teacher_side), side_active
        )
        acc["added_norm_l2"] += _masked_sum(
          torch.linalg.vector_norm(added_residual[:, joint_mask], dim=-1),
          side_active,
        )
        acc["injected_norm_l2"] += _masked_sum(
          torch.linalg.vector_norm(pred_side, dim=-1),
          side_active,
        )

        # Signal 5: where does the robot's arm currently sit along the
        # raw->filtered reference segment? t=0 is at the raw target, t=1 is
        # at the filtered (teacher-corrected) target. This is a STATE
        # comparison, not a motion comparison, so it stays meaningful even
        # when the policy already sits at the corrected pose and has no
        # further reason to move.
        q_robot_side = joint_pos[:, joint_mask]
        q_raw_side = raw_joint_pos[:, joint_mask]
        q_filtered_side = q_raw_side + teacher_side
        state_t = ((q_robot_side - q_raw_side) * teacher_side).sum(dim=-1) / (
          teacher_side.square().sum(dim=-1).clamp_min(1.0e-8)
        )
        acc["state_t_sum"] += _masked_sum(state_t, side_active)
        acc["state_t_ge_half_frames"] += _masked_sum(
          (state_t >= 0.5).to(state_t.dtype), side_active
        )
        acc["state_dist_to_filtered_l2"] += _masked_sum(
          torch.linalg.vector_norm(q_robot_side - q_filtered_side, dim=-1),
          side_active,
        )
        acc["state_dist_to_raw_l2"] += _masked_sum(
          torch.linalg.vector_norm(q_robot_side - q_raw_side, dim=-1),
          side_active,
        )

        # Signal 5b: the same state-compliance quantities, restricted to
        # active frames where the teacher correction has been stable (see
        # stable_signal above), lag-isolating true tracking compliance.
        stable_active = side_active & stable_signal[side]
        acc["state_stable_frames"] += stable_active.count_nonzero()
        acc["state_stable_t_sum"] += _masked_sum(state_t, stable_active)
        acc["state_stable_t_ge_half_frames"] += _masked_sum(
          (state_t >= 0.5).to(state_t.dtype), stable_active
        )
        acc["state_stable_dist_to_filtered_l2"] += _masked_sum(
          torch.linalg.vector_norm(q_robot_side - q_filtered_side, dim=-1),
          stable_active,
        )
        acc["state_stable_dist_to_raw_l2"] += _masked_sum(
          torch.linalg.vector_norm(q_robot_side - q_raw_side, dim=-1),
          stable_active,
        )

      for horizon in HORIZON_STEPS:
        base = step - horizon
        if base < 0:
          continue
        slot = base % RING_LENGTH
        window_valid = done_count == done_count_ring[slot]
        dq = joint_pos - q_ring[slot]
        for side in SIDES:
          joint_mask = side_joint_masks[side]
          selected = active_ring[side][slot] & window_valid
          dq_side = dq[:, joint_mask]
          teacher_base = teacher_ring[slot][:, joint_mask]
          pred_base = pred_ring[slot][:, joint_mask]
          teacher_direction = teacher_base / torch.linalg.vector_norm(
            teacher_base, dim=-1, keepdim=True
          ).clamp_min(1.0e-8)
          acc = accumulators[side]
          acc[f"h{horizon}_valid_windows"] += selected.count_nonzero()
          acc[f"h{horizon}_cosine_vs_teacher"] += _masked_sum(
            _cosine(dq_side, teacher_base), selected
          )
          acc[f"h{horizon}_cosine_vs_pred"] += _masked_sum(
            _cosine(dq_side, pred_base), selected
          )
          acc[f"h{horizon}_projection_on_teacher"] += _masked_sum(
            (dq_side * teacher_direction).sum(dim=-1), selected
          )
          acc[f"h{horizon}_dq_l2"] += _masked_sum(
            torch.linalg.vector_norm(dq_side, dim=-1), selected
          )
          if horizon == max(HORIZON_STEPS):
            gated = selected & (clearance_ring[side][slot] < CLEARANCE_GATE_M)
            improved = gated & (clearance[side] - clearance_ring[side][slot] > 0.0)
            acc["clearance_gated"] += gated.count_nonzero()
            acc["clearance_improved"] += improved.count_nonzero()

      slot = step % RING_LENGTH
      q_ring[slot] = joint_pos
      teacher_ring[slot] = teacher_joints
      pred_ring[slot] = pred_joints
      done_count_ring[slot] = done_count
      for side in SIDES:
        active_ring[side][slot] = active[side]
        clearance_ring[side][slot] = clearance[side]

      if args.oracle_injection:
        actions = policy.action_with_avoidance_override(observations, teacher)
      else:
        actions = policy(observations)
      observations, _, dones, _ = env.step(actions)
      done_count = done_count + dones.bool().to(torch.long)

  total_frames = args.total_steps * num_envs
  step_dt = float(raw_env.step_dt)

  def side_summary(side: str) -> dict:
    acc = accumulators[side]
    active_frames = int(acc["active_frames"])
    execution = {}
    for horizon in HORIZON_STEPS:
      valid = acc[f"h{horizon}_valid_windows"]
      execution[f"h{horizon}"] = {
        "seconds": horizon * step_dt,
        "valid_windows": int(valid),
        "cosine_vs_teacher": _mean(acc[f"h{horizon}_cosine_vs_teacher"], valid),
        "cosine_vs_pred": _mean(acc[f"h{horizon}_cosine_vs_pred"], valid),
        "projection_on_teacher_rad": _mean(
          acc[f"h{horizon}_projection_on_teacher"], valid
        ),
        "dq_l2_rad": _mean(acc[f"h{horizon}_dq_l2"], valid),
      }
    return {
      "frames": total_frames,
      "active_frames": active_frames,
      "active_rate": active_frames / total_frames,
      "prediction": {
        "mae_rad": _mean(acc["prediction_mae"], acc["active_frames"]),
        "cosine": _mean(acc["prediction_cosine"], acc["active_frames"]),
      },
      "mechanical": (
        {"added_norm_l2": _mean(acc["added_norm_l2"], acc["active_frames"])}
        if mode == "action_residual"
        else {
          "injected_correction_norm_l2": _mean(
            acc["injected_norm_l2"], acc["active_frames"]
          ),
        }
      ),
      "execution": execution,
      "clearance": {
        "gated_frames": int(acc["clearance_gated"]),
        "improved_fraction": _mean(acc["clearance_improved"], acc["clearance_gated"]),
      },
      "state_compliance": {
        "mean_t": _mean(acc["state_t_sum"], acc["active_frames"]),
        "fraction_t_ge_half": _mean(
          acc["state_t_ge_half_frames"], acc["active_frames"]
        ),
        "mean_dist_to_filtered_rad": _mean(
          acc["state_dist_to_filtered_l2"], acc["active_frames"]
        ),
        "mean_dist_to_raw_rad": _mean(
          acc["state_dist_to_raw_l2"], acc["active_frames"]
        ),
      },
      "state_compliance_stable": {
        "stable_frames": int(acc["state_stable_frames"]),
        "mean_t": _mean(acc["state_stable_t_sum"], acc["state_stable_frames"]),
        "fraction_t_ge_half": _mean(
          acc["state_stable_t_ge_half_frames"], acc["state_stable_frames"]
        ),
        "mean_dist_to_filtered_rad": _mean(
          acc["state_stable_dist_to_filtered_l2"], acc["state_stable_frames"]
        ),
        "mean_dist_to_raw_rad": _mean(
          acc["state_stable_dist_to_raw_l2"], acc["state_stable_frames"]
        ),
      },
    }

  def pooled_summary() -> dict:
    def total(name: str) -> torch.Tensor:
      return sum(accumulators[side][name] for side in SIDES)

    active_frames = int(total("active_frames"))
    execution = {}
    for horizon in HORIZON_STEPS:
      valid = total(f"h{horizon}_valid_windows")
      execution[f"h{horizon}"] = {
        "seconds": horizon * step_dt,
        "valid_windows": int(valid),
        "cosine_vs_teacher": _mean(total(f"h{horizon}_cosine_vs_teacher"), valid),
        "cosine_vs_pred": _mean(total(f"h{horizon}_cosine_vs_pred"), valid),
        "projection_on_teacher_rad": _mean(
          total(f"h{horizon}_projection_on_teacher"), valid
        ),
        "dq_l2_rad": _mean(total(f"h{horizon}_dq_l2"), valid),
      }
    return {
      "active_frames": active_frames,
      "active_rate": active_frames / (2 * total_frames),
      "prediction": {
        "mae_rad": _mean(total("prediction_mae"), total("active_frames")),
        "cosine": _mean(total("prediction_cosine"), total("active_frames")),
      },
      "mechanical": (
        {"added_norm_l2": _mean(total("added_norm_l2"), total("active_frames"))}
        if mode == "action_residual"
        else {
          "injected_correction_norm_l2": _mean(
            total("injected_norm_l2"), total("active_frames")
          ),
        }
      ),
      "execution": execution,
      "clearance": {
        "gated_frames": int(total("clearance_gated")),
        "improved_fraction": _mean(
          total("clearance_improved"), total("clearance_gated")
        ),
      },
      "state_compliance": {
        "mean_t": _mean(total("state_t_sum"), active_frames),
        "fraction_t_ge_half": _mean(total("state_t_ge_half_frames"), active_frames),
        "mean_dist_to_filtered_rad": _mean(
          total("state_dist_to_filtered_l2"), active_frames
        ),
        "mean_dist_to_raw_rad": _mean(total("state_dist_to_raw_l2"), active_frames),
      },
      "state_compliance_stable": {
        "stable_frames": int(total("state_stable_frames")),
        "mean_t": _mean(total("state_stable_t_sum"), total("state_stable_frames")),
        "fraction_t_ge_half": _mean(
          total("state_stable_t_ge_half_frames"), total("state_stable_frames")
        ),
        "mean_dist_to_filtered_rad": _mean(
          total("state_stable_dist_to_filtered_l2"), total("state_stable_frames")
        ),
        "mean_dist_to_raw_rad": _mean(
          total("state_stable_dist_to_raw_l2"), total("state_stable_frames")
        ),
      },
    }

  payload = {
    "checkpoint": str(args.checkpoint.resolve()),
    "motion_file": str(args.motion_file.resolve()),
    "seed": args.seed,
    "device": args.device,
    "configuration": {
      "task_id": args.task_id,
      "num_envs": num_envs,
      "min_radius_m": args.min_radius_m,
      "max_radius_m": args.max_radius_m,
      "min_intercept_s": args.min_intercept_s,
      "max_intercept_s": args.max_intercept_s,
      "episode_length_s": cfg.episode_length_s,
      "encounter_preset": args.encounter_preset,
      "total_steps": args.total_steps,
      "step_dt": step_dt,
      "active_threshold_rad": args.active_threshold_rad,
      "clearance_gate_m": CLEARANCE_GATE_M,
      "horizon_steps": list(HORIZON_STEPS),
      "human_runtime": "online",
      "crowd_count": 0,
      "expose_filtered_command": args.expose_filtered_command,
      "oracle_injection": args.oracle_injection,
      "sampling_mode": "uniform" if args.random_start else "start",
    },
    "residual_path": {
      "mode": mode,
      "gain": gain,
      "max_added_residual_deviation": max_added_deviation,
    },
    "sides": {side: side_summary(side) for side in SIDES},
    "pooled": pooled_summary(),
    "planar": {
      "frames": int(planar_frames),
      "mean_cosine": _mean(planar_cosine_sum, planar_frames),
    },
    "planar_state_compliance": {
      "active_frames": int(planar_state_active_frames),
      "active_rate": int(planar_state_active_frames) / total_frames,
      "escape_ratio_mean": _mean(
        planar_state_escape_ratio_sum, planar_state_active_frames
      ),
      "fraction_escape_ratio_ge_half": _mean(
        planar_state_escape_ratio_ge_half_frames, planar_state_active_frames
      ),
      "cos_escape_mean": _mean(planar_state_cos_escape_sum, planar_state_active_frames),
      "tracking_err_filt_mean_mps": _mean(
        planar_state_tracking_err_filt_sum, planar_state_active_frames
      ),
      "tracking_err_raw_mean_mps": _mean(
        planar_state_tracking_err_raw_sum, planar_state_active_frames
      ),
    },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

  def fmt(value: float | None, precision: int = 3) -> str:
    return "n/a" if value is None else f"{value:+.{precision}f}"

  print("\nSignal 1: teacher vs predicted joint residual (active frames)")
  for side in SIDES:
    summary = payload["sides"][side]
    print(
      f"  {side:5s} active={summary['active_frames']:8d} "
      f"({100.0 * summary['active_rate']:5.1f}%)  "
      f"mae={fmt(summary['prediction']['mae_rad'])} rad  "
      f"cos={fmt(summary['prediction']['cosine'])}"
    )
  if mode == "action_residual":
    print("Signal 2: residual actually added to the normalized action")
    print(f"  gain={gain:.2f}  max deviation from actor path={max_added_deviation:.2e}")
    for side in SIDES:
      summary = payload["sides"][side]
      print(
        f"  {side:5s} mean added |residual|="
        f"{fmt(summary['mechanical']['added_norm_l2'])}"
      )
  else:
    print("Signal 2: command-slice injection (command_adjustment mode)")
    for side in SIDES:
      summary = payload["sides"][side]
      print(
        f"  {side:5s} mean injected correction |q|="
        f"{fmt(summary['mechanical']['injected_correction_norm_l2'])} rad"
      )
  print("Signal 3: achieved joint motion vs teacher/prediction")
  for side in SIDES:
    summary = payload["sides"][side]
    for horizon in HORIZON_STEPS:
      row = summary["execution"][f"h{horizon}"]
      print(
        f"  {side:5s} {row['seconds']:.1f}s windows={row['valid_windows']:8d} "
        f"cos_vs_teacher={fmt(row['cosine_vs_teacher'])} "
        f"cos_vs_pred={fmt(row['cosine_vs_pred'])} "
        f"proj={fmt(row['projection_on_teacher_rad'])} rad "
        f"|dq|={fmt(row['dq_l2_rad'])} rad"
      )
  print("Signal 4: clearance improvement on threatened active frames")
  for side in SIDES:
    summary = payload["sides"][side]
    print(
      f"  {side:5s} gated={summary['clearance']['gated_frames']:8d} "
      f"improved={fmt(summary['clearance']['improved_fraction'])}"
    )
  print("Signal 5: state compliance (position along raw->filtered segment)")
  for side in SIDES:
    summary = payload["sides"][side]
    sc = summary["state_compliance"]
    print(
      f"  {side:5s} t_mean={fmt(sc['mean_t'])} "
      f"frac(t>=0.5)={fmt(sc['fraction_t_ge_half'])}  "
      f"|q-q_filtered|={fmt(sc['mean_dist_to_filtered_rad'])} rad "
      f"|q-q_raw|={fmt(sc['mean_dist_to_raw_rad'])} rad"
    )
  pooled_sc = payload["pooled"]["state_compliance"]
  print(
    f"  pooled t_mean={fmt(pooled_sc['mean_t'])} "
    f"frac(t>=0.5)={fmt(pooled_sc['fraction_t_ge_half'])}  "
    f"|q-q_filtered|={fmt(pooled_sc['mean_dist_to_filtered_rad'])} rad "
    f"|q-q_raw|={fmt(pooled_sc['mean_dist_to_raw_rad'])} rad"
  )
  print("Signal 5b — state compliance on stable-correction frames")
  for side in SIDES:
    summary = payload["sides"][side]
    scs = summary["state_compliance_stable"]
    print(
      f"  {side:5s} stable={scs['stable_frames']:8d} "
      f"t_mean={fmt(scs['mean_t'])} "
      f"frac(t>=0.5)={fmt(scs['fraction_t_ge_half'])}  "
      f"|q-q_filtered|={fmt(scs['mean_dist_to_filtered_rad'])} rad "
      f"|q-q_raw|={fmt(scs['mean_dist_to_raw_rad'])} rad"
    )
  pooled_scs = payload["pooled"]["state_compliance_stable"]
  print(
    f"  pooled stable={pooled_scs['stable_frames']:8d} "
    f"t_mean={fmt(pooled_scs['mean_t'])} "
    f"frac(t>=0.5)={fmt(pooled_scs['fraction_t_ge_half'])}  "
    f"|q-q_filtered|={fmt(pooled_scs['mean_dist_to_filtered_rad'])} rad "
    f"|q-q_raw|={fmt(pooled_scs['mean_dist_to_raw_rad'])} rad"
  )
  planar = payload["planar"]
  print(f"Planar compass: frames={planar['frames']} cos={fmt(planar['mean_cosine'])}")
  print("Signal 6 — planar (root) state compliance")
  psc = payload["planar_state_compliance"]
  print(
    f"  active={psc['active_frames']:8d} ({100.0 * psc['active_rate']:5.1f}%)  "
    f"escape_ratio={fmt(psc['escape_ratio_mean'])} "
    f"frac(>=0.5)={fmt(psc['fraction_escape_ratio_ge_half'])}  "
    f"cos_escape={fmt(psc['cos_escape_mean'])}  "
    f"|v-v_filt|={fmt(psc['tracking_err_filt_mean_mps'])} m/s "
    f"|v-v_raw|={fmt(psc['tracking_err_raw_mean_mps'])} m/s"
  )
  print(f"\nWROTE {args.output}")

  env.close()
  del policy, runner, env, raw_env
  gc.collect()
  if args.device.startswith("cuda"):
    torch.cuda.empty_cache()


if __name__ == "__main__":
  main()
