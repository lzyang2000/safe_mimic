"""Measure collision probability over human distance, speed, and intercept TTC."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from safe_mimic.evaluation import (
  ENCOUNTER_PRESETS,
  apply_encounter_preset,
  bearing_deg,
  fair_regime_by_region,
  fair_regime_summary,
)
from safe_mimic.tasks import (
  LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
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

# Fair regime: encounters a ~0.6 m/s robot can physically win.
FAIR_REGIME_MAX_SPEED_MPS = 0.75
FAIR_REGIME_MIN_CLEARANCE_M = 0.8


def _bin_summary(
  values: torch.Tensor,
  collision: torch.Tensor,
  edges: tuple[float, ...],
  *,
  precision: int = 2,
) -> list[dict[str, float | int | str]]:
  rows: list[dict[str, float | int | str]] = []
  for lower, upper in zip(edges[:-1], edges[1:], strict=True):
    selected = torch.isfinite(values) & (values >= lower) & (values < upper)
    episodes = int(selected.count_nonzero())
    collisions = int((selected & collision).count_nonzero())
    upper_text = f"{upper:.{precision}f}" if math.isfinite(upper) else "inf"
    rows.append(
      {
        "range": f"[{lower:.{precision}f}, {upper_text})",
        "episodes": episodes,
        "collisions": collisions,
        "collision_rate": collisions / episodes if episodes else float("nan"),
      }
    )
  return rows


def _grid_summary(
  distance: torch.Tensor,
  speed: torch.Tensor,
  collision: torch.Tensor,
  distance_edges: tuple[float, ...],
  speed_edges: tuple[float, ...],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for distance_lower, distance_upper in zip(
    distance_edges[:-1], distance_edges[1:], strict=True
  ):
    distance_selected = (distance >= distance_lower) & (distance < distance_upper)
    cells = []
    for speed_lower, speed_upper in zip(speed_edges[:-1], speed_edges[1:], strict=True):
      selected = distance_selected & (speed >= speed_lower) & (speed < speed_upper)
      episodes = int(selected.count_nonzero())
      collisions = int((selected & collision).count_nonzero())
      cells.append(
        {
          "speed_range_mps": f"[{speed_lower:.2f}, {speed_upper:.2f})",
          "episodes": episodes,
          "collisions": collisions,
          "collision_rate": collisions / episodes if episodes else float("nan"),
        }
      )
    rows.append(
      {
        "distance_range_m": f"[{distance_lower:.2f}, {distance_upper:.2f})",
        "speed_cells": cells,
      }
    )
  return rows


def _print_table(title: str, rows: list[dict[str, float | int | str]]) -> None:
  print(f"\n{title}")
  print("range                 episodes  collisions  collision_rate")
  for row in rows:
    rate = float(row["collision_rate"])
    rate_text = f"{100.0 * rate:6.1f}%" if math.isfinite(rate) else "   n/a "
    print(
      f"{str(row['range']):<21} {int(row['episodes']):8d} "
      f"{int(row['collisions']):11d}  {rate_text}"
    )


def _print_termination_causes(causes: dict[str, int], completed: int) -> None:
  print("\nTermination causes")
  print("term                          count  share_of_completed")
  for name, count in causes.items():
    share = count / completed if completed else float("nan")
    share_text = f"{100.0 * share:6.1f}%" if math.isfinite(share) else "   n/a "
    print(f"{name:<29} {count:6d}  {share_text}")


def _minimum_clearance(
  raw_env: ManagerBasedRlEnv,
  *,
  robot_link_ids: torch.Tensor,
  human_geom_ids: torch.Tensor,
  link_radius_m: float,
) -> torch.Tensor:
  robot = raw_env.scene["robot"]
  human = raw_env.scene[PRIMARY_HUMAN_ENTITY_NAME]
  clearances = mdp.capsule_link_surface_clearances(
    robot.data.body_link_pos_w[:, robot_link_ids],
    human.data.geom_pos_w,
    human.data.geom_quat_w,
    raw_env.sim.model.geom_size[:, human_geom_ids],
    link_radius_m=link_radius_m,
  )
  return clearances.amin(dim=(-2, -1))


def _blind_action(policy: Any, observations: Any) -> torch.Tensor:
  lidar = observations["lidar"]
  directional = lidar[..., :-1]
  saved = directional.clone()
  cell_count = directional.shape[-1] // 2
  directional[..., :cell_count] = 1.0
  directional[..., cell_count:] = 0.0
  try:
    return policy(observations)
  finally:
    directional.copy_(saved)


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
    ),
    default=LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
    help="rl-cfg/runner task id used to load the checkpoint's actor config",
  )
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--seed", type=int, default=31)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--min-radius-m", type=float, default=0.75)
  parser.add_argument("--max-radius-m", type=float, default=4.0)
  parser.add_argument("--min-intercept-s", type=float, default=0.5)
  parser.add_argument("--max-intercept-s", type=float, default=4.0)
  parser.add_argument("--episode-length-s", type=float, default=6.0)
  parser.add_argument("--action-reaction-threshold", type=float, default=0.10)
  parser.add_argument(
    "--expose-filtered-command",
    action="store_true",
    help=(
      "Expose the filter-adjusted joint targets in the actor's command "
      "observation (tests reference-adjustment obedience)."
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
  else:
    cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=True)
  # Pin frame-0 starts for comparability with existing artifacts.
  cfg.commands["motion"].sampling_mode = "uniform" if args.random_start else "start"
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.episode_length_s = args.episode_length_s
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
  primary_event.params["encounter_sampling"] = "independent"
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
  # Checkpoint load restores the long-running training step counter. Reschedule
  # the human once against that clock so the initial intercept TTC is valid.
  observations, _ = env.reset()

  robot = raw_env.scene["robot"]
  human = raw_env.scene[PRIMARY_HUMAN_ENTITY_NAME]
  command = raw_env.command_manager.get_term("motion")
  link_names = tuple(command.cfg.link_filter.body_names)
  link_ids, resolved_names = robot.find_bodies(link_names, preserve_order=True)
  if tuple(resolved_names) != link_names:
    raise RuntimeError("robot link order mismatch")
  link_ids_t = torch.tensor(link_ids, dtype=torch.long, device=args.device)
  human_geom_ids = human.indexing.geom_ids.to(dtype=torch.long)
  event_term = raw_env.event_manager.get_term_cfg(PRIMARY_HUMAN_EVENT_NAME).func
  if not hasattr(event_term, "sampler"):
    raise RuntimeError("primary human event does not expose its sampler")
  sampler = event_term.sampler

  now_s = float(raw_env.common_step_counter) * raw_env.step_dt
  scheduled_ttc_s = sampler.global_intersection_times_s.clone() - now_s
  initial_human_root_w = sampler._poses.root_positions_w.clone()  # noqa: SLF001
  initial_robot_root_w = robot.data.root_link_pos_w.clone()
  initial_root_distance_m = torch.linalg.vector_norm(
    initial_human_root_w[:, :2] - initial_robot_root_w[:, :2], dim=-1
  )
  # Where the human comes from, in the robot's heading frame (0 = ahead,
  # +90 = robot's left, +-180 = behind). Tracked again at minimum clearance.
  initial_bearing_deg = bearing_deg(
    initial_robot_root_w[:, :2],
    robot.data.root_link_quat_w,
    initial_human_root_w[:, :2],
  )
  bearing_at_min_clearance_deg = initial_bearing_deg.clone()
  initial_clearance_m = _minimum_clearance(
    raw_env,
    robot_link_ids=link_ids_t,
    human_geom_ids=human_geom_ids,
    link_radius_m=command.cfg.link_filter.link_radius_m,
  )
  nominal_approach_speed_mps = initial_root_distance_m / scheduled_ttc_s.clamp_min(
    raw_env.step_dt
  )

  num_envs = args.num_envs
  active = torch.ones(num_envs, dtype=torch.bool, device=args.device)
  causes = {
    name: torch.zeros(num_envs, dtype=torch.bool, device=args.device)
    for name in raw_env.termination_manager.active_terms
  }
  collision_time_s = torch.full((num_envs,), torch.nan, device=args.device)
  minimum_clearance_m = initial_clearance_m.clone()
  reaction_time_s = torch.full((num_envs,), torch.nan, device=args.device)
  weak_reaction_time_s = torch.full((num_envs,), torch.nan, device=args.device)
  maximum_action_delta = torch.zeros(num_envs, device=args.device)
  first_outward_motion_time_s = torch.full((num_envs,), torch.nan, device=args.device)
  outward_speed_threshold_mps = 0.25

  with torch.inference_mode():
    for step in range(raw_env.max_episode_length):
      if not torch.any(active):
        break
      elapsed_s = step * raw_env.step_dt
      active_before = active.clone()
      step_clearance = _minimum_clearance(
        raw_env,
        robot_link_ids=link_ids_t,
        human_geom_ids=human_geom_ids,
        link_radius_m=command.cfg.link_filter.link_radius_m,
      )
      new_minimum = active_before & (step_clearance < minimum_clearance_m)
      minimum_clearance_m = torch.where(
        new_minimum, step_clearance, minimum_clearance_m
      )
      step_bearing = bearing_deg(
        robot.data.root_link_pos_w[:, :2],
        robot.data.root_link_quat_w,
        sampler._poses.root_positions_w[:, :2],  # noqa: SLF001
      )
      bearing_at_min_clearance_deg = torch.where(
        new_minimum, step_bearing, bearing_at_min_clearance_deg
      )

      actions = policy(observations)
      blind_actions = _blind_action(policy, observations)
      action_delta = torch.linalg.vector_norm(actions - blind_actions, dim=-1)
      maximum_action_delta = torch.where(
        active_before,
        torch.maximum(maximum_action_delta, action_delta),
        maximum_action_delta,
      )
      weak_reaction = (
        active_before & torch.isnan(weak_reaction_time_s) & (action_delta >= 0.05)
      )
      reaction = (
        active_before
        & torch.isnan(reaction_time_s)
        & (action_delta >= args.action_reaction_threshold)
      )
      weak_reaction_time_s[weak_reaction] = elapsed_s
      reaction_time_s[reaction] = elapsed_s

      human_root_w = sampler._poses.root_positions_w  # noqa: SLF001
      away_xy = robot.data.root_link_pos_w[:, :2] - human_root_w[:, :2]
      away_xy = away_xy / torch.linalg.vector_norm(
        away_xy, dim=-1, keepdim=True
      ).clamp_min(1.0e-6)
      outward_speed = torch.sum(robot.data.root_link_lin_vel_w[:, :2] * away_xy, dim=-1)
      outward = (
        active_before
        & torch.isnan(first_outward_motion_time_s)
        & (outward_speed >= outward_speed_threshold_mps)
      )
      first_outward_motion_time_s[outward] = elapsed_s

      observations, _, dones, _ = env.step(actions)
      new_done = active_before & dones.bool()
      if torch.any(new_done):
        for name in causes:
          causes[name][new_done] = raw_env.termination_manager.get_term(name)[new_done]
        collision_now = causes["primary_human_collision"] | causes["crowd_collision"]
        collision_time_s[new_done & collision_now] = elapsed_s + raw_env.step_dt
        active[new_done] = False

  collision = causes["primary_human_collision"] | causes["crowd_collision"]
  timeout = causes["time_out"]
  other_failure = ~active & ~collision & ~timeout

  tensors = {
    "scheduled_intercept_ttc_s": scheduled_ttc_s,
    "initial_root_distance_m": initial_root_distance_m,
    "initial_clearance_m": initial_clearance_m,
    "nominal_approach_speed_mps": nominal_approach_speed_mps,
    "collision": collision,
    "timeout": timeout,
    "other_failure": other_failure,
    "collision_time_s": collision_time_s,
    "minimum_clearance_m": minimum_clearance_m,
    "initial_bearing_deg": initial_bearing_deg,
    "bearing_at_min_clearance_deg": bearing_at_min_clearance_deg,
    "weak_reaction_time_s": weak_reaction_time_s,
    "reaction_time_s": reaction_time_s,
    "first_outward_motion_time_s": first_outward_motion_time_s,
    "maximum_action_delta": maximum_action_delta,
  }
  cpu = {name: tensor.cpu() for name, tensor in tensors.items()}
  collision_cpu = cpu["collision"]
  ttc_rows = _bin_summary(
    cpu["scheduled_intercept_ttc_s"],
    collision_cpu,
    (0.0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, float("inf")),
  )
  distance_rows = _bin_summary(
    cpu["initial_root_distance_m"],
    collision_cpu,
    (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, float("inf")),
  )
  speed_rows = _bin_summary(
    cpu["nominal_approach_speed_mps"],
    collision_cpu,
    (0.0, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, float("inf")),
  )
  clearance_rows = _bin_summary(
    cpu["initial_clearance_m"],
    collision_cpu,
    (-float("inf"), 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float("inf")),
  )
  distance_speed_grid = _grid_summary(
    cpu["initial_root_distance_m"],
    cpu["nominal_approach_speed_mps"],
    collision_cpu,
    (0.0, 1.5, 2.0, 2.5, 3.0, 3.5, float("inf")),
    (0.0, 0.75, 1.0, 1.5, 2.0, 3.0, float("inf")),
  )

  completed = ~active.cpu()
  causes_cpu = {name: tensor.cpu() for name, tensor in causes.items()}
  priority_terms = ("primary_human_collision", "crowd_collision")
  other_terms = sorted(
    name for name in causes_cpu if name not in priority_terms and name != "time_out"
  )
  cause_order = [*priority_terms, *other_terms, "time_out"]
  termination_cause: list[str | None] = [None] * num_envs
  unresolved = completed.clone()
  for name in cause_order:
    fired = causes_cpu[name] & unresolved
    for index in fired.nonzero(as_tuple=True)[0].tolist():
      termination_cause[index] = name
    unresolved &= ~fired
  termination_causes = {
    name: int((causes_cpu[name] & completed).count_nonzero()) for name in cause_order
  }
  reaction = cpu["reaction_time_s"]
  weak_reaction = cpu["weak_reaction_time_s"]
  outward = cpu["first_outward_motion_time_s"]

  def finite_mean(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if len(finite) else None

  def finite_quantiles(values: torch.Tensor) -> dict[str, float] | None:
    finite = values[torch.isfinite(values)]
    if not len(finite):
      return None
    quantiles = torch.quantile(finite, torch.tensor([0.1, 0.5, 0.9]))
    return {
      "p10": float(quantiles[0]),
      "p50": float(quantiles[1]),
      "p90": float(quantiles[2]),
    }

  cases_payload = [
    {
      **{
        name: bool(values[index])
        if values.dtype == torch.bool
        else None
        if not math.isfinite(float(values[index]))
        else float(values[index])
        for name, values in cpu.items()
      },
      "termination_cause": termination_cause[index],
    }
    for index in range(num_envs)
  ]
  summary = {
    "episodes": num_envs,
    "completed": int(completed.count_nonzero()),
    "collisions": int(collision_cpu.count_nonzero()),
    "collision_rate": float(collision_cpu.float().mean()),
    "timeouts": int(cpu["timeout"].count_nonzero()),
    "other_failures": int(cpu["other_failure"].count_nonzero()),
    "termination_causes": termination_causes,
    "mean_scheduled_ttc_s": float(cpu["scheduled_intercept_ttc_s"].mean()),
    "mean_initial_root_distance_m": float(cpu["initial_root_distance_m"].mean()),
    "mean_nominal_approach_speed_mps": float(cpu["nominal_approach_speed_mps"].mean()),
    "action_reaction_threshold_l2": args.action_reaction_threshold,
    "reaction_detected_rate": float(torch.isfinite(reaction).float().mean()),
    "reaction_time_s": finite_quantiles(reaction),
    "weak_reaction_time_s": finite_quantiles(weak_reaction),
    "outward_motion_threshold_mps": outward_speed_threshold_mps,
    "outward_motion_detected_rate": float(torch.isfinite(outward).float().mean()),
    "outward_motion_time_s": finite_quantiles(outward),
    "mean_maximum_normal_blind_action_l2": finite_mean(cpu["maximum_action_delta"]),
    "fair_regime": fair_regime_summary(
      cases_payload,
      max_speed_mps=FAIR_REGIME_MAX_SPEED_MPS,
      min_clearance_m=FAIR_REGIME_MIN_CLEARANCE_M,
    ),
    "fair_regime_by_initial_bearing": fair_regime_by_region(
      cases_payload,
      max_speed_mps=FAIR_REGIME_MAX_SPEED_MPS,
      min_clearance_m=FAIR_REGIME_MIN_CLEARANCE_M,
      bearing_key="initial_bearing_deg",
    ),
    "fair_regime_by_closest_bearing": fair_regime_by_region(
      cases_payload,
      max_speed_mps=FAIR_REGIME_MAX_SPEED_MPS,
      min_clearance_m=FAIR_REGIME_MIN_CLEARANCE_M,
      bearing_key="bearing_at_min_clearance_deg",
    ),
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
      "human_runtime": "online",
      "crowd_count": 0,
      "collision_clearance_m": 0.1,
      "expose_filtered_command": args.expose_filtered_command,
      "sampling_mode": "uniform" if args.random_start else "start",
    },
    "summary": summary,
    "scheduled_ttc_bins": ttc_rows,
    "initial_distance_bins": distance_rows,
    "nominal_speed_bins": speed_rows,
    "initial_clearance_bins": clearance_rows,
    "distance_speed_grid": distance_speed_grid,
    "cases": cases_payload,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True))
  _print_table("Scheduled intercept TTC", ttc_rows)
  _print_table("Initial root distance", distance_rows)
  _print_table("Nominal average approach speed", speed_rows)
  _print_table("Initial robot-link clearance", clearance_rows)
  _print_termination_causes(termination_causes, int(completed.count_nonzero()))
  fair = summary["fair_regime"]
  print(
    f"\nFair regime (speed <= {FAIR_REGIME_MAX_SPEED_MPS} m/s, initial clearance >= "
    f"{FAIR_REGIME_MIN_CLEARANCE_M} m): n={fair['episodes']} "
    f"safety_failure={fair['safety_failure_rate']:.3f} "
    f"(collision={fair['collision_rate']:.3f}) "
    f"tracking_termination={fair['tracking_termination_rate']:.3f} "
    f"any_failure={fair['failure_rate']:.3f} survival={fair['survival_rate']:.3f}"
  )
  for key, label in (
    ("fair_regime_by_initial_bearing", "initial bearing"),
    ("fair_regime_by_closest_bearing", "bearing at minimum clearance"),
  ):
    print(f"\nFair regime by {label}:")
    for region, row in summary[key].items():
      print(
        f"  {region:6s} n={row['episodes']:3d} failure={row['failure_rate']:.3f} "
        f"collision={row['collision_rate']:.3f} ee={row['ee_body_pos']}"
      )
  print(f"\nWROTE {args.output}")

  env.close()
  del policy, runner, env, raw_env
  gc.collect()
  if args.device.startswith("cuda"):
    torch.cuda.empty_cache()


if __name__ == "__main__":
  main()
