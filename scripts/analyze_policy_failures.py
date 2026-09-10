"""Diagnose 100-case LiDAR avoidance failures at perception and control level.

The analysis follows one finite episode in every vectorized environment. It
attributes raw ray hits to the primary human's capsules, checks whether those
hits survive directional minimum pooling into the actor observation, measures
approach speed, and compares the policy's arm target with the privileged
link-filter arm correction used during training.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from safe_mimic.evaluation import ENCOUNTER_PRESETS, apply_encounter_preset
from safe_mimic.sensing.held_scan import HeldScanRayCastSensor
from safe_mimic.sensing.observations import _directional_minimum_pool
from safe_mimic.tasks import (
  LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
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
  LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
  UNIFIED_ROOT_LEAD_M,
  mdp,
)
from safe_mimic.tasks.env_cfg import (
  DEFAULT_G1_BALLET_MANIFEST,
  DEFAULT_G1_BALLET_MIRROR_MANIFEST,
  HUMAN_MOTION_EVENT_NAME,
  LIDAR_SENSOR_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
  unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)

BLIND_NOMINAL_TASK_ID = (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID
)
BLIND_NOHUMANS_TASK_ID = (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID
)
MOVES_TASK_ID = LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID


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


def _human_hit_mask(
  hit_positions_w: torch.Tensor,
  distances: torch.Tensor,
  capsule_centers_w: torch.Tensor,
  capsule_quaternions_w: torch.Tensor,
  capsule_sizes: torch.Tensor,
  *,
  surface_tolerance_m: float,
) -> torch.Tensor:
  """Identify rays whose hit point lies on a current primary-human capsule."""
  axes_w = _quaternion_z_axis(capsule_quaternions_w)
  segment_half_w = axes_w * capsule_sizes[..., 1, None]
  starts_w = capsule_centers_w - segment_half_w
  segments_w = 2.0 * segment_half_w
  human_hit = torch.zeros_like(distances, dtype=torch.bool)
  for capsule_index in range(capsule_centers_w.shape[1]):
    start_w = starts_w[:, capsule_index, None]
    segment_w = segments_w[:, capsule_index, None]
    relative_w = hit_positions_w - start_w
    denominator = segment_w.square().sum(dim=-1).clamp_min(1e-12)
    fraction = ((relative_w * segment_w).sum(dim=-1) / denominator).clamp(0.0, 1.0)
    closest_w = start_w + fraction[..., None] * segment_w
    centerline_distance = torch.linalg.vector_norm(hit_positions_w - closest_w, dim=-1)
    surface_error = torch.abs(
      centerline_distance - capsule_sizes[:, capsule_index, 0, None]
    )
    human_hit |= surface_error <= surface_tolerance_m
  return human_hit & (distances >= 0.0)


def _new_stats(num_envs: int, device: str) -> dict[str, torch.Tensor]:
  zeros = lambda dtype=torch.float32: torch.zeros(  # noqa: E731
    num_envs, dtype=dtype, device=device
  )
  return {
    "episode_steps": zeros(torch.long),
    "collision": zeros(torch.bool),
    "timeout": zeros(torch.bool),
    "other_failure": zeros(torch.bool),
    "minimum_clearance_m": torch.full((num_envs,), torch.inf, device=device),
    "minimum_clearance_link_id": torch.full(
      (num_envs,), -1, dtype=torch.long, device=device
    ),
    "raw_hit_publications": zeros(torch.long),
    "represented_publications": zeros(torch.long),
    "masked_only_publications": zeros(torch.long),
    "max_raw_human_hits": zeros(torch.long),
    "max_represented_cells": zeros(torch.long),
    "human_cell_sum": zeros(),
    "represented_cell_sum": zeros(),
    "last_cell_retention": torch.full((num_envs,), torch.nan, device=device),
    "last_raw_hit_step": torch.full(
      (num_envs,), -10_000, dtype=torch.long, device=device
    ),
    "last_represented_step": torch.full(
      (num_envs,), -10_000, dtype=torch.long, device=device
    ),
    "first_represented_distance_m": torch.full((num_envs,), torch.nan, device=device),
    "first_represented_closing_speed_mps": torch.full(
      (num_envs,), torch.nan, device=device
    ),
    "max_closing_speed_mps": torch.zeros(num_envs, device=device),
    "max_near_closing_speed_mps": torch.zeros(num_envs, device=device),
    "closing_speed_at_minimum_clearance_mps": torch.full(
      (num_envs,), torch.nan, device=device
    ),
    "risk_steps": zeros(torch.long),
    "arm_request_steps": zeros(torch.long),
    "max_arm_request_rad": zeros(),
    "target_projection_sum": zeros(),
    "actual_projection_sum": zeros(),
    "positive_target_response_steps": zeros(torch.long),
  }


def _configure_env(
  *,
  num_envs: int,
  seed: int,
  motion_file: Path,
  online_human: bool,
  task_id: str,
  encounter_preset: str = "standard",
) -> Any:
  if task_id == LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID:
    cfg = unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg(play=online_human)
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID:
    # The unified task's reward set and both filtered-reference flags must
    # match training exactly; build it from its own env-cfg builder instead
    # of the auxiliary builder plus the FKC patch in the branch below.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(play=online_human)
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID:
    # Same unified builder plus the active-joint reward term added on top of
    # it during training; must match exactly like the unified branch above.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human, active_joint_reward=True
    )
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID:
    # Unified-Joint plus the leashed root target and robot-evaluated CBF.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
    )
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID:
    # Leash plus slow-regime encounters and the filter-gated ee termination.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      slow_regime=True,
      filter_gated_ee_termination=True,
    )
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID:
    # Leash-Slow with dense encounters and the strict stock ee termination.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      slow_regime=True,
      dense_encounters=True,
    )
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID:
    # Leash (canonical setup) with the whole ballet library as the reference
    # (pass the manifest as --motion-file).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    )
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID:
    # Leash-Ballet plus the slow encounter regime (eval overrides the spawn
    # and TTC ranges anyway; the flag keeps the cfg identical to training).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      slow_regime=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    )
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID:
    # Leash-Ballet with the lag-aware ee_body_pos termination.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      lag_aware_ee_termination=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    )
  elif task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID:
    # Blind baseline: Leash-Ballet with the actor's LiDAR term reading "no returns".
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      blind_actor=True,
    )
  elif task_id == BLIND_NOMINAL_TASK_ID:
    # Blind + nominal baseline: blind actor AND both CBF filters off (raw reference).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      blind_actor=True,
      nominal_reference=True,
    )
  elif task_id == BLIND_NOHUMANS_TASK_ID:
    # Blind + nominal + no humans in training (play cfg keeps the populated scene).
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      blind_actor=True,
      nominal_reference=True,
      training_humans=False,
    )
  elif task_id == MOVES_TASK_ID:
    # Escape moves: mirrored ballet library, travelling steps as the escape.
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=online_human,
      active_joint_reward=True,
      root_lead_m=UNIFIED_ROOT_LEAD_M,
      planar_filter_at_robot_root=True,
      motion_manifest=str(DEFAULT_G1_BALLET_MIRROR_MANIFEST),
      escape_moves=True,
    )
  else:
    cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=online_human)
    if task_id == LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID:
      # Match training semantics: the FKC task's body position/orientation
      # targets follow the filtered joint corrections, so task-space metrics
      # reported here are consistent with what the checkpoint was trained on.
      cfg.commands["motion"].propagate_arm_corrections_to_body_targets = True
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 10.0
  cfg.commands["motion"].motion_file = str(motion_file)
  cfg.events.pop("push_robot", None)
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["encounter_sampling"] = "independent"
  # Pin frame-0 starts for comparability with existing artifacts.
  cfg.commands["motion"].sampling_mode = "start"
  # Pin the pre-TTC-sampling evaluation distribution for comparability with
  # historical artifacts (training defaults widened to 0.75-4.0 m / 0.5-4.0 s).
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["min_initial_spawn_radius_m"] = 2.0
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["max_initial_spawn_radius_m"] = 4.0
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["min_intersection_delay_s"] = 1.0
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["max_intersection_delay_s"] = 3.0
  if encounter_preset != "standard":
    apply_encounter_preset(cfg, encounter_preset)
  crowd_event = cfg.events[HUMAN_MOTION_EVENT_NAME]
  crowd_event.params["obstacle_free_probability"] = 0.0
  crowd_event.params["min_count"] = 0
  crowd_event.params["max_count"] = 0
  crowd_event.params["randomize_density"] = False
  if online_human:
    for event_name in (HUMAN_MOTION_EVENT_NAME, PRIMARY_HUMAN_EVENT_NAME):
      cfg.events[event_name].params["show_mesh"] = False
    cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["print_velocity"] = False
  for sensor in cfg.scene.sensors or ():
    sensor.debug_vis = False
  for group in cfg.observations.values():
    group.enable_corruption = False
  cfg.observations["lidar"].terms["directional_range_rate"].params["noise_cfg"] = None
  return cfg


def _float_or_none(value: torch.Tensor) -> float | None:
  scalar = float(value)
  return scalar if math.isfinite(scalar) else None


def _rate(mask: torch.Tensor, selected: torch.Tensor) -> float | None:
  count = int(selected.count_nonzero())
  return float((mask & selected).count_nonzero()) / count if count else None


def _speed_bin_summary(
  collision: torch.Tensor, speeds: torch.Tensor
) -> list[dict[str, float | int | None | str]]:
  result: list[dict[str, float | int | None | str]] = []

  def append(label: str, selected: torch.Tensor) -> None:
    count = int(selected.count_nonzero())
    result.append(
      {
        "range_mps": label,
        "episodes": count,
        "collisions": int((collision & selected).count_nonzero()),
        "collision_rate": _rate(collision, selected),
      }
    )

  finite = torch.isfinite(speeds)
  append("unknown", ~finite)
  append("<0.00 (receding)", finite & (speeds < 0.0))
  bins = ((0.0, 0.75), (0.75, 1.5), (1.5, float("inf")))
  for lower, upper in bins:
    selected = finite & (speeds >= lower) & (speeds < upper)
    label = f"{lower:.2f}--{upper:.2f}" if math.isfinite(upper) else f">={lower:.2f}"
    append(label, selected)
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--task-id",
    choices=(
      LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
      LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
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
    default=LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
    help="rl-cfg/runner task id used to load the checkpoint's actor config",
  )
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=100)
  parser.add_argument("--seed", type=int, default=23)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--human-runtime", choices=("online", "packed"), default="online")
  parser.add_argument("--surface-tolerance-m", type=float, default=0.10)
  parser.add_argument("--recent-window-s", type=float, default=0.30)
  parser.add_argument("--fast-approach-mps", type=float, default=1.5)
  parser.add_argument("--minimum-arm-request-rad", type=float, default=0.05)
  parser.add_argument("--minimum-limb-response", type=float, default=0.25)
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
  configure_torch_backends()

  cfg = _configure_env(
    num_envs=args.num_envs,
    seed=args.seed,
    motion_file=args.motion_file,
    online_human=args.human_runtime == "online",
    task_id=args.task_id,
    encounter_preset=args.encounter_preset,
  )
  raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task_id)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task_id)
  if runner_cls is None:
    raise RuntimeError("avoidance task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)
  # Loading restores common_step_counter after the initial environment reset;
  # reset again so time-based human trajectories use that restored clock.
  observations, _ = env.reset()

  robot = raw_env.scene["robot"]
  human = raw_env.scene[PRIMARY_HUMAN_ENTITY_NAME]
  lidar = raw_env.scene[LIDAR_SENSOR_NAME]
  if not isinstance(lidar, HeldScanRayCastSensor):
    raise TypeError("diagnostic requires HeldScanRayCastSensor")
  command = raw_env.command_manager.get_term("motion")
  action_term = raw_env.action_manager.get_term("joint_pos")
  link_names = tuple(command.cfg.link_filter.body_names)
  link_ids, resolved_link_names = robot.find_bodies(link_names, preserve_order=True)
  if tuple(resolved_link_names) != link_names:
    raise RuntimeError("link order mismatch")
  link_ids_t = torch.tensor(link_ids, dtype=torch.long, device=args.device)
  human_geom_ids = human.indexing.geom_ids.to(dtype=torch.long)
  target_ids = action_term.target_ids.to(dtype=torch.long)
  arm_mask = torch.tensor(
    [
      any(token in name for token in command.cfg.link_filter.arm_joint_name_tokens)
      for name in action_term.target_names
    ],
    dtype=torch.bool,
    device=args.device,
  )
  lidar_term = cfg.observations["lidar"].terms["directional_range_rate"]
  params = lidar_term.params
  azimuth_samples = int(params["azimuth_samples"])
  elevation_samples = int(params["elevation_samples"])
  azimuth_bins = int(params["azimuth_bins"])
  elevation_bins = int(params["elevation_bins"])
  cell_count = azimuth_bins * elevation_bins

  stats = _new_stats(args.num_envs, args.device)
  active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
  previous_publication = lidar.publication_count.clone()
  previous_scan_distance = torch.linalg.vector_norm(
    human.data.geom_pos_w[:, 0] - lidar.data.pos_w, dim=-1
  )
  current_scan_closing_speed = torch.zeros(args.num_envs, device=args.device)
  recent_window_steps = round(args.recent_window_s / raw_env.step_dt)

  with torch.inference_mode():
    for _ in range(raw_env.max_episode_length):
      if not torch.any(active):
        break
      active_before = active.clone()
      step_index = stats["episode_steps"].clone()

      centers_w = human.data.geom_pos_w
      quaternions_w = human.data.geom_quat_w
      sizes = raw_env.sim.model.geom_size[:, human_geom_ids]
      link_positions_w = robot.data.body_link_pos_w[:, link_ids_t]
      clearances = mdp.capsule_link_surface_clearances(
        link_positions_w,
        centers_w,
        quaternions_w,
        sizes,
        link_radius_m=command.cfg.link_filter.link_radius_m,
      )
      per_link_clearance = clearances.amin(dim=-1)
      step_clearance, step_link_id = per_link_clearance.min(dim=-1)
      new_minimum = active_before & (step_clearance < stats["minimum_clearance_m"])
      stats["minimum_clearance_m"] = torch.where(
        new_minimum, step_clearance, stats["minimum_clearance_m"]
      )
      stats["minimum_clearance_link_id"] = torch.where(
        new_minimum, step_link_id, stats["minimum_clearance_link_id"]
      )
      stats["closing_speed_at_minimum_clearance_mps"] = torch.where(
        new_minimum,
        current_scan_closing_speed,
        stats["closing_speed_at_minimum_clearance_mps"],
      )

      risk = active_before & (
        step_clearance < command.cfg.link_filter.activation_clearance_m
      )
      stats["risk_steps"] += risk.long()
      raw_joint_pos = command._raw_joint_pos()[:, target_ids]
      filtered_joint_pos = command.joint_pos[:, target_ids]
      arm_request = (filtered_joint_pos - raw_joint_pos)[:, arm_mask]
      arm_request_norm = torch.linalg.vector_norm(arm_request, dim=-1)
      requested = risk & (arm_request_norm >= args.minimum_arm_request_rad)
      stats["arm_request_steps"] += requested.long()
      stats["max_arm_request_rad"] = torch.maximum(
        stats["max_arm_request_rad"],
        torch.where(requested, arm_request_norm, 0.0),
      )

      actions = policy(observations)
      clipped_actions = actions
      if agent_cfg.clip_actions is not None:
        clipped_actions = torch.clamp(
          actions,
          -float(agent_cfg.clip_actions),
          float(agent_cfg.clip_actions),
        )
      processed_target = clipped_actions * action_term.scale + action_term.offset
      if hasattr(action_term, "_clip"):
        processed_target = torch.clamp(
          processed_target,
          min=action_term._clip[:, :, 0],
          max=action_term._clip[:, :, 1],
        )
      target_delta = (processed_target - raw_joint_pos)[:, arm_mask]
      actual_delta = (robot.data.joint_pos[:, target_ids] - raw_joint_pos)[:, arm_mask]
      denominator = arm_request.square().sum(dim=-1).clamp_min(1e-6)
      target_projection = (target_delta * arm_request).sum(dim=-1) / denominator
      actual_projection = (actual_delta * arm_request).sum(dim=-1) / denominator
      stats["target_projection_sum"] += torch.where(requested, target_projection, 0.0)
      stats["actual_projection_sum"] += torch.where(requested, actual_projection, 0.0)
      stats["positive_target_response_steps"] += (
        requested & (target_projection >= args.minimum_limb_response)
      ).long()

      publication = lidar.publication_count
      published = active_before & (publication != previous_publication)
      if torch.any(published):
        hit_mask = _human_hit_mask(
          lidar.data.hit_pos_w,
          lidar.data.distances,
          centers_w,
          quaternions_w,
          sizes,
          surface_tolerance_m=args.surface_tolerance_m,
        )
        raw_hit_count = hit_mask.sum(dim=-1)
        human_only_ranges = torch.where(
          hit_mask,
          lidar.data.distances / lidar.cfg.max_distance,
          torch.ones_like(lidar.data.distances),
        )
        human_pooled = _directional_minimum_pool(
          human_only_ranges,
          scan_count=1,
          elevation_samples=elevation_samples,
          azimuth_samples=azimuth_samples,
          elevation_bins=elevation_bins,
          azimuth_bins=azimuth_bins,
        )
        actor_current = observations["lidar"][:, :cell_count]
        human_cells = human_pooled < 1.0
        human_cell_count = human_cells.sum(dim=-1)
        represented_cells = human_cells & (
          torch.abs(actor_current - human_pooled) <= 0.025
        )
        represented_count = represented_cells.sum(dim=-1)
        raw_seen = published & (raw_hit_count > 0)
        represented = published & (represented_count > 0)
        stats["raw_hit_publications"] += raw_seen.long()
        stats["represented_publications"] += represented.long()
        stats["masked_only_publications"] += (raw_seen & ~represented).long()
        stats["max_raw_human_hits"] = torch.maximum(
          stats["max_raw_human_hits"],
          torch.where(published, raw_hit_count, 0),
        )
        stats["max_represented_cells"] = torch.maximum(
          stats["max_represented_cells"],
          torch.where(published, represented_count, 0),
        )
        stats["human_cell_sum"] += torch.where(published, human_cell_count.float(), 0.0)
        stats["represented_cell_sum"] += torch.where(
          published, represented_count.float(), 0.0
        )
        cell_retention = represented_count.float() / human_cell_count.clamp_min(1)
        stats["last_cell_retention"] = torch.where(
          raw_seen, cell_retention, stats["last_cell_retention"]
        )
        stats["last_raw_hit_step"] = torch.where(
          raw_seen, step_index, stats["last_raw_hit_step"]
        )
        stats["last_represented_step"] = torch.where(
          represented, step_index, stats["last_represented_step"]
        )

        human_pelvis_w = centers_w[:, 0]
        scan_distance = torch.linalg.vector_norm(
          human_pelvis_w - lidar.data.pos_w, dim=-1
        )
        closing_speed = (previous_scan_distance - scan_distance) / lidar.cfg.scan_period
        valid_speed = published & torch.isfinite(previous_scan_distance)
        stats["max_closing_speed_mps"] = torch.maximum(
          stats["max_closing_speed_mps"],
          torch.where(valid_speed, closing_speed, 0.0),
        )
        near = valid_speed & (scan_distance <= 2.5)
        stats["max_near_closing_speed_mps"] = torch.maximum(
          stats["max_near_closing_speed_mps"],
          torch.where(near, closing_speed, 0.0),
        )
        current_scan_closing_speed = torch.where(
          valid_speed, closing_speed, current_scan_closing_speed
        )
        first_represented = represented & torch.isnan(
          stats["first_represented_distance_m"]
        )
        stats["first_represented_distance_m"] = torch.where(
          first_represented,
          scan_distance,
          stats["first_represented_distance_m"],
        )
        stats["first_represented_closing_speed_mps"] = torch.where(
          first_represented & valid_speed,
          closing_speed,
          stats["first_represented_closing_speed_mps"],
        )
        previous_scan_distance = torch.where(
          published, scan_distance, previous_scan_distance
        )
      previous_publication = publication.clone()

      observations, _, dones, _ = env.step(actions)
      stats["episode_steps"][active_before] += 1
      new_done = active_before & dones.bool()
      if torch.any(new_done):
        collision = raw_env.termination_manager.get_term(
          "primary_human_collision"
        ) | raw_env.termination_manager.get_term("crowd_collision")
        timeout = raw_env.termination_manager.get_term("time_out")
        stats["collision"][new_done] = collision[new_done]
        stats["timeout"][new_done] = timeout[new_done]
        stats["other_failure"][new_done] = ~(collision[new_done] | timeout[new_done])
        active[new_done] = False

  if args.device.startswith("cuda"):
    torch.cuda.synchronize(args.device)
  stats_cpu = {name: values.cpu() for name, values in stats.items()}
  collision = stats_cpu["collision"]
  timeout = stats_cpu["timeout"]
  final_step = stats_cpu["episode_steps"]
  recent_raw = final_step - stats_cpu["last_raw_hit_step"] <= recent_window_steps
  recent_represented = (
    final_step - stats_cpu["last_represented_step"] <= recent_window_steps
  )
  blindspot_collision = collision & ~recent_raw
  pooled_away_collision = collision & recent_raw & ~recent_represented
  visible_collision = collision & recent_represented
  pooling_degraded_collision = (
    collision & recent_raw & (stats_cpu["last_cell_retention"] < 0.25)
  )
  fast_collision = collision & (
    stats_cpu["closing_speed_at_minimum_clearance_mps"] >= args.fast_approach_mps
  )
  arm_count = stats_cpu["arm_request_steps"].clamp_min(1)
  mean_target_projection = stats_cpu["target_projection_sum"] / arm_count
  mean_actual_projection = stats_cpu["actual_projection_sum"] / arm_count
  target_response_fraction = (
    stats_cpu["positive_target_response_steps"].float() / arm_count
  )
  missing_limb_response = (
    collision
    & (stats_cpu["arm_request_steps"] >= 3)
    & (mean_target_projection < args.minimum_limb_response)
  )

  case_rows = []
  for index in range(args.num_envs):
    link_id = int(stats_cpu["minimum_clearance_link_id"][index])
    case_rows.append(
      {
        "case_id": index,
        "outcome": (
          "collision"
          if bool(collision[index])
          else "timeout"
          if bool(timeout[index])
          else "other_failure"
        ),
        "episode_steps": int(final_step[index]),
        "minimum_clearance_m": _float_or_none(stats_cpu["minimum_clearance_m"][index]),
        "closest_link": link_names[link_id] if link_id >= 0 else None,
        "raw_hit_publications": int(stats_cpu["raw_hit_publications"][index]),
        "represented_publications": int(stats_cpu["represented_publications"][index]),
        "masked_only_publications": int(stats_cpu["masked_only_publications"][index]),
        "max_raw_human_hits": int(stats_cpu["max_raw_human_hits"][index]),
        "max_represented_cells": int(stats_cpu["max_represented_cells"][index]),
        "mean_human_cell_retention": float(
          stats_cpu["represented_cell_sum"][index]
          / stats_cpu["human_cell_sum"][index].clamp_min(1.0)
        ),
        "last_human_cell_retention": _float_or_none(
          stats_cpu["last_cell_retention"][index]
        ),
        "first_represented_distance_m": _float_or_none(
          stats_cpu["first_represented_distance_m"][index]
        ),
        "first_represented_closing_speed_mps": _float_or_none(
          stats_cpu["first_represented_closing_speed_mps"][index]
        ),
        "max_closing_speed_mps": float(stats_cpu["max_closing_speed_mps"][index]),
        "max_near_closing_speed_mps": float(
          stats_cpu["max_near_closing_speed_mps"][index]
        ),
        "closing_speed_at_minimum_clearance_mps": _float_or_none(
          stats_cpu["closing_speed_at_minimum_clearance_mps"][index]
        ),
        "risk_steps": int(stats_cpu["risk_steps"][index]),
        "arm_request_steps": int(stats_cpu["arm_request_steps"][index]),
        "max_arm_request_rad": float(stats_cpu["max_arm_request_rad"][index]),
        "mean_policy_target_projection": float(mean_target_projection[index]),
        "mean_actual_arm_projection": float(mean_actual_projection[index]),
        "target_response_fraction": float(target_response_fraction[index]),
        "blindspot_collision": bool(blindspot_collision[index]),
        "pooled_away_collision": bool(pooled_away_collision[index]),
        "pooling_degraded_collision": bool(pooling_degraded_collision[index]),
        "visible_collision": bool(visible_collision[index]),
        "fast_approach_collision": bool(fast_collision[index]),
        "missing_limb_response": bool(missing_limb_response[index]),
      }
    )

  collisions = int(collision.count_nonzero())
  payload = {
    "checkpoint": str(args.checkpoint.resolve()),
    "task_id": args.task_id,
    "motion_file": str(args.motion_file.resolve()),
    "seed": args.seed,
    "human_runtime": args.human_runtime,
    "num_episodes": args.num_envs,
    "definitions": {
      "recent_visibility_window_s": args.recent_window_s,
      "human_hit_surface_tolerance_m": args.surface_tolerance_m,
      "fast_approach_threshold_mps": args.fast_approach_mps,
      "minimum_arm_request_rad": args.minimum_arm_request_rad,
      "minimum_limb_response_projection": args.minimum_limb_response,
    },
    "summary": {
      "collisions": collisions,
      "collision_rate": collisions / args.num_envs,
      "timeouts": int(timeout.count_nonzero()),
      "other_failures": int(stats_cpu["other_failure"].count_nonzero()),
      "blindspot_collisions": int(blindspot_collision.count_nonzero()),
      "pooled_away_collisions": int(pooled_away_collision.count_nonzero()),
      "pooling_degraded_collisions": int(pooling_degraded_collision.count_nonzero()),
      "visible_but_collision": int(visible_collision.count_nonzero()),
      "fast_approach_collisions": int(fast_collision.count_nonzero()),
      "missing_limb_response_collisions": int(missing_limb_response.count_nonzero()),
      "collision_speed_bins": _speed_bin_summary(
        collision,
        stats_cpu["closing_speed_at_minimum_clearance_mps"],
      ),
      "relative_range_spike_episodes_gt_10_mps": int(
        (stats_cpu["max_near_closing_speed_mps"] > 10.0).count_nonzero()
      ),
    },
    "cases": case_rows,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(payload["summary"], indent=2, sort_keys=True))
  print(f"WROTE {args.output}")
  env.close()


if __name__ == "__main__":
  main()
