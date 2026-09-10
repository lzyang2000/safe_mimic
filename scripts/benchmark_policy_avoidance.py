"""Paired policy benchmark for LiDAR-dependent human avoidance.

This evaluates the first episode from identical seeded scene distributions.
The normal policy receives its real LiDAR observation; the blinded control
receives max-range returns while retaining the real scan-age value.  A policy
that genuinely uses perception should collide less often in the normal arm.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

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
  LIDAR_AVOIDANCE_TASK_ID,
  LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
  UNIFIED_ROOT_LEAD_M,
  mdp,
)
from safe_mimic.tasks.env_cfg import (
  CROWD_CAPSULES_PER_PERSON,
  CROWD_PRIVILEGED_NEAREST_PEOPLE,
  DEFAULT_G1_BALLET_MANIFEST,
  DEFAULT_G1_BALLET_MIRROR_MANIFEST,
  HUMAN_ENTITY_NAME,
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
  unitree_g1_lidar_avoidance_tracking_env_cfg,
  unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)

Mode = Literal[
  "normal",
  "blind",
  "shuffled",
  "aux-zero",
  "aux-oracle",
  "aux-shuffled",
]
Scenario = Literal["default", "forced-human", "dense", "primary-only"]
HumanRuntime = Literal["packed", "online"]
TaskVariant = Literal[
  "baseline",
  "range-rate",
  "auxiliary",
  "coadjust",
  "coadjust-fkc",
  "coadjust-fkc2",
  "coadjust-unified",
  "coadjust-unified-joint",
  "coadjust-unified-joint-leash",
  "coadjust-unified-joint-leash-slow",
  "coadjust-unified-joint-leash-slow-dense",
  "coadjust-unified-joint-leash-ballet",
  "coadjust-unified-joint-leash-ballet-slow",
  "coadjust-unified-joint-leash-ballet-lag",
  "coadjust-unified-joint-leash-ballet-blind",
  "coadjust-unified-joint-leash-ballet-blind-nominal",
  "coadjust-unified-joint-leash-ballet-blind-nohumans",
  "coadjust-unified-joint-leash-ballet-moves",
]

_CROWD_CLEARANCE_METRIC = "benchmark_crowd_clearance"
_PRIMARY_CLEARANCE_METRIC = "benchmark_primary_clearance"
_CROWD_COLLISION_TERM = "crowd_collision"
_PRIMARY_COLLISION_TERM = "primary_human_collision"


@dataclass
class EvaluationResult:
  mode: str
  scenario: str
  human_runtime: str
  num_envs: int
  completed_episodes: int
  human_present_episodes: int
  mean_episode_length: float
  mean_reward_per_step: float
  timeout_rate: float
  collision_rate: float
  collision_rate_when_human_present: float
  crowd_collision_rate: float
  primary_collision_rate: float
  other_failure_rate: float
  incomplete_rate: float
  clearance_lt_080_rate: float
  clearance_lt_040_rate: float
  clearance_lt_020_rate: float
  clearance_lt_010_rate: float
  minimum_clearance_p05_m: float | None
  minimum_clearance_p50_m: float | None
  minimum_clearance_mean_m: float | None
  wall_time_s: float


@dataclass
class _EvaluationArrays:
  collision: torch.Tensor
  crowd_collision: torch.Tensor
  primary_collision: torch.Tensor
  timeout: torch.Tensor
  human_present: torch.Tensor
  minimum_clearance: torch.Tensor


def _configure_scenario(cfg: Any, scenario: Scenario) -> None:
  crowd_event = cfg.events[HUMAN_MOTION_EVENT_NAME]
  if scenario == "default":
    return
  crowd_event.params["obstacle_free_probability"] = 0.0
  if scenario == "primary-only":
    # Keep the shared obstacle-free mask false so the primary human remains
    # active, but park every stationary crowd member below the world.
    crowd_event.params["min_count"] = 0
    crowd_event.params["max_count"] = 0
    crowd_event.params["randomize_density"] = False
    return
  if scenario == "dense":
    # Retain the trained 2--4 m radius distribution, but always use the packed
    # count implied by the configured shoulder spacing.
    crowd_event.params["randomize_density"] = False


def _build_cfg(
  *,
  task_variant: TaskVariant,
  scenario: Scenario,
  human_runtime: HumanRuntime,
  num_envs: int,
  seed: int,
  motion_file: Path | None,
) -> Any:
  cfg_fn = {
    "baseline": unitree_g1_lidar_avoidance_tracking_env_cfg,
    "range-rate": unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg,
    "auxiliary": unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
    "coadjust": unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
    "coadjust-fkc": unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
    "coadjust-fkc2": unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
    "coadjust-unified": unitree_g1_lidar_unified_reference_tracking_env_cfg,
    "coadjust-unified-joint": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play, active_joint_reward=True
      )
    ),
    "coadjust-unified-joint-leash": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
      )
    ),
    "coadjust-unified-joint-leash-slow": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        slow_regime=True,
        filter_gated_ee_termination=True,
      )
    ),
    "coadjust-unified-joint-leash-slow-dense": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        slow_regime=True,
        dense_encounters=True,
      )
    ),
    "coadjust-unified-joint-leash-ballet": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      )
    ),
    "coadjust-unified-joint-leash-ballet-slow": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        slow_regime=True,
        motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      )
    ),
    "coadjust-unified-joint-leash-ballet-lag": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        lag_aware_ee_termination=True,
        motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
      )
    ),
    "coadjust-unified-joint-leash-ballet-blind": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
        blind_actor=True,
      )
    ),
    "coadjust-unified-joint-leash-ballet-blind-nominal": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
        blind_actor=True,
        nominal_reference=True,
      )
    ),
    "coadjust-unified-joint-leash-ballet-blind-nohumans": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
        blind_actor=True,
        nominal_reference=True,
        training_humans=False,
      )
    ),
    "coadjust-unified-joint-leash-ballet-moves": lambda play: (
      unitree_g1_lidar_unified_reference_tracking_env_cfg(
        play=play,
        active_joint_reward=True,
        root_lead_m=UNIFIED_ROOT_LEAD_M,
        planar_filter_at_robot_root=True,
        motion_manifest=str(DEFAULT_G1_BALLET_MIRROR_MANIFEST),
        escape_moves=True,
      )
    ),
  }[task_variant]
  cfg = cfg_fn(play=human_runtime == "online")
  if task_variant in ("coadjust-fkc", "coadjust-fkc2"):
    # Match training semantics: the FKC task's body position/orientation
    # targets follow the filtered joint corrections, so task-space metrics
    # reported here are consistent with what the checkpoint was trained on.
    cfg.commands["motion"].propagate_arm_corrections_to_body_targets = True
  if task_variant == "coadjust-fkc2":
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
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  # Play mode normally runs forever and retains Viser-only meshes/debug rays.
  # Restore the finite training horizon and headless geometry for evaluation.
  if human_runtime == "online":
    cfg.episode_length_s = 10.0
    for event_name in (HUMAN_MOTION_EVENT_NAME, PRIMARY_HUMAN_EVENT_NAME):
      cfg.events[event_name].params["show_mesh"] = False
    cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["print_velocity"] = False
    for sensor in cfg.scene.sensors or ():
      sensor.debug_vis = False
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
  for group in cfg.observations.values():
    group.enable_corruption = False
  lidar_term_name = (
    "directional_range_rate"
    if task_variant
    in (
      "range-rate",
      "auxiliary",
      "coadjust",
      "coadjust-fkc",
      "coadjust-fkc2",
      "coadjust-unified",
      "coadjust-unified-joint",
      "coadjust-unified-joint-leash",
      "coadjust-unified-joint-leash-slow",
      "coadjust-unified-joint-leash-slow-dense",
      "coadjust-unified-joint-leash-ballet",
      "coadjust-unified-joint-leash-ballet-slow",
      "coadjust-unified-joint-leash-ballet-lag",
      "coadjust-unified-joint-leash-ballet-blind",
      "coadjust-unified-joint-leash-ballet-blind-nominal",
      "coadjust-unified-joint-leash-ballet-blind-nohumans",
      "coadjust-unified-joint-leash-ballet-moves",
    )
    else "directional_scan_pair"
  )
  lidar_term = cfg.observations["lidar"].terms[lidar_term_name]
  lidar_term.params["noise_cfg"] = None
  if motion_file is not None:
    cfg.commands["motion"].motion_file = str(motion_file)
  _configure_scenario(cfg, scenario)
  link_filter = cfg.commands["motion"].link_filter
  cfg.metrics[_CROWD_CLEARANCE_METRIC] = MetricsTermCfg(
    func=mdp.HumanCapsuleLinkClearance,
    params={
      "robot_entity": "robot",
      "human_entity": HUMAN_ENTITY_NAME,
      "robot_link_names": link_filter.body_names,
      "link_radius": link_filter.link_radius_m,
      "capsules_per_group": CROWD_CAPSULES_PER_PERSON,
      "nearest_groups": CROWD_PRIVILEGED_NEAREST_PEOPLE,
    },
    reduce="last",
  )
  cfg.metrics[_PRIMARY_CLEARANCE_METRIC] = MetricsTermCfg(
    func=mdp.HumanCapsuleLinkClearance,
    params={
      "robot_entity": "robot",
      "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
      "robot_link_names": link_filter.body_names,
      "link_radius": link_filter.link_radius_m,
    },
    reduce="last",
  )
  return cfg


def _policy_action(
  policy: Any,
  observations: TensorDict,
  mode: Mode,
  task_variant: TaskVariant,
) -> torch.Tensor:
  if mode == "normal":
    return policy(observations)
  if mode in ("aux-zero", "aux-oracle", "aux-shuffled"):
    if task_variant not in (
      "auxiliary",
      "coadjust",
      "coadjust-fkc",
      "coadjust-fkc2",
      "coadjust-unified",
      "coadjust-unified-joint",
      "coadjust-unified-joint-leash",
      "coadjust-unified-joint-leash-slow",
      "coadjust-unified-joint-leash-slow-dense",
      "coadjust-unified-joint-leash-ballet",
      "coadjust-unified-joint-leash-ballet-slow",
      "coadjust-unified-joint-leash-ballet-lag",
      "coadjust-unified-joint-leash-ballet-blind",
      "coadjust-unified-joint-leash-ballet-blind-nominal",
      "coadjust-unified-joint-leash-ballet-blind-nohumans",
      "coadjust-unified-joint-leash-ballet-moves",
    ) or not isinstance(policy, PerceptiveLidarActor):
      raise ValueError(f"{mode} requires an auxiliary LiDAR checkpoint")
    if mode == "aux-zero":
      override = observations[policy.tracking_obs_group].new_zeros(
        (*observations.batch_size, policy.avoidance_prediction_dim)
      )
    elif mode == "aux-oracle":
      override = observations["avoidance_teacher"]
    else:
      override = policy.predict_avoidance(observations).roll(shifts=1, dims=0)
    return policy.action_with_avoidance_override(observations, override)
  lidar = observations["lidar"]
  directional = lidar[..., :-1]
  saved = directional.clone()
  if mode == "blind":
    if task_variant in (
      "range-rate",
      "auxiliary",
      "coadjust",
      "coadjust-fkc",
      "coadjust-fkc2",
      "coadjust-unified",
      "coadjust-unified-joint",
      "coadjust-unified-joint-leash",
      "coadjust-unified-joint-leash-slow",
      "coadjust-unified-joint-leash-slow-dense",
      "coadjust-unified-joint-leash-ballet",
      "coadjust-unified-joint-leash-ballet-slow",
      "coadjust-unified-joint-leash-ballet-lag",
      "coadjust-unified-joint-leash-ballet-blind",
      "coadjust-unified-joint-leash-ballet-blind-nominal",
      "coadjust-unified-joint-leash-ballet-blind-nohumans",
      "coadjust-unified-joint-leash-ballet-moves",
    ):
      cell_count = directional.shape[-1] // 2
      directional[..., :cell_count] = 1.0
      directional[..., cell_count:] = 0.0
    else:
      directional.fill_(1.0)
  elif mode == "shuffled":
    directional.copy_(saved.roll(shifts=1, dims=0))
  else:
    raise ValueError(f"unknown benchmark mode: {mode}")
  try:
    return policy(observations)
  finally:
    directional.copy_(saved)


def _metric_column(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  manager = env.metrics_manager
  index = manager.active_terms.index(name)
  return manager._step_values[:, index]


def _finite_summary(values: torch.Tensor) -> tuple[float | None, ...]:
  finite = values[torch.isfinite(values)]
  if finite.numel() == 0:
    return None, None, None
  quantiles = torch.quantile(
    finite,
    torch.tensor([0.05, 0.5], device=finite.device),
  )
  return float(quantiles[0]), float(quantiles[1]), float(finite.mean())


def _evaluate(
  *,
  checkpoint: Path,
  task_variant: TaskVariant,
  mode: Mode,
  scenario: Scenario,
  human_runtime: HumanRuntime,
  num_envs: int,
  seed: int,
  device: str,
  max_steps: int | None,
  motion_file: Path | None,
) -> tuple[EvaluationResult, _EvaluationArrays]:
  cfg = _build_cfg(
    task_variant=task_variant,
    scenario=scenario,
    human_runtime=human_runtime,
    num_envs=num_envs,
    seed=seed,
    motion_file=motion_file,
  )
  raw_env = ManagerBasedRlEnv(cfg=cfg, device=device)
  task_id = {
    "baseline": LIDAR_AVOIDANCE_TASK_ID,
    "range-rate": LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
    "auxiliary": LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
    "coadjust": LIDAR_AUXILIARY_COADJUST_TASK_ID,
    "coadjust-fkc": LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
    "coadjust-fkc2": LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
    "coadjust-unified": LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
    "coadjust-unified-joint": LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
    "coadjust-unified-joint-leash": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID
    ),
    "coadjust-unified-joint-leash-slow": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID
    ),
    "coadjust-unified-joint-leash-slow-dense": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID
    ),
    "coadjust-unified-joint-leash-ballet": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID
    ),
    "coadjust-unified-joint-leash-ballet-slow": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID
    ),
    "coadjust-unified-joint-leash-ballet-lag": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID
    ),
    "coadjust-unified-joint-leash-ballet-blind": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID
    ),
    "coadjust-unified-joint-leash-ballet-blind-nominal": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID
    ),
    "coadjust-unified-joint-leash-ballet-blind-nohumans": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID
    ),
    "coadjust-unified-joint-leash-ballet-moves": (
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID
    ),
  }[task_variant]
  agent_cfg = load_rl_cfg(task_id)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    raise RuntimeError("LiDAR avoidance task has no registered runner")
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  # Loading restores common_step_counter after the initial environment reset;
  # reset again so time-based human trajectories use that restored clock.
  observations, _ = env.reset()

  episode_limit = raw_env.max_episode_length
  step_limit = episode_limit if max_steps is None else min(max_steps, episode_limit)
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  reward_sum = torch.zeros(num_envs, device=device)
  minimum_clearance = torch.full((num_envs,), torch.inf, device=device)
  causes = {
    name: torch.zeros(num_envs, dtype=torch.bool, device=device)
    for name in raw_env.termination_manager.active_terms
  }
  human_present = ~raw_env._safe_mimic_obstacle_free_envs.clone()

  start = time.perf_counter()
  with torch.inference_mode():
    for _ in range(step_limit):
      active_before = active.clone()
      actions = _policy_action(policy, observations, mode, task_variant)
      observations, reward, dones, _ = env.step(actions)
      episode_steps[active_before] += 1
      reward_sum[active_before] += reward[active_before]
      step_clearance = torch.minimum(
        _metric_column(raw_env, _CROWD_CLEARANCE_METRIC),
        _metric_column(raw_env, _PRIMARY_CLEARANCE_METRIC),
      )
      minimum_clearance[active_before] = torch.minimum(
        minimum_clearance[active_before],
        step_clearance[active_before],
      )
      new_done = active_before & dones.bool()
      if torch.any(new_done):
        for name in causes:
          causes[name][new_done] = raw_env.termination_manager.get_term(name)[new_done]
        active[new_done] = False
      if not torch.any(active):
        break
  if device.startswith("cuda"):
    torch.cuda.synchronize(device)
  wall_time_s = time.perf_counter() - start

  timeout = causes.get("time_out")
  if timeout is None:
    timeout = raw_env.reset_time_outs.clone() & ~active
  crowd_collision = causes[_CROWD_COLLISION_TERM]
  primary_collision = causes[_PRIMARY_COLLISION_TERM]
  collision = crowd_collision | primary_collision
  completed = ~active
  other_failure = completed & ~timeout & ~collision
  reward_per_step = reward_sum / episode_steps.clamp_min(1)
  human_completed = completed & human_present
  p05, p50, clearance_mean = _finite_summary(minimum_clearance[completed])

  def rate(mask: torch.Tensor, denominator: torch.Tensor | None = None) -> float:
    selected = completed if denominator is None else denominator
    count = int(selected.count_nonzero())
    if count == 0:
      return float("nan")
    return float((mask & selected).count_nonzero()) / count

  result = EvaluationResult(
    mode=mode,
    scenario=scenario,
    human_runtime=human_runtime,
    num_envs=num_envs,
    completed_episodes=int(completed.count_nonzero()),
    human_present_episodes=int(human_completed.count_nonzero()),
    mean_episode_length=float(episode_steps[completed].float().mean()),
    mean_reward_per_step=float(reward_per_step[completed].mean()),
    timeout_rate=rate(timeout),
    collision_rate=rate(collision),
    collision_rate_when_human_present=rate(collision, human_completed),
    crowd_collision_rate=rate(crowd_collision),
    primary_collision_rate=rate(primary_collision),
    other_failure_rate=rate(other_failure),
    incomplete_rate=float(active.count_nonzero()) / num_envs,
    clearance_lt_080_rate=rate(minimum_clearance < 0.8),
    clearance_lt_040_rate=rate(minimum_clearance < 0.4),
    clearance_lt_020_rate=rate(minimum_clearance < 0.2),
    clearance_lt_010_rate=rate(minimum_clearance < 0.1),
    minimum_clearance_p05_m=p05,
    minimum_clearance_p50_m=p50,
    minimum_clearance_mean_m=clearance_mean,
    wall_time_s=wall_time_s,
  )
  arrays = _EvaluationArrays(
    collision=collision.cpu(),
    crowd_collision=crowd_collision.cpu(),
    primary_collision=primary_collision.cpu(),
    timeout=timeout.cpu(),
    human_present=human_present.cpu(),
    minimum_clearance=minimum_clearance.cpu(),
  )
  env.close()
  del policy, runner, env, raw_env
  gc.collect()
  if device.startswith("cuda"):
    torch.cuda.empty_cache()
  return result, arrays


def _paired_summary(
  normal: _EvaluationArrays,
  control: _EvaluationArrays,
  *,
  control_name: str,
) -> dict[str, float | int | str]:
  human_present = normal.human_present & control.human_present
  normal_collision = normal.collision & human_present
  control_collision = control.collision & human_present
  control_only = (~normal_collision & control_collision).count_nonzero().item()
  normal_only = (normal_collision & ~control_collision).count_nonzero().item()
  pairs = int(human_present.count_nonzero())
  discordant = control_only + normal_only
  if pairs == 0:
    return {
      "control": control_name,
      "paired_human_present_episodes": 0,
      "normal_collision_rate": float("nan"),
      "control_collision_rate": float("nan"),
      "control_minus_normal_collision_rate": float("nan"),
      "control_only_collision_pairs": int(control_only),
      "normal_only_collision_pairs": int(normal_only),
      "mcnemar_z_approx": 0.0,
    }
  z = (control_only - normal_only) / math.sqrt(discordant) if discordant else 0.0
  return {
    "control": control_name,
    "paired_human_present_episodes": pairs,
    "normal_collision_rate": float(normal_collision.count_nonzero()) / pairs,
    "control_collision_rate": float(control_collision.count_nonzero()) / pairs,
    "control_minus_normal_collision_rate": (
      float(control_collision.count_nonzero() - normal_collision.count_nonzero())
      / pairs
    ),
    "control_only_collision_pairs": int(control_only),
    "normal_only_collision_pairs": int(normal_only),
    "mcnemar_z_approx": float(z),
  }


def _print_result(result: EvaluationResult) -> None:
  print(
    "RESULT "
    f"scenario={result.scenario} runtime={result.human_runtime} "
    f"mode={result.mode} "
    f"episodes={result.completed_episodes}/{result.num_envs} "
    f"human_present={result.human_present_episodes} "
    f"collision={100.0 * result.collision_rate:.2f}% "
    f"collision_human_present={100.0 * result.collision_rate_when_human_present:.2f}% "
    f"crowd={100.0 * result.crowd_collision_rate:.2f}% "
    f"primary={100.0 * result.primary_collision_rate:.2f}% "
    f"timeout={100.0 * result.timeout_rate:.2f}% "
    f"other_failure={100.0 * result.other_failure_rate:.2f}% "
    f"clearance_lt_0.2m={100.0 * result.clearance_lt_020_rate:.2f}% "
    f"clearance_p05={result.minimum_clearance_p05_m} "
    f"reward_per_step={result.mean_reward_per_step:.5f} "
    f"mean_length={result.mean_episode_length:.1f} "
    f"wall={result.wall_time_s:.1f}s"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--task-variant",
    choices=(
      "baseline",
      "range-rate",
      "auxiliary",
      "coadjust",
      "coadjust-fkc",
      "coadjust-fkc2",
      "coadjust-unified",
      "coadjust-unified-joint",
      "coadjust-unified-joint-leash",
      "coadjust-unified-joint-leash-slow",
      "coadjust-unified-joint-leash-slow-dense",
      "coadjust-unified-joint-leash-ballet",
      "coadjust-unified-joint-leash-ballet-slow",
      "coadjust-unified-joint-leash-ballet-lag",
      "coadjust-unified-joint-leash-ballet-blind",
      "coadjust-unified-joint-leash-ballet-blind-nominal",
      "coadjust-unified-joint-leash-ballet-blind-nohumans",
      "coadjust-unified-joint-leash-ballet-moves",
    ),
    default="baseline",
    help="observation/reward task contract used to train the checkpoint",
  )
  parser.add_argument("--motion-file", type=Path)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--max-steps", type=int)
  parser.add_argument(
    "--human-runtime",
    choices=("packed", "online"),
    default="packed",
    help="use the headless training bank or the online play/demo composer",
  )
  parser.add_argument(
    "--scenarios",
    nargs="+",
    choices=("default", "forced-human", "dense", "primary-only"),
    default=("default", "forced-human"),
  )
  parser.add_argument(
    "--modes",
    nargs="+",
    choices=(
      "normal",
      "blind",
      "shuffled",
      "aux-zero",
      "aux-oracle",
      "aux-shuffled",
    ),
    default=("normal", "blind"),
  )
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.num_envs < 1:
    parser.error("--num-envs must be positive")
  if not args.checkpoint.is_file():
    parser.error(f"checkpoint not found: {args.checkpoint}")
  if args.motion_file is not None and not args.motion_file.is_file():
    parser.error(f"motion file not found: {args.motion_file}")
  configure_torch_backends()

  results: list[EvaluationResult] = []
  paired: list[dict[str, float | int | str]] = []
  for scenario in args.scenarios:
    arrays_by_mode: dict[str, _EvaluationArrays] = {}
    for mode in args.modes:
      result, arrays = _evaluate(
        checkpoint=args.checkpoint,
        task_variant=args.task_variant,
        mode=mode,
        scenario=scenario,
        human_runtime=args.human_runtime,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        motion_file=args.motion_file,
      )
      results.append(result)
      arrays_by_mode[mode] = arrays
      _print_result(result)
    if "normal" in arrays_by_mode:
      for control_name in (
        "blind",
        "shuffled",
        "aux-zero",
        "aux-oracle",
        "aux-shuffled",
      ):
        if control_name not in arrays_by_mode:
          continue
        comparison = _paired_summary(
          arrays_by_mode["normal"],
          arrays_by_mode[control_name],
          control_name=control_name,
        )
        comparison["scenario"] = scenario
        paired.append(comparison)
        print("PAIRED " + json.dumps(comparison, sort_keys=True))

  payload = {
    "checkpoint": str(args.checkpoint.resolve()),
    "task_variant": args.task_variant,
    "motion_file": None
    if args.motion_file is None
    else str(args.motion_file.resolve()),
    "num_envs": args.num_envs,
    "seed": args.seed,
    "device": args.device,
    "human_runtime": args.human_runtime,
    "results": [asdict(result) for result in results],
    "paired": paired,
  }
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
  main()
