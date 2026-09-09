"""Unified-Joint-Leash-Slow: spawn outside reach, slow encounters, gated ee check."""

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking import mdp as tracking_mdp

from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  mdp,
)
from safe_mimic.tasks.encounter_sampling import EncounterSampler
from safe_mimic.tasks.env_cfg import (
  EE_LOOSENED_THRESHOLD_M,
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  SLOW_REGIME_DELAY_RANGE_S,
  SLOW_REGIME_DENSE_DELAY_RANGE_S,
  SLOW_REGIME_DENSE_TTC_EDGES_S,
  SLOW_REGIME_MAX_SPEED_MPS,
  SLOW_REGIME_MIN_SPAWN_RADIUS_M,
  SLOW_REGIME_SPEED_EDGES_MPS,
  SLOW_REGIME_TTC_EDGES_S,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)


def _leash(**kwargs):
  return unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=0.3,
    planar_filter_at_robot_root=True,
    **kwargs,
  )


def test_slow_regime_moves_the_walking_human_outside_the_critical_distance() -> None:
  base = _leash().events[PRIMARY_HUMAN_EVENT_NAME].params
  slow = _leash(slow_regime=True).events[PRIMARY_HUMAN_EVENT_NAME].params
  assert base["min_initial_spawn_radius_m"] == 0.75
  assert slow["min_initial_spawn_radius_m"] == SLOW_REGIME_MIN_SPAWN_RADIUS_M
  assert SLOW_REGIME_MIN_SPAWN_RADIUS_M >= 1.6
  assert slow["max_initial_spawn_radius_m"] == base["max_initial_spawn_radius_m"]


def test_slow_regime_sets_the_encounter_sampler_and_delay_range() -> None:
  for play in (False, True):
    p = _leash(play=play, slow_regime=True).events[PRIMARY_HUMAN_EVENT_NAME].params
    assert p["encounter_sampling"] == "ttc"
    assert p["speed_bin_edges_mps"] == SLOW_REGIME_SPEED_EDGES_MPS
    assert p["ttc_bin_edges_s"] == SLOW_REGIME_TTC_EDGES_S
    assert (p["min_intersection_delay_s"], p["max_intersection_delay_s"]) == (
      SLOW_REGIME_DELAY_RANGE_S
    )
    # No bin is "hard": the curriculum ramp must not hide part of the slow regime.
    assert p["hard_speed_above_mps"] > SLOW_REGIME_SPEED_EDGES_MPS[-1]
    assert p["hard_ttc_below_s"] < SLOW_REGIME_TTC_EDGES_S[0]
  assert SLOW_REGIME_SPEED_EDGES_MPS[-1] == SLOW_REGIME_MAX_SPEED_MPS


def test_slow_regime_crowd_is_untouched() -> None:
  base = _leash().events[HUMAN_MOTION_EVENT_NAME].params
  slow = _leash(slow_regime=True).events[HUMAN_MOTION_EVENT_NAME].params
  assert slow == base


def test_slow_regime_sampler_never_exceeds_the_slow_speed() -> None:
  sampler = EncounterSampler(
    4096,
    "cpu",
    ttc_bin_edges_s=SLOW_REGIME_TTC_EDGES_S,
    speed_bin_edges_mps=SLOW_REGIME_SPEED_EDGES_MPS,
    spawn_radius_clamp_m=(SLOW_REGIME_MIN_SPAWN_RADIUS_M, 4.0),
    hard_ttc_below_s=0.1,
    hard_speed_above_mps=99.0,
  )
  torch.manual_seed(0)
  env_ids = torch.arange(4096)
  speeds = []
  for _ in range(4):
    _, speed, radius = sampler.sample(
      env_ids, 1.0e6, torch.zeros(4096, dtype=torch.bool)
    )
    speeds.append(speed)
    assert float(radius.min()) >= SLOW_REGIME_MIN_SPAWN_RADIUS_M - 1e-6
  speed = torch.cat(speeds)
  assert float(speed.max()) <= SLOW_REGIME_MAX_SPEED_MPS + 1e-6


def test_filter_gated_termination_is_opt_in_and_keeps_the_stock_bodies() -> None:
  stock = _leash().terminations["ee_body_pos"]
  gated = _leash(filter_gated_ee_termination=True).terminations["ee_body_pos"]
  assert stock.func is tracking_mdp.bad_motion_body_pos_z_only
  assert gated.func is mdp.bad_motion_body_pos_z_only_filter_gated
  assert gated.params["body_names"] == stock.params["body_names"]
  assert gated.params["threshold"] == stock.params["threshold"] == 0.25
  assert gated.params["loosened_threshold"] == EE_LOOSENED_THRESHOLD_M == 0.5
  assert gated.params["activation_threshold_rad"] == 0.05
  assert gated.params["command_name"] == "motion"


def test_leash_slow_task_registered_and_leash_untouched() -> None:
  for play in (False, True):
    cfg = load_env_cfg(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID, play=play
    )
    p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
    assert p["min_initial_spawn_radius_m"] == SLOW_REGIME_MIN_SPAWN_RADIUS_M
    assert p["speed_bin_edges_mps"] == SLOW_REGIME_SPEED_EDGES_MPS
    assert (
      cfg.terminations["ee_body_pos"].func
      is mdp.bad_motion_body_pos_z_only_filter_gated
    )
    motion = cfg.commands["motion"]
    assert motion.max_root_lead_m == 0.3 and motion.planar_filter_at_robot_root
    assert "motion_active_joint_pos" in cfg.rewards
  leash = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID)
  assert (
    leash.events[PRIMARY_HUMAN_EVENT_NAME].params["min_initial_spawn_radius_m"] == 0.75
  )
  assert (
    leash.terminations["ee_body_pos"].func is tracking_mdp.bad_motion_body_pos_z_only
  )
  rl = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID)
  base = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID)
  assert rl.experiment_name.endswith("coadjust_unified_joint_leash_slow")
  assert rl.actor.adjust_command_with_joint_prediction is True
  assert rl.algorithm.avoidance_teacher_mix_decay_updates == 8000
  assert rl.actor.__class__ is base.actor.__class__


# --- dense slow regime (2026-09-04): shorter TTC cap, strict ee termination ---


def test_dense_slow_regime_shortens_only_the_ttc_side() -> None:
  slow = _leash(slow_regime=True).events[PRIMARY_HUMAN_EVENT_NAME].params
  dense = (
    _leash(slow_regime=True, dense_encounters=True)
    .events[PRIMARY_HUMAN_EVENT_NAME]
    .params
  )
  assert dense["ttc_bin_edges_s"] == SLOW_REGIME_DENSE_TTC_EDGES_S
  assert SLOW_REGIME_DENSE_TTC_EDGES_S[-1] <= 5.0 < SLOW_REGIME_TTC_EDGES_S[-1]
  assert (dense["min_intersection_delay_s"], dense["max_intersection_delay_s"]) == (
    SLOW_REGIME_DENSE_DELAY_RANGE_S
  )
  for key in (
    "speed_bin_edges_mps",
    "min_initial_spawn_radius_m",
    "encounter_sampling",
  ):
    assert dense[key] == slow[key]
  # Explicitly NOT touched (user direction): the obstacle-free probability.
  crowd_slow = _leash(slow_regime=True).events[HUMAN_MOTION_EVENT_NAME].params
  crowd_dense = (
    _leash(slow_regime=True, dense_encounters=True)
    .events[HUMAN_MOTION_EVENT_NAME]
    .params
  )
  assert (
    crowd_dense["obstacle_free_probability"] == crowd_slow["obstacle_free_probability"]
  )


def test_dense_encounters_requires_the_slow_regime() -> None:
  import pytest

  with pytest.raises(ValueError, match="slow_regime"):
    _leash(dense_encounters=True)


def test_dense_slow_regime_sampler_stays_under_the_slow_speed() -> None:
  sampler = EncounterSampler(
    4096,
    "cpu",
    ttc_bin_edges_s=SLOW_REGIME_DENSE_TTC_EDGES_S,
    speed_bin_edges_mps=SLOW_REGIME_SPEED_EDGES_MPS,
    spawn_radius_clamp_m=(SLOW_REGIME_MIN_SPAWN_RADIUS_M, 4.0),
    hard_ttc_below_s=0.1,
    hard_speed_above_mps=99.0,
  )
  torch.manual_seed(1)
  env_ids = torch.arange(4096)
  ttc, speed, radius = sampler.sample(
    env_ids, 1.0e6, torch.zeros(4096, dtype=torch.bool)
  )
  assert float(speed.max()) <= SLOW_REGIME_MAX_SPEED_MPS + 1e-6
  assert float(ttc.max()) <= SLOW_REGIME_DENSE_TTC_EDGES_S[-1] + 1e-6
  assert float(radius.min()) >= SLOW_REGIME_MIN_SPAWN_RADIUS_M - 1e-6


def test_leash_slow_dense_task_registered_with_strict_ee_termination() -> None:
  for play in (False, True):
    cfg = load_env_cfg(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID, play=play
    )
    p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
    assert p["ttc_bin_edges_s"] == SLOW_REGIME_DENSE_TTC_EDGES_S
    assert p["min_initial_spawn_radius_m"] == SLOW_REGIME_MIN_SPAWN_RADIUS_M
    # The gated termination regressed arm compliance (slow@15k gate): strict again.
    assert (
      cfg.terminations["ee_body_pos"].func is tracking_mdp.bad_motion_body_pos_z_only
    )
    assert cfg.terminations["ee_body_pos"].params["threshold"] == 0.25
    motion = cfg.commands["motion"]
    assert motion.max_root_lead_m == 0.3 and motion.planar_filter_at_robot_root
  rl = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID)
  assert rl.experiment_name.endswith("coadjust_unified_joint_leash_slow_dense")
  assert rl.algorithm.avoidance_teacher_mix_decay_updates == 8000


# --- ballet library variant (2026-09-05): on the LEASH base, not dense -------

from safe_mimic.tasks import (  # noqa: E402
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
)
from safe_mimic.tasks.env_cfg import DEFAULT_G1_BALLET_MANIFEST  # noqa: E402


def test_motion_manifest_kwarg_points_the_reference_at_the_library() -> None:
  base = _leash().commands["motion"]
  ballet = _leash(motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST)).commands["motion"]
  assert base.motion_file.endswith(".npz")
  assert ballet.motion_file == str(DEFAULT_G1_BALLET_MANIFEST)
  assert ballet.manifest_splits is None  # every clip, user direction
  assert DEFAULT_G1_BALLET_MANIFEST.is_file()


def test_ballet_task_is_the_leash_setup_with_the_library_reference() -> None:
  """The leash configuration is canonical (user, 2026-09-05); ballet only swaps
  the reference. No slow regime, no dense encounters, strict stock ee check."""
  leash = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID)
  for play in (False, True):
    cfg = load_env_cfg(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID, play=play
    )
    motion = cfg.commands["motion"]
    assert motion.motion_file == str(DEFAULT_G1_BALLET_MANIFEST)
    assert motion.manifest_splits is None
    assert motion.max_root_lead_m == leash.commands["motion"].max_root_lead_m == 0.3
    assert motion.planar_filter_at_robot_root is True
    p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
    q = (
      load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID, play=play)
      .events[PRIMARY_HUMAN_EVENT_NAME]
      .params
    )
    for key in (
      "min_initial_spawn_radius_m",
      "max_initial_spawn_radius_m",
      "min_intersection_delay_s",
      "max_intersection_delay_s",
      "encounter_sampling",
    ):
      assert p[key] == q[key], key
    assert "ttc_bin_edges_s" not in p or p["ttc_bin_edges_s"] == q.get(
      "ttc_bin_edges_s"
    )
    assert (
      cfg.terminations["ee_body_pos"].func is tracking_mdp.bad_motion_body_pos_z_only
    )
    assert set(cfg.rewards) == set(leash.rewards)
  rl = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID)
  assert rl.experiment_name.endswith("coadjust_unified_joint_leash_ballet")
  assert rl.algorithm.avoidance_teacher_mix_decay_updates == 8000


# --- ballet + slow regime (2026-09-06) ----------------------------------------

from safe_mimic.tasks import (  # noqa: E402
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
)


def test_ballet_slow_task_is_ballet_plus_the_slow_regime_only() -> None:
  ballet = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID)
  for play in (False, True):
    cfg = load_env_cfg(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID, play=play
    )
    motion = cfg.commands["motion"]
    assert motion.motion_file == str(DEFAULT_G1_BALLET_MANIFEST)
    assert motion.manifest_splits is None
    assert motion.max_root_lead_m == 0.3 and motion.planar_filter_at_robot_root is True
    p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
    assert p["min_initial_spawn_radius_m"] == SLOW_REGIME_MIN_SPAWN_RADIUS_M
    assert p["speed_bin_edges_mps"] == SLOW_REGIME_SPEED_EDGES_MPS
    assert p["ttc_bin_edges_s"] == SLOW_REGIME_TTC_EDGES_S  # not the dense edges
    assert (p["min_intersection_delay_s"], p["max_intersection_delay_s"]) == (
      SLOW_REGIME_DELAY_RANGE_S
    )
    # Strict stock ee termination, same rewards and crowd as ballet/leash.
    assert (
      cfg.terminations["ee_body_pos"].func is tracking_mdp.bad_motion_body_pos_z_only
    )
    assert set(cfg.rewards) == set(ballet.rewards)
    assert (
      cfg.events[HUMAN_MOTION_EVENT_NAME].params
      == load_env_cfg(
        LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID, play=play
      )
      .events[HUMAN_MOTION_EVENT_NAME]
      .params
    )
  rl = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID)
  assert rl.experiment_name.endswith("coadjust_unified_joint_leash_ballet_slow")
  assert rl.algorithm.avoidance_teacher_mix_decay_updates == 8000


# --- ballet + lag-aware ee termination (2026-09-06) ---------------------------

from safe_mimic.tasks import (  # noqa: E402
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
)
from safe_mimic.tasks.env_cfg import EE_LAG_MAX_THRESHOLD_M, EE_LAG_TIME_S  # noqa: E402


def test_ballet_lag_task_is_ballet_with_the_lag_aware_ee_check() -> None:
  ballet = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID)
  for play in (False, True):
    cfg = load_env_cfg(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID, play=play
    )
    term = cfg.terminations["ee_body_pos"]
    assert term.func is mdp.bad_motion_body_pos_z_only_lag_aware
    assert term.params["threshold"] == 0.25
    assert term.params["lag_time_s"] == EE_LAG_TIME_S == 0.2
    assert term.params["max_threshold"] == EE_LAG_MAX_THRESHOLD_M == 0.6
    assert (
      term.params["body_names"]
      == ballet.terminations["ee_body_pos"].params["body_names"]
    )
    assert cfg.commands["motion"].motion_file == str(DEFAULT_G1_BALLET_MANIFEST)
    p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
    q = ballet.events[PRIMARY_HUMAN_EVENT_NAME].params
    assert p["min_initial_spawn_radius_m"] == q["min_initial_spawn_radius_m"] == 0.75
    assert set(cfg.rewards) == set(ballet.rewards)
  rl = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID)
  assert rl.experiment_name.endswith("coadjust_unified_joint_leash_ballet_lag")


def test_ee_variants_are_mutually_exclusive() -> None:
  import pytest

  with pytest.raises(ValueError, match="one ee_body_pos variant"):
    _leash(filter_gated_ee_termination=True, lag_aware_ee_termination=True)
