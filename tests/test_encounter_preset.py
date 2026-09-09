"""Evaluation encounter presets shared by the benchmark scripts."""

import pytest

from safe_mimic.evaluation import ENCOUNTER_PRESETS, apply_encounter_preset
from safe_mimic.tasks.env_cfg import (
  PRIMARY_HUMAN_EVENT_NAME,
  SLOW_REGIME_MIN_SPAWN_RADIUS_M,
  SLOW_REGIME_SPEED_EDGES_MPS,
  SLOW_REGIME_TTC_EDGES_S,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)


def _cfg():
  return unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=0.3,
    planar_filter_at_robot_root=True,
  )


def test_presets_are_standard_and_slow() -> None:
  assert tuple(ENCOUNTER_PRESETS) == ("standard", "slow")


def test_standard_preset_is_the_frozen_envelope() -> None:
  cfg = apply_encounter_preset(_cfg(), "standard")
  p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
  assert p["encounter_sampling"] == "independent"
  assert (p["min_initial_spawn_radius_m"], p["max_initial_spawn_radius_m"]) == (
    0.75,
    4.0,
  )
  assert (p["min_intersection_delay_s"], p["max_intersection_delay_s"]) == (0.5, 4.0)
  assert cfg.episode_length_s == 6.0


def test_slow_preset_matches_the_slow_training_regime_without_adaptation() -> None:
  cfg = apply_encounter_preset(_cfg(), "slow")
  p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
  assert p["encounter_sampling"] == "ttc"
  assert p["min_initial_spawn_radius_m"] == SLOW_REGIME_MIN_SPAWN_RADIUS_M
  assert p["max_initial_spawn_radius_m"] == 4.0
  assert p["speed_bin_edges_mps"] == SLOW_REGIME_SPEED_EDGES_MPS
  assert p["ttc_bin_edges_s"] == SLOW_REGIME_TTC_EDGES_S
  assert (p["min_intersection_delay_s"], p["max_intersection_delay_s"]) == (
    SLOW_REGIME_TTC_EDGES_S[0],
    SLOW_REGIME_TTC_EDGES_S[-1],
  )
  # A benchmark must not re-weight bins by the policy's own failures, and no
  # curriculum may hide part of the regime.
  assert p["bin_base_weight"] >= 1.0e6
  assert p["curriculum_ramp_s"] <= 1.0e-3
  assert p["hard_speed_above_mps"] > SLOW_REGIME_SPEED_EDGES_MPS[-1]
  assert p["hard_ttc_below_s"] < SLOW_REGIME_TTC_EDGES_S[0]
  # Episodes must outlast the longest intercept.
  assert cfg.episode_length_s >= SLOW_REGIME_TTC_EDGES_S[-1] + 2.0


def test_slow_preset_sampler_stays_in_regime_and_is_uniform_over_bins() -> None:
  import torch

  from safe_mimic.tasks.encounter_sampling import (
    EncounterSampler,
    encounter_sampler_overrides,
  )

  p = apply_encounter_preset(_cfg(), "slow").events[PRIMARY_HUMAN_EVENT_NAME].params
  s = EncounterSampler(
    20000,
    "cpu",
    spawn_radius_clamp_m=(
      p["min_initial_spawn_radius_m"],
      p["max_initial_spawn_radius_m"],
    ),
    **encounter_sampler_overrides(p),
  )
  torch.manual_seed(0)
  ttc, speed, radius = s.sample(
    torch.arange(20000), 0.0, torch.zeros(20000, dtype=torch.bool)
  )
  assert float(speed.max()) <= SLOW_REGIME_SPEED_EDGES_MPS[-1] + 1e-6
  assert float(radius.min()) >= SLOW_REGIME_MIN_SPAWN_RADIUS_M - 1e-6
  weights = s.bin_weights(0.0)
  assert float(weights.max() / weights.min()) < 1.01


def test_unknown_preset_is_rejected() -> None:
  with pytest.raises(ValueError, match="preset"):
    apply_encounter_preset(_cfg(), "fast")


def test_preset_can_keep_the_configs_own_episode_length() -> None:
  cfg = _cfg()
  cfg.episode_length_s = 12345.0
  apply_encounter_preset(cfg, "slow", keep_episode_length=True)
  assert cfg.episode_length_s == 12345.0
  p = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
  assert p["min_initial_spawn_radius_m"] == SLOW_REGIME_MIN_SPAWN_RADIUS_M
  assert p["speed_bin_edges_mps"] == SLOW_REGIME_SPEED_EDGES_MPS
