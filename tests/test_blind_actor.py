"""Blind baseline: the actor's LiDAR term reads 'no returns' forever."""

from types import SimpleNamespace

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from safe_mimic.sensing.observations import (
  BlindDirectionalHeldLidarRangeRate,
  CachedDirectionalHeldLidarRangeRate,
)
from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
)
from safe_mimic.tasks.env_cfg import unitree_g1_lidar_unified_reference_tracking_env_cfg


def _term(azimuth_bins: int, elevation_bins: int) -> BlindDirectionalHeldLidarRangeRate:
  term = BlindDirectionalHeldLidarRangeRate.__new__(BlindDirectionalHeldLidarRangeRate)
  term.azimuth_bins = azimuth_bins
  term.elevation_bins = elevation_bins
  return term


def test_blind_term_is_the_benchmarks_no_return_pattern() -> None:
  term = _term(120, 9)
  env = SimpleNamespace(num_envs=3, device="cpu")
  out = term(env)
  cells = 120 * 9
  assert out.shape == (3, 2 * cells)
  assert torch.all(out[:, :cells] == 1.0)  # every range at maximum
  assert torch.all(out[:, cells:] == 0.0)  # every range rate zero


def test_blind_term_is_a_drop_in_for_the_sighted_one() -> None:
  # Same class family, so scripts reading azimuth_bins / params keep working,
  # and reset() is inherited harmlessly.
  assert issubclass(
    BlindDirectionalHeldLidarRangeRate, CachedDirectionalHeldLidarRangeRate
  )
  term = _term(4, 2)
  term._cached = None
  term.reset(None)


def test_blind_actor_flag_swaps_only_the_lidar_term_function() -> None:
  sighted = unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True, root_lead_m=0.3, planar_filter_at_robot_root=True
  )
  blind = unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=0.3,
    planar_filter_at_robot_root=True,
    blind_actor=True,
  )
  s_term = sighted.observations["lidar"].terms["directional_range_rate"]
  b_term = blind.observations["lidar"].terms["directional_range_rate"]
  assert s_term.func is CachedDirectionalHeldLidarRangeRate
  assert b_term.func is BlindDirectionalHeldLidarRangeRate
  assert b_term.params == s_term.params  # same bins, same noise cfg slot
  assert tuple(blind.observations["lidar"].terms) == tuple(
    sighted.observations["lidar"].terms
  )
  # Critic keeps its privileged LiDAR and human vectors: the baseline is
  # "the same training signal, without the actor's eyes".
  assert set(blind.observations["critic"].terms) == set(
    sighted.observations["critic"].terms
  )
  assert set(blind.rewards) == set(sighted.rewards)


def test_ballet_blind_task_registered() -> None:
  ballet = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID)
  for play in (False, True):
    cfg = load_env_cfg(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID, play=play
    )
    assert (
      cfg.observations["lidar"].terms["directional_range_rate"].func
      is BlindDirectionalHeldLidarRangeRate
    )
    assert cfg.commands["motion"].motion_file == ballet.commands["motion"].motion_file
    assert (
      cfg.terminations["ee_body_pos"].func is ballet.terminations["ee_body_pos"].func
    )
    assert set(cfg.rewards) == set(ballet.rewards)
  assert (
    ballet.observations["lidar"].terms["directional_range_rate"].func
    is CachedDirectionalHeldLidarRangeRate
  )
  rl = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID)
  assert rl.experiment_name.endswith("coadjust_unified_joint_leash_ballet_blind")
  assert rl.actor.adjust_command_with_joint_prediction is True


# --- nominal-reference blind baseline (2026-09-08) ---------------------------

from safe_mimic.tasks import (  # noqa: E402
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
)

NOMINAL_ID = LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID
BLIND_ID = LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID


def test_nominal_reference_flag_disables_the_filters_only() -> None:
  blind = unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=0.3,
    planar_filter_at_robot_root=True,
    blind_actor=True,
  )
  nominal = unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=0.3,
    planar_filter_at_robot_root=True,
    blind_actor=True,
    nominal_reference=True,
  )
  assert blind.commands["motion"].disable_filters is False
  assert nominal.commands["motion"].disable_filters is True
  assert set(nominal.rewards) == set(blind.rewards)
  assert set(nominal.terminations) == set(blind.terminations)
  assert tuple(nominal.observations["actor"].terms) == tuple(
    blind.observations["actor"].terms
  )
  assert (
    nominal.observations["lidar"].terms["directional_range_rate"].func
    is BlindDirectionalHeldLidarRangeRate
  )


def test_ballet_blind_nominal_task_registered() -> None:
  ballet = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID)
  blind = load_env_cfg(
    LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID
  )
  # The earlier blind baseline keeps its filtered reference.
  assert blind.commands["motion"].disable_filters is False
  for play in (False, True):
    cfg = load_env_cfg(NOMINAL_ID, play=play)
    motion = cfg.commands["motion"]
    assert motion.disable_filters is True
    assert motion.motion_file == ballet.commands["motion"].motion_file
    assert (
      cfg.observations["lidar"].terms["directional_range_rate"].func
      is BlindDirectionalHeldLidarRangeRate
    )
  rl = load_rl_cfg(NOMINAL_ID)
  assert rl.experiment_name.endswith("leash_ballet_blind_nominal")


# --- no humans in training (2026-09-08) --------------------------------------

from safe_mimic.tasks import (  # noqa: E402
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
)
from safe_mimic.tasks.env_cfg import (  # noqa: E402
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
)

NOHUMANS_ID = LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID
HUMAN_EVENTS = (HUMAN_MOTION_EVENT_NAME, PRIMARY_HUMAN_EVENT_NAME)
HUMAN_TERMINATIONS = ("crowd_collision", "primary_human_collision")


def _blind_nominal(**extra):
  return unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=0.3,
    planar_filter_at_robot_root=True,
    blind_actor=True,
    nominal_reference=True,
    **extra,
  )


def test_training_humans_off_removes_only_the_human_events_and_terminations() -> None:
  ref = _blind_nominal()
  cfg = _blind_nominal(training_humans=False)
  for name in HUMAN_EVENTS:
    assert name in ref.events and name not in cfg.events
  for name in HUMAN_TERMINATIONS:
    assert name in ref.terminations and name not in cfg.terminations
  assert set(cfg.events) == set(ref.events) - set(HUMAN_EVENTS)
  assert set(cfg.terminations) == set(ref.terminations) - set(HUMAN_TERMINATIONS)
  # Everything the network touches is identical, so the play scene can add
  # the humans back without changing observation shapes.
  assert set(cfg.rewards) == set(ref.rewards)
  for group in ref.observations:
    assert tuple(cfg.observations[group].terms) == tuple(ref.observations[group].terms)
  assert cfg.commands["motion"].disable_filters is True


def test_training_humans_off_is_ignored_for_play() -> None:
  play = _blind_nominal(play=True, training_humans=False)
  for name in HUMAN_EVENTS:
    assert name in play.events
  for name in HUMAN_TERMINATIONS:
    assert name in play.terminations


def test_ballet_blind_nohumans_task_registered() -> None:
  train = load_env_cfg(NOHUMANS_ID)
  play = load_env_cfg(NOHUMANS_ID, play=True)
  for name in HUMAN_EVENTS:
    assert name not in train.events and name in play.events
  for name in HUMAN_TERMINATIONS:
    assert name not in train.terminations and name in play.terminations
  for cfg in (train, play):
    assert cfg.commands["motion"].disable_filters is True
    assert (
      cfg.observations["lidar"].terms["directional_range_rate"].func
      is BlindDirectionalHeldLidarRangeRate
    )
  rl = load_rl_cfg(NOHUMANS_ID)
  assert rl.experiment_name.endswith("leash_ballet_blind_nohumans")
