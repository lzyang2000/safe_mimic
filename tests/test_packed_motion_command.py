from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)

from safe_mimic.tasks.env_cfg import (
  DEFAULT_EXAMPLE_DANCE_MOTION_FILE,
  DEFAULT_G1_MOTION_LIBRARY_MANIFEST,
  LIDAR_SENSOR_NAME,
  unitree_g1_example_dance_implicit_state_tracking_env_cfg,
  unitree_g1_example_dance_tracking_env_cfg,
  unitree_g1_motion_library_tracking_env_cfg,
)
from safe_mimic.tasks.packed_motion_command import (
  PackedMotionCommand,
  PackedMotionCommandCfg,
)


def test_motion_library_env_is_raw_upstream_mimic_without_lidar() -> None:
  upstream = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)
  cfg = unitree_g1_motion_library_tracking_env_cfg()

  assert cfg.scene.entities.keys() == upstream.scene.entities.keys()
  assert cfg.actions.keys() == upstream.actions.keys()
  assert cfg.observations.keys() == upstream.observations.keys()
  assert cfg.rewards.keys() == upstream.rewards.keys()
  assert cfg.terminations.keys() == upstream.terminations.keys()
  assert cfg.events.keys() == upstream.events.keys()
  assert all(sensor.name != LIDAR_SENSOR_NAME for sensor in cfg.scene.sensors or ())
  for group_name in upstream.observations:
    assert (
      cfg.observations[group_name].terms.keys()
      == upstream.observations[group_name].terms.keys()
    )
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "motion_anchor_pos_b" not in cfg.observations["actor"].terms
  assert "base_ang_vel" in cfg.observations["actor"].terms
  assert "motion_anchor_ori_b" in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  assert "motion_anchor_pos_b" in cfg.observations["critic"].terms

  motion = cfg.commands["motion"]
  assert isinstance(motion, PackedMotionCommandCfg)
  assert Path(motion.motion_file) == DEFAULT_G1_MOTION_LIBRARY_MANIFEST
  assert DEFAULT_G1_MOTION_LIBRARY_MANIFEST.is_file()
  assert motion.manifest_splits == ("train",)
  assert motion.sampling_mode == "adaptive"
  assert motion.adaptive_bin_duration_s == 1.0
  assert motion.adaptive_alpha == 0.05
  assert motion.adaptive_uniform_ratio == 0.1
  assert motion.adaptive_couple_pairs


def test_motion_library_play_keeps_upstream_play_randomization_overrides() -> None:
  upstream = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False,
    play=True,
  )
  cfg = unitree_g1_motion_library_tracking_env_cfg(play=True)
  motion = cfg.commands["motion"]

  assert isinstance(motion, PackedMotionCommandCfg)
  assert motion.sampling_mode == "start"
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.joint_position_range == upstream.commands[
    "motion"
  ].joint_position_range
  assert "push_robot" not in cfg.events
  assert not cfg.observations["actor"].enable_corruption


def test_motion_library_state_estimation_restores_translational_actor_terms() -> None:
  upstream = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True)
  cfg = unitree_g1_motion_library_tracking_env_cfg(has_state_estimation=True)

  assert (
    cfg.observations["actor"].terms.keys()
    == upstream.observations["actor"].terms.keys()
  )
  assert "base_lin_vel" in cfg.observations["actor"].terms
  assert "motion_anchor_pos_b" in cfg.observations["actor"].terms


def test_example_dance_task_is_single_motion_without_state_estimation() -> None:
  cfg = unitree_g1_example_dance_tracking_env_cfg()

  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "motion_anchor_pos_b" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  assert "motion_anchor_pos_b" in cfg.observations["critic"].terms

  motion = cfg.commands["motion"]
  assert not isinstance(motion, PackedMotionCommandCfg)
  assert Path(motion.motion_file) == DEFAULT_EXAMPLE_DANCE_MOTION_FILE


def test_example_dance_implicit_state_task_stays_single_motion() -> None:
  cfg = unitree_g1_example_dance_implicit_state_tracking_env_cfg()

  motion = cfg.commands["motion"]
  assert not isinstance(motion, PackedMotionCommandCfg)
  assert Path(motion.motion_file) == DEFAULT_EXAMPLE_DANCE_MOTION_FILE
  assert "proprio_history" in cfg.observations
  assert "implicit_state_target" in cfg.observations
  assert "implicit_proprio_target" in cfg.observations


def test_example_dance_play_starts_at_the_beginning_without_corruption() -> None:
  cfg = unitree_g1_example_dance_tracking_env_cfg(play=True)
  motion = cfg.commands["motion"]

  assert motion.sampling_mode == "start"
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert not cfg.observations["actor"].enable_corruption


def test_start_sampling_cycles_independently_per_environment() -> None:
  command = object.__new__(PackedMotionCommand)
  command.cfg = SimpleNamespace(sampling_mode="start")
  command._env = SimpleNamespace(device="cpu")
  command._play_cursor = torch.tensor([0, 3, 6])
  command.motion = SimpleNamespace(num_motions=lambda: 8)
  env_ids = torch.tensor([0, 2])

  first = command._sample_references(env_ids)
  second = command._sample_references(env_ids)

  assert first.motion_ids.tolist() == [0, 6]
  assert second.motion_ids.tolist() == [1, 7]
  assert torch.equal(first.motion_times, torch.zeros(2))


def test_adaptive_outcome_update_ignores_initial_unsampled_environments() -> None:
  recorded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
  command = object.__new__(PackedMotionCommand)
  command.cfg = SimpleNamespace(sampling_mode="adaptive")
  command._has_reference = torch.tensor([False, True, True])
  command.motion_ids = torch.tensor([10, 11, 12])
  command.motion_times = torch.tensor([0.1, 0.2, 0.3])
  command._env = SimpleNamespace(
    termination_manager=SimpleNamespace(
      terminated=torch.tensor([False, True, False])
    )
  )
  command.adaptive_sampler = SimpleNamespace(
    update=lambda *args: recorded.append(args)
  )

  command._record_finished_references(torch.tensor([0, 1, 2]))

  assert len(recorded) == 1
  failures, motion_ids, motion_times = recorded[0]
  assert failures.tolist() == [True, False]
  assert motion_ids.tolist() == [11, 12]
  assert torch.allclose(motion_times, torch.tensor([0.2, 0.3]))
