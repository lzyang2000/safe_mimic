from dataclasses import asdict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from safe_mimic.tasks import (
  COMBINED_TASK_ID,
  EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID,
  EXAMPLE_DANCE_NO_STATE_EST_TASK_ID,
  LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
  LIDAR_AVOIDANCE_TASK_ID,
  LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
  MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID,
  MOTION_LIBRARY_STATE_EST_TASK_ID,
  MOTION_LIBRARY_TASK_ID,
  NOMINAL_LIDAR_DEBUG_TASK_ID,
  REFERENCE_FILTER_DEMO_TASK_ID,
  REFERENCE_FILTER_POLICY_DEMO_TASK_ID,
  REFERENCE_REPLAY_DEMO_TASK_ID,
  SPARSE_LIDAR_AVOIDANCE_TASK_ID,
  TASK_ID,
)
from safe_mimic.tasks.motion_library_runner import MotionLibraryOnPolicyRunner


def test_all_safe_mimic_tasks_use_repository_runner_defaults() -> None:
  task_ids = (
    TASK_ID,
    MOTION_LIBRARY_TASK_ID,
    MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID,
    MOTION_LIBRARY_STATE_EST_TASK_ID,
    EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID,
    EXAMPLE_DANCE_NO_STATE_EST_TASK_ID,
    LIDAR_AVOIDANCE_TASK_ID,
    LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
    LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
    SPARSE_LIDAR_AVOIDANCE_TASK_ID,
    COMBINED_TASK_ID,
    NOMINAL_LIDAR_DEBUG_TASK_ID,
    REFERENCE_REPLAY_DEMO_TASK_ID,
    REFERENCE_FILTER_DEMO_TASK_ID,
    REFERENCE_FILTER_POLICY_DEMO_TASK_ID,
  )

  for task_id in task_ids:
    cfg = load_rl_cfg(task_id)
    assert cfg.logger == "tensorboard", task_id
    assert cfg.save_interval == 1000, task_id
    assert cfg.upload_model is False, task_id


def test_motion_library_task_uses_registry_compatible_runner() -> None:
  assert load_runner_cls(MOTION_LIBRARY_TASK_ID) is MotionLibraryOnPolicyRunner
  assert (
    load_runner_cls(MOTION_LIBRARY_STATE_EST_TASK_ID) is MotionLibraryOnPolicyRunner
  )
  assert (
    load_runner_cls(MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID)
    is MotionLibraryOnPolicyRunner
  )
  assert (
    load_runner_cls(EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID)
    is MotionTrackingOnPolicyRunner
  )


def test_lidar_avoidance_keeps_upstream_ppo_parameters() -> None:
  upstream = unitree_g1_tracking_ppo_runner_cfg()
  cfg = load_rl_cfg(LIDAR_AVOIDANCE_TASK_ID)

  assert cfg.num_steps_per_env == upstream.num_steps_per_env
  assert asdict(cfg.algorithm) == asdict(upstream.algorithm)
  assert cfg.obs_groups["actor"] == ("actor", "lidar")
  assert cfg.obs_groups["critic"] == upstream.obs_groups["critic"]
  assert cfg.actor.class_name == "safe_mimic.rl:PerceptiveLidarActor"
  assert cfg.critic.class_name == upstream.critic.class_name
  assert cfg.actor.azimuth_samples == 185
  assert cfg.actor.elevation_samples == 27
  assert cfg.actor.scan_count == 2
  assert cfg.actor.lidar_direction_bins == 120
  assert cfg.actor.lidar_elevation_bins == 9

  range_rate = load_rl_cfg(LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID)
  assert range_rate.num_steps_per_env == upstream.num_steps_per_env
  assert asdict(range_rate.algorithm) == asdict(upstream.algorithm)
  assert range_rate.obs_groups["actor"] == ("actor", "lidar")
  assert range_rate.actor.scan_count == 2
  assert range_rate.actor.lidar_direction_bins == 120
  assert range_rate.actor.lidar_elevation_bins == 9

  sparse = load_rl_cfg(SPARSE_LIDAR_AVOIDANCE_TASK_ID)
  assert sparse.num_steps_per_env == upstream.num_steps_per_env
  assert asdict(sparse.algorithm) == asdict(upstream.algorithm)
  assert sparse.actor.azimuth_samples == 120
  assert sparse.actor.elevation_samples == 4
  assert sparse.actor.scan_count == 2
  assert sparse.actor.lidar_direction_channels == 8
  assert sparse.actor.lidar_direction_bins == 24
  assert sparse.actor.lidar_elevation_bins == 3


def test_auxiliary_lidar_task_keeps_deployable_inputs_and_upstream_ppo() -> None:
  upstream = unitree_g1_tracking_ppo_runner_cfg()
  cfg = load_rl_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)

  for field_name, upstream_value in asdict(upstream.algorithm).items():
    if field_name != "class_name":
      assert getattr(cfg.algorithm, field_name) == upstream_value
  assert cfg.algorithm.class_name == "safe_mimic.rl:AvoidanceAuxiliaryPPO"
  assert cfg.algorithm.avoidance_active_joint_weight == 12.0
  assert cfg.algorithm.avoidance_teacher_mix_start == 0.1
  assert cfg.algorithm.avoidance_teacher_mix_end == 0.0
  assert cfg.algorithm.avoidance_conditioning_noise_scale == 0.03
  assert cfg.actor.avoidance_planar_dim == 2
  assert cfg.actor.avoidance_joint_dim == 29
  assert cfg.actor.avoidance_joint_action_residual_gain == 0.5
  assert len(cfg.actor.avoidance_joint_action_scales) == 29
  assert len(cfg.actor.avoidance_joint_action_mask) == 29
  assert sum(cfg.actor.avoidance_joint_action_mask) == 14
  assert not any(cfg.actor.avoidance_joint_action_mask[:15])
  assert all(cfg.actor.avoidance_joint_action_mask[15:])
  assert cfg.obs_groups["actor"] == ("actor", "lidar")
  assert "avoidance_teacher" not in cfg.obs_groups["actor"]

  env_cfg = load_env_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)
  assert tuple(env_cfg.observations["avoidance_teacher"].terms) == (
    "corrections",
  )
  assert env_cfg.observations["avoidance_teacher"].enable_corruption is False
  assert tuple(env_cfg.observations["avoidance_robustness"].terms) == (
    "conditioning_noise",
  )
  assert env_cfg.observations["avoidance_robustness"].enable_corruption is False
  urgent_escape = env_cfg.rewards["urgent_escape_progress"]
  assert urgent_escape.weight == 1.0
  assert urgent_escape.params == {
    "command_name": "motion",
    "human_entity": "primary_human",
  }
  range_rate_env_cfg = load_env_cfg(LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID)
  assert "urgent_escape_progress" not in range_rate_env_cfg.rewards


def test_implicit_state_tasks_are_clean_no_position_ablations() -> None:
  for task_id in (
    MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID,
    EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID,
  ):
    env_cfg = load_env_cfg(task_id)
    actor_terms = env_cfg.observations["actor"].terms
    assert "motion_anchor_pos_b" not in actor_terms
    assert "base_lin_vel" not in actor_terms
    assert "proprio_history" in env_cfg.observations
    assert "implicit_state_target" in env_cfg.observations
    assert "implicit_proprio_target" in env_cfg.observations
    target_terms = env_cfg.observations["implicit_state_target"].terms
    assert tuple(target_terms) == ("base_lin_vel", "motion_anchor_pos_b")
    assert env_cfg.observations["proprio_history"].history_length == 10
    assert env_cfg.observations["implicit_proprio_target"].history_length is None

    rl_cfg = load_rl_cfg(task_id)
    assert rl_cfg.obs_groups["actor"] == ("actor", "proprio_history")
    assert rl_cfg.actor.class_name == "safe_mimic.rl:ImplicitStateActor"
    assert rl_cfg.actor.dynamics_latent_dim == 16
    assert rl_cfg.actor.num_dynamics_prototypes == 32
    assert rl_cfg.algorithm.class_name == "safe_mimic.rl:ImplicitStatePPO"
    assert rl_cfg.algorithm.implicit_dynamics_loss_coef == 1.0
