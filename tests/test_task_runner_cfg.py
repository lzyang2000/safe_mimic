from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from safe_mimic.tasks import (
  COMBINED_TASK_ID,
  EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID,
  EXAMPLE_DANCE_NO_STATE_EST_TASK_ID,
  MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID,
  MOTION_LIBRARY_STATE_EST_TASK_ID,
  MOTION_LIBRARY_TASK_ID,
  NOMINAL_LIDAR_DEBUG_TASK_ID,
  REFERENCE_FILTER_DEMO_TASK_ID,
  REFERENCE_FILTER_POLICY_DEMO_TASK_ID,
  REFERENCE_REPLAY_DEMO_TASK_ID,
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
    load_runner_cls(MOTION_LIBRARY_STATE_EST_TASK_ID)
    is MotionLibraryOnPolicyRunner
  )
  assert (
    load_runner_cls(MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID)
    is MotionLibraryOnPolicyRunner
  )
  assert (
    load_runner_cls(EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID)
    is MotionTrackingOnPolicyRunner
  )


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
