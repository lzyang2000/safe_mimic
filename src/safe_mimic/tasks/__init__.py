"""Register Safe Mimic tasks through mjlab's task plug-in entry point."""

from dataclasses import fields

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
)
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from safe_mimic.rl import (
  ImplicitStateModelCfg,
  ImplicitStatePpoAlgorithmCfg,
)
from safe_mimic.tasks.env_cfg import (
  unitree_g1_crowd_and_human_tracking_env_cfg,
  unitree_g1_example_dance_implicit_state_tracking_env_cfg,
  unitree_g1_example_dance_tracking_env_cfg,
  unitree_g1_kinematic_reference_lidar_demo_env_cfg,
  unitree_g1_motion_library_implicit_state_tracking_env_cfg,
  unitree_g1_motion_library_tracking_env_cfg,
  unitree_g1_nominal_lidar_debug_env_cfg,
  unitree_g1_obstacle_aware_tracking_env_cfg,
  unitree_g1_reference_filter_lidar_demo_env_cfg,
  unitree_g1_reference_filter_policy_lidar_demo_env_cfg,
)
from safe_mimic.tasks.motion_library_runner import MotionLibraryOnPolicyRunner

TASK_ID = "SafeMimic-Tracking-Obstacles-Unitree-G1-Lidar"
EXAMPLE_DANCE_NO_STATE_EST_TASK_ID = (
  "SafeMimic-Tracking-ExampleDance-NoStateEst-Unitree-G1"
)
EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID = (
  "SafeMimic-Tracking-ExampleDance-ImplicitState-Unitree-G1"
)
MOTION_LIBRARY_TASK_ID = "SafeMimic-Tracking-MotionLib-Unitree-G1"
MOTION_LIBRARY_STATE_EST_TASK_ID = (
  "SafeMimic-Tracking-MotionLib-StateEst-Unitree-G1"
)
MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID = (
  "SafeMimic-Tracking-MotionLib-ImplicitState-Unitree-G1"
)
COMBINED_TASK_ID = "SafeMimic-Tracking-Crowd-Human-Unitree-G1-Lidar"
NOMINAL_LIDAR_DEBUG_TASK_ID = "SafeMimic-Tracking-Flat-Unitree-G1-Lidar-Debug"
REFERENCE_REPLAY_DEMO_TASK_ID = (
  "SafeMimic-Reference-Replay-Crowd-Human-Unitree-G1-Lidar-Demo"
)
REFERENCE_FILTER_DEMO_TASK_ID = (
  "SafeMimic-Reference-Filter-Crowd-Human-Unitree-G1-Lidar-Demo"
)
REFERENCE_FILTER_POLICY_DEMO_TASK_ID = (
  "SafeMimic-Reference-Filter-Policy-Crowd-Human-Unitree-G1-Lidar-Demo"
)


def _safe_mimic_runner_cfg():
  """Create the repository-wide runner defaults for Safe Mimic tasks."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.logger = "tensorboard"
  cfg.save_interval = 1000
  cfg.upload_model = False
  return cfg


def _implicit_state_runner_cfg(experiment_name: str):
  """Create the clean implicit translational-state estimator config."""

  cfg = _safe_mimic_runner_cfg()
  actor_values = {
    field.name: getattr(cfg.actor, field.name)
    for field in fields(cfg.actor)
    if field.name != "class_name"
  }
  algorithm_values = {
    field.name: getattr(cfg.algorithm, field.name)
    for field in fields(cfg.algorithm)
    if field.name != "class_name"
  }
  cfg.actor = ImplicitStateModelCfg(**actor_values)
  cfg.algorithm = ImplicitStatePpoAlgorithmCfg(**algorithm_values)
  cfg.obs_groups["actor"] = ("actor", "proprio_history")
  cfg.experiment_name = experiment_name
  return cfg


rl_cfg = _safe_mimic_runner_cfg()
rl_cfg.experiment_name = "safe_mimic_g1_lidar_tracking"
rl_cfg.actor.hidden_dims = (1024, 512, 256)
rl_cfg.critic.hidden_dims = (1024, 512, 256)

motion_library_rl_cfg = _safe_mimic_runner_cfg()
motion_library_rl_cfg.experiment_name = "safe_mimic_g1_motion_library_tracking"
register_mjlab_task(
  task_id=MOTION_LIBRARY_TASK_ID,
  env_cfg=unitree_g1_motion_library_tracking_env_cfg(),
  play_env_cfg=unitree_g1_motion_library_tracking_env_cfg(play=True),
  rl_cfg=motion_library_rl_cfg,
  # The upstream tracking runner exports one reference trajectory into ONNX.
  # A policy trained over a motion library has no single trajectory to bundle.
  runner_cls=MotionLibraryOnPolicyRunner,
)

motion_library_implicit_state_rl_cfg = _implicit_state_runner_cfg(
  "safe_mimic_g1_motion_library_implicit_state_tracking"
)
register_mjlab_task(
  task_id=MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID,
  env_cfg=unitree_g1_motion_library_implicit_state_tracking_env_cfg(),
  play_env_cfg=unitree_g1_motion_library_implicit_state_tracking_env_cfg(
    play=True
  ),
  rl_cfg=motion_library_implicit_state_rl_cfg,
  runner_cls=MotionLibraryOnPolicyRunner,
)

example_dance_implicit_state_rl_cfg = _implicit_state_runner_cfg(
  "safe_mimic_g1_example_dance_implicit_state_tracking"
)
register_mjlab_task(
  task_id=EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID,
  env_cfg=unitree_g1_example_dance_implicit_state_tracking_env_cfg(),
  play_env_cfg=unitree_g1_example_dance_implicit_state_tracking_env_cfg(
    play=True
  ),
  rl_cfg=example_dance_implicit_state_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

motion_library_state_est_rl_cfg = _safe_mimic_runner_cfg()
motion_library_state_est_rl_cfg.experiment_name = (
  "safe_mimic_g1_motion_library_state_est_tracking"
)
register_mjlab_task(
  task_id=MOTION_LIBRARY_STATE_EST_TASK_ID,
  env_cfg=unitree_g1_motion_library_tracking_env_cfg(has_state_estimation=True),
  play_env_cfg=unitree_g1_motion_library_tracking_env_cfg(
    play=True,
    has_state_estimation=True,
  ),
  rl_cfg=motion_library_state_est_rl_cfg,
  runner_cls=MotionLibraryOnPolicyRunner,
)

example_dance_no_state_est_rl_cfg = _safe_mimic_runner_cfg()
example_dance_no_state_est_rl_cfg.experiment_name = (
  "safe_mimic_g1_example_dance_no_state_est_tracking"
)
register_mjlab_task(
  task_id=EXAMPLE_DANCE_NO_STATE_EST_TASK_ID,
  env_cfg=unitree_g1_example_dance_tracking_env_cfg(),
  play_env_cfg=unitree_g1_example_dance_tracking_env_cfg(play=True),
  rl_cfg=example_dance_no_state_est_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_obstacle_aware_tracking_env_cfg(),
  play_env_cfg=unitree_g1_obstacle_aware_tracking_env_cfg(play=True),
  rl_cfg=rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

combined_rl_cfg = _safe_mimic_runner_cfg()
combined_rl_cfg.experiment_name = "safe_mimic_g1_crowd_human_lidar_tracking"
combined_rl_cfg.actor.hidden_dims = (1024, 512, 256)
combined_rl_cfg.critic.hidden_dims = (1024, 512, 256)
register_mjlab_task(
  task_id=COMBINED_TASK_ID,
  env_cfg=unitree_g1_crowd_and_human_tracking_env_cfg(),
  play_env_cfg=unitree_g1_crowd_and_human_tracking_env_cfg(play=True),
  rl_cfg=combined_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

nominal_rl_cfg = _safe_mimic_runner_cfg()
register_mjlab_task(
  task_id=NOMINAL_LIDAR_DEBUG_TASK_ID,
  env_cfg=unitree_g1_nominal_lidar_debug_env_cfg(),
  play_env_cfg=unitree_g1_nominal_lidar_debug_env_cfg(play=True),
  rl_cfg=nominal_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

reference_replay_rl_cfg = _safe_mimic_runner_cfg()
register_mjlab_task(
  task_id=REFERENCE_REPLAY_DEMO_TASK_ID,
  env_cfg=unitree_g1_kinematic_reference_lidar_demo_env_cfg(),
  play_env_cfg=unitree_g1_kinematic_reference_lidar_demo_env_cfg(play=True),
  rl_cfg=reference_replay_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

reference_filter_rl_cfg = _safe_mimic_runner_cfg()
register_mjlab_task(
  task_id=REFERENCE_FILTER_DEMO_TASK_ID,
  env_cfg=unitree_g1_reference_filter_lidar_demo_env_cfg(),
  play_env_cfg=unitree_g1_reference_filter_lidar_demo_env_cfg(play=True),
  rl_cfg=reference_filter_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

reference_filter_policy_rl_cfg = _safe_mimic_runner_cfg()
register_mjlab_task(
  task_id=REFERENCE_FILTER_POLICY_DEMO_TASK_ID,
  env_cfg=unitree_g1_reference_filter_policy_lidar_demo_env_cfg(),
  play_env_cfg=unitree_g1_reference_filter_policy_lidar_demo_env_cfg(play=True),
  rl_cfg=reference_filter_policy_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

__all__ = [
  "COMBINED_TASK_ID",
  "EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID",
  "EXAMPLE_DANCE_NO_STATE_EST_TASK_ID",
  "MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID",
  "MOTION_LIBRARY_STATE_EST_TASK_ID",
  "MOTION_LIBRARY_TASK_ID",
  "NOMINAL_LIDAR_DEBUG_TASK_ID",
  "REFERENCE_FILTER_DEMO_TASK_ID",
  "REFERENCE_FILTER_POLICY_DEMO_TASK_ID",
  "REFERENCE_REPLAY_DEMO_TASK_ID",
  "TASK_ID",
]
