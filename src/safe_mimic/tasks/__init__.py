"""Register Safe Mimic tasks through mjlab's task plug-in entry point."""

import re
from dataclasses import fields

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
)
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from safe_mimic.rl import (
  AvoidanceAuxiliaryPpoAlgorithmCfg,
  ImplicitStateModelCfg,
  ImplicitStatePpoAlgorithmCfg,
  PerceptiveLidarModelCfg,
)
from safe_mimic.tasks.env_cfg import (
  DEFAULT_G1_BALLET_MANIFEST,
  DEFAULT_G1_BALLET_MIRROR_MANIFEST,
  unitree_g1_crowd_and_human_tracking_env_cfg,
  unitree_g1_example_dance_implicit_state_tracking_env_cfg,
  unitree_g1_example_dance_tracking_env_cfg,
  unitree_g1_kinematic_reference_lidar_demo_env_cfg,
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
  unitree_g1_lidar_avoidance_tracking_env_cfg,
  unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
  unitree_g1_motion_library_implicit_state_tracking_env_cfg,
  unitree_g1_motion_library_tracking_env_cfg,
  unitree_g1_nominal_lidar_debug_env_cfg,
  unitree_g1_obstacle_aware_tracking_env_cfg,
  unitree_g1_reference_filter_lidar_demo_env_cfg,
  unitree_g1_reference_filter_policy_lidar_demo_env_cfg,
  unitree_g1_sparse_lidar_avoidance_tracking_env_cfg,
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
MOTION_LIBRARY_STATE_EST_TASK_ID = "SafeMimic-Tracking-MotionLib-StateEst-Unitree-G1"
MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID = (
  "SafeMimic-Tracking-MotionLib-ImplicitState-Unitree-G1"
)
COMBINED_TASK_ID = "SafeMimic-Tracking-Crowd-Human-Unitree-G1-Lidar"
LIDAR_AVOIDANCE_TASK_ID = "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar"
LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-RangeRate"
)
LIDAR_AUXILIARY_AVOIDANCE_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary"
)
LIDAR_AUXILIARY_EXPOSED_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-Exposed"
)
LIDAR_AUXILIARY_COADJUST_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust"
)
LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-FKC"
)
LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-FKC2"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Slow"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Slow-Dense"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet-Slow"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet-Lag"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet-Blind"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet-Blind-Nominal"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet-Blind-NoHumans"
)
LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet-Moves"
)
UNIFIED_ROOT_LEAD_M = 0.3
SPARSE_LIDAR_AVOIDANCE_TASK_ID = (
  "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Sparse"
)
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

# MjLab's G1 joint-position action uses natural MuJoCo joint order. The
# privileged joint-filter target follows the same order, so the actor can
# convert radians into normalized actions without simulator-only inputs.
_G1_ACTION_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def _g1_action_scale(joint_name: str) -> float:
  matches = (
    float(scale)
    for pattern, scale in G1_ACTION_SCALE.items()
    if re.fullmatch(pattern, joint_name)
  )
  values = tuple(matches)
  if len(values) != 1:
    raise RuntimeError(f"expected one G1 action scale for {joint_name!r}")
  return values[0]


_G1_ACTION_SCALES = tuple(
  _g1_action_scale(joint_name) for joint_name in _G1_ACTION_JOINT_NAMES
)
_G1_ARM_ACTION_MASK = tuple(
  any(token in joint_name for token in ("shoulder", "elbow", "wrist"))
  for joint_name in _G1_ACTION_JOINT_NAMES
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


def _perceptive_lidar_runner_cfg(
  *,
  azimuth_samples: int = 185,
  elevation_samples: int = 27,
  direction_bins: int = 120,
  elevation_bins: int = 9,
  auxiliary_supervision: bool = False,
  experiment_name: str = "safe_mimic_g1_live_aligned_lidar_avoidance",
):
  """Select the perceptive actor without changing upstream PPO parameters."""
  cfg = _safe_mimic_runner_cfg()
  actor_values = {
    field.name: getattr(cfg.actor, field.name)
    for field in fields(cfg.actor)
    if field.name != "class_name"
  }
  cfg.actor = PerceptiveLidarModelCfg(**actor_values)
  cfg.actor.azimuth_samples = azimuth_samples
  cfg.actor.elevation_samples = elevation_samples
  cfg.actor.lidar_direction_bins = direction_bins
  cfg.actor.lidar_elevation_bins = elevation_bins
  if auxiliary_supervision:
    algorithm_values = {
      field.name: getattr(cfg.algorithm, field.name)
      for field in fields(cfg.algorithm)
      if field.name != "class_name"
    }
    cfg.algorithm = AvoidanceAuxiliaryPpoAlgorithmCfg(**algorithm_values)
    cfg.actor.avoidance_planar_dim = 2
    cfg.actor.avoidance_joint_dim = 29
    # The tanh head must cover the new 2.0 m/s planar intervention cap.
    cfg.actor.avoidance_planar_output_scale = 2.0
    # Planar compass was the weakest prediction in diagnostics.
    cfg.algorithm.avoidance_planar_loss_coef = 2.0
    # The teacher is a safe-reference delta in radians. Apply half of the
    # predicted arm delta explicitly to the normalized joint command so the
    # downstream policy MLP cannot simply ignore the filter signal. Legs and
    # waist remain learned-only until this arm path is validated in play.
    cfg.actor.avoidance_joint_action_residual_gain = 0.5
    cfg.actor.avoidance_joint_action_scales = _G1_ACTION_SCALES
    cfg.actor.avoidance_joint_action_mask = _G1_ARM_ACTION_MASK
  cfg.obs_groups["actor"] = ("actor", "lidar")
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
  play_env_cfg=unitree_g1_motion_library_implicit_state_tracking_env_cfg(play=True),
  rl_cfg=motion_library_implicit_state_rl_cfg,
  runner_cls=MotionLibraryOnPolicyRunner,
)

example_dance_implicit_state_rl_cfg = _implicit_state_runner_cfg(
  "safe_mimic_g1_example_dance_implicit_state_tracking"
)
register_mjlab_task(
  task_id=EXAMPLE_DANCE_IMPLICIT_STATE_TASK_ID,
  env_cfg=unitree_g1_example_dance_implicit_state_tracking_env_cfg(),
  play_env_cfg=unitree_g1_example_dance_implicit_state_tracking_env_cfg(play=True),
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

lidar_avoidance_rl_cfg = _perceptive_lidar_runner_cfg()
register_mjlab_task(
  task_id=LIDAR_AVOIDANCE_TASK_ID,
  env_cfg=unitree_g1_lidar_avoidance_tracking_env_cfg(),
  play_env_cfg=unitree_g1_lidar_avoidance_tracking_env_cfg(play=True),
  rl_cfg=lidar_avoidance_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_range_rate_avoidance_rl_cfg = _perceptive_lidar_runner_cfg(
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_range_rate_link_reward"
  ),
)
register_mjlab_task(
  task_id=LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID,
  env_cfg=unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg(),
  play_env_cfg=unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg(play=True),
  rl_cfg=lidar_range_rate_avoidance_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_avoidance_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_ttc_scratch"
  ),
)
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
  env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(),
  play_env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=True),
  rl_cfg=lidar_auxiliary_avoidance_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_exposed_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_ttc_exposed_ft"
  ),
)
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_EXPOSED_TASK_ID,
  env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    expose_filtered_command=True
  ),
  play_env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    play=True, expose_filtered_command=True
  ),
  rl_cfg=lidar_auxiliary_exposed_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust"
  ),
)
# Route the predicted joint correction into the observed motion command
# (PerceptiveLidarActor._command_adjusted_latent) instead of a residual added
# to the mimic policy's own action output.
lidar_auxiliary_coadjust_rl_cfg.actor.adjust_command_with_joint_prediction = True
# Derivation (see .superpowers/sdd/2026-09-01-phase2b-cotrained-limb-adjuster
# /task-2-report.md for the introspection run that confirmed this against the
# live ObservationManager):
# unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg's actor observation
# group keeps mjlab's upstream tracking term order (command, motion_anchor_pos_b,
# motion_anchor_ori_b, base_lin_vel, base_ang_vel, joint_pos, joint_vel,
# actions), then unitree_g1_lidar_avoidance_tracking_env_cfg pops
# "motion_anchor_pos_b", "base_lin_vel", and the LiDAR term, leaving order
# ["command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel",
# "actions"] with "command" first. That term is
# KinematicReplayMotionCommand.command == cat(joint_pos[29], joint_vel[29])
# (see kinematic_replay_command.py), so its first joint-position value is the
# very first value of the flattened actor vector.
lidar_auxiliary_coadjust_rl_cfg.actor.command_joint_pos_offset = 0
# The command adjustment is the only avoidance pathway trained here; do not
# also add a residual to the policy's own action output.
lidar_auxiliary_coadjust_rl_cfg.actor.avoidance_joint_action_residual_gain = 0.0
lidar_auxiliary_coadjust_rl_cfg.algorithm.avoidance_teacher_mix_start = 1.0
lidar_auxiliary_coadjust_rl_cfg.algorithm.avoidance_teacher_mix_end = 0.0
lidar_auxiliary_coadjust_rl_cfg.algorithm.avoidance_teacher_mix_decay_updates = 8000
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_TASK_ID,
  env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(),
  play_env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=True),
  rl_cfg=lidar_auxiliary_coadjust_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_fkc_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_fkc"
  ),
)
# Identical command-injection setup to LIDAR_AUXILIARY_COADJUST_TASK_ID; see
# that registration's comments above for the derivation of these values.
lidar_auxiliary_coadjust_fkc_rl_cfg.actor.adjust_command_with_joint_prediction = True
lidar_auxiliary_coadjust_fkc_rl_cfg.actor.command_joint_pos_offset = 0
lidar_auxiliary_coadjust_fkc_rl_cfg.actor.avoidance_joint_action_residual_gain = 0.0
lidar_auxiliary_coadjust_fkc_rl_cfg.algorithm.avoidance_teacher_mix_start = 1.0
lidar_auxiliary_coadjust_fkc_rl_cfg.algorithm.avoidance_teacher_mix_end = 0.0
lidar_auxiliary_coadjust_fkc_rl_cfg.algorithm.avoidance_teacher_mix_decay_updates = 8000
# FK-consistent objective: the motion command's arm body position/orientation
# targets follow the filtered joint corrections, so the task-space reward and
# ee_body_pos termination stop opposing the co-adjusted arm motion.
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
  env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    propagate_arm_corrections_to_body_targets=True
  ),
  play_env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    play=True, propagate_arm_corrections_to_body_targets=True
  ),
  rl_cfg=lidar_auxiliary_coadjust_fkc_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_fkc2_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_fkc2"
  ),
)
# Identical command-injection setup to LIDAR_AUXILIARY_COADJUST_TASK_ID; see
# that registration's comments above for the derivation of these values.
lidar_auxiliary_coadjust_fkc2_rl_cfg.actor.adjust_command_with_joint_prediction = True
lidar_auxiliary_coadjust_fkc2_rl_cfg.actor.command_joint_pos_offset = 0
lidar_auxiliary_coadjust_fkc2_rl_cfg.actor.avoidance_joint_action_residual_gain = 0.0
lidar_auxiliary_coadjust_fkc2_rl_cfg.algorithm.avoidance_teacher_mix_start = 1.0
lidar_auxiliary_coadjust_fkc2_rl_cfg.algorithm.avoidance_teacher_mix_end = 0.0
lidar_auxiliary_coadjust_fkc2_rl_cfg.algorithm.avoidance_teacher_mix_decay_updates = (
  8000
)
# Same FK-consistent objective as LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID, plus
# the active-correction reward: precise tracking of the filtered pose only
# where the teacher residual is active, instead of diluting the signal over
# all 29 joints like "filtered_joint_position" does.
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    propagate_arm_corrections_to_body_targets=True,
    active_correction_reward=True,
  ),
  play_env_cfg=unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    play=True,
    propagate_arm_corrections_to_body_targets=True,
    active_correction_reward=True,
  ),
  rl_cfg=lidar_auxiliary_coadjust_fkc2_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified"
  ),
)
# Identical command-injection setup to LIDAR_AUXILIARY_COADJUST_TASK_ID; see
# that registration's comments above for the derivation of these values.
lidar_auxiliary_coadjust_unified_rl_cfg.actor.adjust_command_with_joint_prediction = (
  True
)
lidar_auxiliary_coadjust_unified_rl_cfg.actor.command_joint_pos_offset = 0
lidar_auxiliary_coadjust_unified_rl_cfg.actor.avoidance_joint_action_residual_gain = (
  0.0
)
_unified_algorithm = lidar_auxiliary_coadjust_unified_rl_cfg.algorithm
_unified_algorithm.avoidance_teacher_mix_start = 1.0
_unified_algorithm.avoidance_teacher_mix_end = 0.0
_unified_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_algorithm
# Fully filtered reference: the motion command's whole-body targets (anchor
# included) follow the filtered joint corrections and the filtered root is
# integrated closed-loop, so the stock mjlab tracking reward set and
# terminations evaluate one complete privileged-filtered reference.
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(play=True),
  rl_cfg=lidar_auxiliary_coadjust_unified_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint"
  ),
)
# Identical command-injection setup to LIDAR_AUXILIARY_COADJUST_TASK_ID.
_unified_joint_actor = lidar_auxiliary_coadjust_unified_joint_rl_cfg.actor
_unified_joint_actor.adjust_command_with_joint_prediction = True
_unified_joint_actor.command_joint_pos_offset = 0
_unified_joint_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_joint_actor
_unified_joint_algorithm = lidar_auxiliary_coadjust_unified_joint_rl_cfg.algorithm
_unified_joint_algorithm.avoidance_teacher_mix_start = 1.0
_unified_joint_algorithm.avoidance_teacher_mix_end = 0.0
_unified_joint_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_joint_algorithm
# Unified reference plus ONE joint-space term on the actively corrected
# joints (user direction 2026-09-03): the nominal set alone moved the pelvis
# but left arm corrections without gradient.
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True, active_joint_reward=True
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_leash_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash"
  ),
)
_unified_leash_actor = lidar_auxiliary_coadjust_unified_joint_leash_rl_cfg.actor
_unified_leash_actor.adjust_command_with_joint_prediction = True
_unified_leash_actor.command_joint_pos_offset = 0
_unified_leash_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_leash_actor
_unified_leash_algorithm = lidar_auxiliary_coadjust_unified_joint_leash_rl_cfg.algorithm
_unified_leash_algorithm.avoidance_teacher_mix_start = 1.0
_unified_leash_algorithm.avoidance_teacher_mix_end = 0.0
_unified_leash_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_leash_algorithm
# Unified-Joint plus a leashed root target (<= 0.3 m ahead of the robot, the
# region where the std-0.3 root reward still has gradient) and the planar CBF
# evaluated at the robot (root-tracking diagnosis 2026-09-03: the unleashed
# target outran the robot and the filter then stopped pushing).
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_leash_slow_rl_cfg = _perceptive_lidar_runner_cfg(
  auxiliary_supervision=True,
  experiment_name=(
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_slow"
  ),
)
_unified_slow_actor = lidar_auxiliary_coadjust_unified_joint_leash_slow_rl_cfg.actor
_unified_slow_actor.adjust_command_with_joint_prediction = True
_unified_slow_actor.command_joint_pos_offset = 0
_unified_slow_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_slow_actor
_unified_slow_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_slow_rl_cfg.algorithm
)
_unified_slow_algorithm.avoidance_teacher_mix_start = 1.0
_unified_slow_algorithm.avoidance_teacher_mix_end = 0.0
_unified_slow_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_slow_algorithm
# Leash plus the slow-regime encounter distribution (walking human spawns
# outside the critical distance, approach <= 0.75 m/s) and the filter-gated
# ee_body_pos termination (user direction 2026-09-04).
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    slow_regime=True,
    filter_gated_ee_termination=True,
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    slow_regime=True,
    filter_gated_ee_termination=True,
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_slow_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_leash_slow_dense_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_slow_dense"
    ),
  )
)
_unified_dense_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_slow_dense_rl_cfg.actor
)
_unified_dense_actor.adjust_command_with_joint_prediction = True
_unified_dense_actor.command_joint_pos_offset = 0
_unified_dense_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_dense_actor
_unified_dense_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_slow_dense_rl_cfg.algorithm
)
_unified_dense_algorithm.avoidance_teacher_mix_start = 1.0
_unified_dense_algorithm.avoidance_teacher_mix_end = 0.0
_unified_dense_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_dense_algorithm
# Leash-Slow with denser encounters (5 s TTC cap) and the STRICT stock
# ee_body_pos termination again: the gated version regressed arm compliance
# (slow@15k gate, 2026-09-04). Obstacle-free probability unchanged (user).
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    slow_regime=True,
    dense_encounters=True,
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    slow_regime=True,
    dense_encounters=True,
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_slow_dense_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_leash_ballet_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_ballet"
    ),
  )
)
_unified_ballet_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_rl_cfg.actor
)
_unified_ballet_actor.adjust_command_with_joint_prediction = True
_unified_ballet_actor.command_joint_pos_offset = 0
_unified_ballet_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_ballet_actor
_unified_ballet_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_rl_cfg.algorithm
)
_unified_ballet_algorithm.avoidance_teacher_mix_start = 1.0
_unified_ballet_algorithm.avoidance_teacher_mix_end = 0.0
_unified_ballet_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_ballet_algorithm
# CANONICAL setup (user, 2026-09-05) = the Leash task: nominal rewards + one
# active-joint term, strict stock terminations, leashed closed-loop root
# target with the planar CBF at the robot, full 0.75-4 m / 0.5-4 s encounter
# range. Ballet swaps ONLY the reference for the whole G1 ballet library;
# clips chain when they end and the robot continues from its current pose.
# (The slow / dense narrowings evaluated worse than or equal to Leash.)
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_ballet_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_leash_ballet_slow_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_ballet_slow"
    ),
  )
)
_unified_bslow_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_slow_rl_cfg.actor
)
_unified_bslow_actor.adjust_command_with_joint_prediction = True
_unified_bslow_actor.command_joint_pos_offset = 0
_unified_bslow_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_bslow_actor
_unified_bslow_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_slow_rl_cfg.algorithm
)
_unified_bslow_algorithm.avoidance_teacher_mix_start = 1.0
_unified_bslow_algorithm.avoidance_teacher_mix_end = 0.0
_unified_bslow_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_bslow_algorithm
# Ballet reference + the slow encounter regime (user direction 2026-09-06):
# human spawns >= 1.8 m away and approaches at <= 0.75 m/s, TTC 2.5-8 s.
# Everything else identical to Leash-Ballet (strict ee termination, full
# crowd). Isolates "slower humans" on top of the best lineage so far.
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    slow_regime=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    slow_regime=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_ballet_slow_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_leash_ballet_lag_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_ballet_lag"
    ),
  )
)
_unified_blag_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_lag_rl_cfg.actor
)
_unified_blag_actor.adjust_command_with_joint_prediction = True
_unified_blag_actor.command_joint_pos_offset = 0
_unified_blag_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_blag_actor
_unified_blag_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_lag_rl_cfg.algorithm
)
_unified_blag_algorithm.avoidance_teacher_mix_start = 1.0
_unified_blag_algorithm.avoidance_teacher_mix_end = 0.0
_unified_blag_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_blag_algorithm
# Leash-Ballet with the LAG-AWARE ee_body_pos termination (2026-09-06): the
# bound widens with the reference wrist/ankle speed so fast sweeps and filter
# corrections are not punished as lag; still targets keep the strict 0.25 m.
# Built ready to launch; not trained yet.
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    lag_aware_ee_termination=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    lag_aware_ee_termination=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_ballet_lag_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_ballet_blind"
    ),
  )
)
_unified_bblind_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_rl_cfg.actor
)
_unified_bblind_actor.adjust_command_with_joint_prediction = True
_unified_bblind_actor.command_joint_pos_offset = 0
_unified_bblind_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_bblind_actor
_unified_bblind_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_rl_cfg.algorithm
)
_unified_bblind_algorithm.avoidance_teacher_mix_start = 1.0
_unified_bblind_algorithm.avoidance_teacher_mix_end = 0.0
_unified_bblind_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_bblind_algorithm
# BLIND baseline (user, 2026-09-07): Leash-Ballet with the actor's LiDAR term
# reading "no returns" throughout training. Same network, rewards, reference
# and privileged critic; the honest no-perception lower bound.
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    blind_actor=True,
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    blind_actor=True,
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

# BLIND + NOMINAL baseline (user, 2026-09-08): the blind task above still
# trained on the CBF-filtered reference (rewards and teacher correction), so it
# had an avoidance signal without perception. This one turns both filters off:
# raw live-aligned reference, zero teacher residual, no LiDAR returns. Same
# network and rl cfg so the comparison isolates perception + filtering.
lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nominal_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_ballet_blind_nominal"
    ),
  )
)
_unified_bnominal_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nominal_rl_cfg.actor
)
_unified_bnominal_actor.adjust_command_with_joint_prediction = True
_unified_bnominal_actor.command_joint_pos_offset = 0
_unified_bnominal_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_bnominal_actor
_unified_bnominal_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nominal_rl_cfg.algorithm
)
_unified_bnominal_algorithm.avoidance_teacher_mix_start = 1.0
_unified_bnominal_algorithm.avoidance_teacher_mix_end = 0.0
_unified_bnominal_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_bnominal_algorithm
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    blind_actor=True,
    nominal_reference=True,
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    blind_actor=True,
    nominal_reference=True,
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nominal_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

# BLIND + NOMINAL + NO HUMANS (user, 2026-09-08): "literally the default
# policy". Training sees no humans at all (events and collision terminations
# removed; entities stay parked at -100 m), no filters, no LiDAR returns. The
# play cfg keeps the populated scene so evaluation is the same as everyone
# else's. Same network and rl cfg.
lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nohumans_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_ballet_blind_nohumans"
    ),
  )
)
_unified_bnoh_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nohumans_rl_cfg.actor
)
_unified_bnoh_actor.adjust_command_with_joint_prediction = True
_unified_bnoh_actor.command_joint_pos_offset = 0
_unified_bnoh_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_bnoh_actor
_unified_bnoh_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nohumans_rl_cfg.algorithm
)
_unified_bnoh_algorithm.avoidance_teacher_mix_start = 1.0
_unified_bnoh_algorithm.avoidance_teacher_mix_end = 0.0
_unified_bnoh_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_bnoh_algorithm
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    blind_actor=True,
    nominal_reference=True,
    training_humans=False,
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MANIFEST),
    blind_actor=True,
    nominal_reference=True,
    training_humans=False,
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_ballet_blind_nohumans_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

# ESCAPE MOVES (user, 2026-09-10): Leash-Ballet on the left/right mirrored
# library; a sustained planar CBF correction switches the RAW clip to a
# travelling ballet step aligned with the escape direction, then resumes the
# interrupted clip. Same network, rewards, terminations and rl cfg.
lidar_auxiliary_coadjust_unified_joint_leash_ballet_moves_rl_cfg = (
  _perceptive_lidar_runner_cfg(
    auxiliary_supervision=True,
    experiment_name=(
      "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash_ballet_moves"
    ),
  )
)
_unified_bmoves_actor = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_moves_rl_cfg.actor
)
_unified_bmoves_actor.adjust_command_with_joint_prediction = True
_unified_bmoves_actor.command_joint_pos_offset = 0
_unified_bmoves_actor.avoidance_joint_action_residual_gain = 0.0
del _unified_bmoves_actor
_unified_bmoves_algorithm = (
  lidar_auxiliary_coadjust_unified_joint_leash_ballet_moves_rl_cfg.algorithm
)
_unified_bmoves_algorithm.avoidance_teacher_mix_start = 1.0
_unified_bmoves_algorithm.avoidance_teacher_mix_end = 0.0
_unified_bmoves_algorithm.avoidance_teacher_mix_decay_updates = 8000
del _unified_bmoves_algorithm
register_mjlab_task(
  task_id=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID,
  env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MIRROR_MANIFEST),
    escape_moves=True,
  ),
  play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(
    play=True,
    active_joint_reward=True,
    root_lead_m=UNIFIED_ROOT_LEAD_M,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MIRROR_MANIFEST),
    escape_moves=True,
  ),
  rl_cfg=lidar_auxiliary_coadjust_unified_joint_leash_ballet_moves_rl_cfg,
  runner_cls=MotionTrackingOnPolicyRunner,
)

sparse_lidar_avoidance_rl_cfg = _perceptive_lidar_runner_cfg(
  azimuth_samples=120,
  elevation_samples=4,
  direction_bins=24,
  elevation_bins=3,
  experiment_name="safe_mimic_g1_live_aligned_lidar_avoidance_sparse",
)
register_mjlab_task(
  task_id=SPARSE_LIDAR_AVOIDANCE_TASK_ID,
  env_cfg=unitree_g1_sparse_lidar_avoidance_tracking_env_cfg(),
  play_env_cfg=unitree_g1_sparse_lidar_avoidance_tracking_env_cfg(play=True),
  rl_cfg=sparse_lidar_avoidance_rl_cfg,
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
  "LIDAR_AVOIDANCE_TASK_ID",
  "LIDAR_AUXILIARY_AVOIDANCE_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID",
  "LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID",
  "LIDAR_AUXILIARY_EXPOSED_TASK_ID",
  "LIDAR_RANGE_RATE_AVOIDANCE_TASK_ID",
  "MOTION_LIBRARY_IMPLICIT_STATE_TASK_ID",
  "MOTION_LIBRARY_STATE_EST_TASK_ID",
  "MOTION_LIBRARY_TASK_ID",
  "NOMINAL_LIDAR_DEBUG_TASK_ID",
  "REFERENCE_FILTER_DEMO_TASK_ID",
  "REFERENCE_FILTER_POLICY_DEMO_TASK_ID",
  "REFERENCE_REPLAY_DEMO_TASK_ID",
  "SPARSE_LIDAR_AVOIDANCE_TASK_ID",
  "TASK_ID",
]
