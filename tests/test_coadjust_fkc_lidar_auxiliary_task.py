from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from safe_mimic.tasks import (
  LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_TASK_ID,
  LIDAR_AUXILIARY_EXPOSED_TASK_ID,
)
from safe_mimic.tasks.env_cfg import (
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
)

# Mirrors tests/test_coadjust_lidar_auxiliary_task.py's pinned actor term
# order; see that file's comment for the introspection run this depends on.
_EXPECTED_ACTOR_TERM_ORDER = (
  "command",
  "motion_anchor_ori_b",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)


def test_env_cfg_kwarg_defaults_propagation_off() -> None:
  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg()
  assert cfg.commands["motion"].propagate_arm_corrections_to_body_targets is False


def test_env_cfg_kwarg_enables_propagation_without_reordering_observations() -> None:
  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    propagate_arm_corrections_to_body_targets=True
  )
  assert cfg.commands["motion"].propagate_arm_corrections_to_body_targets is True
  # The propagation flag only changes reward-side body targets; it must not
  # perturb the actor observation layout the co-adjust offset depends on.
  assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERM_ORDER


def test_coadjust_fkc_task_is_registered_with_command_adjustment_enabled() -> None:
  env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID)
  assert env_cfg.commands["motion"].expose_filtered_command is False
  assert env_cfg.commands["motion"].propagate_arm_corrections_to_body_targets is True

  play_env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID, play=True)
  assert play_env_cfg.commands["motion"].expose_filtered_command is False
  assert (
    play_env_cfg.commands["motion"].propagate_arm_corrections_to_body_targets is True
  )

  rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID)
  assert rl_cfg.experiment_name == (
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_fkc"
  )
  assert rl_cfg.logger == "tensorboard"
  assert rl_cfg.save_interval == 1000
  assert rl_cfg.upload_model is False

  assert rl_cfg.actor.adjust_command_with_joint_prediction is True
  # Offset of the command term's first joint-position value in the flattened
  # actor observation vector; see _EXPECTED_ACTOR_TERM_ORDER above.
  assert rl_cfg.actor.command_joint_pos_offset == 0
  assert rl_cfg.actor.avoidance_joint_action_residual_gain == 0.0

  assert rl_cfg.algorithm.avoidance_teacher_mix_start == 1.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_end == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_decay_updates == 8000


def test_coadjust_fkc_task_otherwise_mirrors_coadjust_task() -> None:
  fkc_env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID)
  coadjust_env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  assert (
    tuple(fkc_env_cfg.observations["avoidance_teacher"].terms)
    == tuple(coadjust_env_cfg.observations["avoidance_teacher"].terms)
    == ("corrections",)
  )
  assert (
    tuple(fkc_env_cfg.observations["actor"].terms)
    == tuple(coadjust_env_cfg.observations["actor"].terms)
    == _EXPECTED_ACTOR_TERM_ORDER
  )

  fkc_rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID)
  coadjust_rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  assert fkc_rl_cfg.algorithm.class_name == coadjust_rl_cfg.algorithm.class_name
  assert fkc_rl_cfg.actor.class_name == coadjust_rl_cfg.actor.class_name
  assert fkc_rl_cfg.obs_groups["actor"] == coadjust_rl_cfg.obs_groups["actor"]
  assert (
    fkc_rl_cfg.actor.avoidance_planar_dim == coadjust_rl_cfg.actor.avoidance_planar_dim
  )
  assert (
    fkc_rl_cfg.actor.avoidance_joint_dim == coadjust_rl_cfg.actor.avoidance_joint_dim
  )
  assert (
    fkc_rl_cfg.actor.adjust_command_with_joint_prediction
    == coadjust_rl_cfg.actor.adjust_command_with_joint_prediction
  )
  assert fkc_rl_cfg.actor.command_joint_pos_offset == (
    coadjust_rl_cfg.actor.command_joint_pos_offset
  )
  assert fkc_rl_cfg.actor.avoidance_joint_action_residual_gain == (
    coadjust_rl_cfg.actor.avoidance_joint_action_residual_gain
  )
  # The only intended difference between the two tasks: FKC propagates the
  # filtered arm joint corrections into the command's body targets.
  assert fkc_env_cfg.commands["motion"].propagate_arm_corrections_to_body_targets
  assert not (
    coadjust_env_cfg.commands["motion"].propagate_arm_corrections_to_body_targets
  )


def test_other_lidar_auxiliary_tasks_leave_propagation_off() -> None:
  for task_id in (
    LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_TASK_ID,
    LIDAR_AUXILIARY_EXPOSED_TASK_ID,
  ):
    env_cfg = load_env_cfg(task_id)
    assert (
      env_cfg.commands["motion"].propagate_arm_corrections_to_body_targets is False
    ), task_id

    play_env_cfg = load_env_cfg(task_id, play=True)
    assert (
      play_env_cfg.commands["motion"].propagate_arm_corrections_to_body_targets
      is False
    ), task_id
