from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from safe_mimic.tasks import (
  LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_TASK_ID,
)
from safe_mimic.tasks.env_cfg import (
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
)

# The command term ordering pinned here was discovered by building the real
# auxiliary env and reading the ObservationManager directly (see
# .superpowers/sdd/2026-09-01-phase2b-cotrained-limb-adjuster/task-2-report.md
# and the scratchpad script it records): the "command" term lands first, at
# flattened offset 0, followed by motion_anchor_ori_b (6), base_ang_vel (3),
# joint_pos (29), joint_vel (29), and actions (29), for a 154-value actor
# vector. The co-adjust runner cfg hardcodes command_joint_pos_offset from
# this term order; if a future change reorders the actor observation terms,
# this test must fail before the hardcoded offset silently corrupts the slice.
_EXPECTED_ACTOR_TERM_ORDER = (
  "command",
  "motion_anchor_ori_b",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)


def test_actor_observation_term_order_matches_discovered_layout() -> None:
  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg()
  assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERM_ORDER
  # The command term is first, so its offset is 0 and there are no terms
  # before it whose widths would need to be pinned separately.
  assert _EXPECTED_ACTOR_TERM_ORDER[0] == "command"


def test_coadjust_task_is_registered_with_command_adjustment_enabled() -> None:
  env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  assert env_cfg.commands["motion"].expose_filtered_command is False

  play_env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID, play=True)
  assert play_env_cfg.commands["motion"].expose_filtered_command is False

  rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  assert rl_cfg.experiment_name == (
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust"
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


def test_coadjust_task_otherwise_mirrors_auxiliary_task() -> None:
  coadjust_env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  auxiliary_env_cfg = load_env_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)
  assert (
    tuple(coadjust_env_cfg.observations["avoidance_teacher"].terms)
    == tuple(auxiliary_env_cfg.observations["avoidance_teacher"].terms)
    == ("corrections",)
  )
  assert (
    tuple(coadjust_env_cfg.observations["actor"].terms)
    == tuple(auxiliary_env_cfg.observations["actor"].terms)
    == _EXPECTED_ACTOR_TERM_ORDER
  )

  coadjust_rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  auxiliary_rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)
  assert coadjust_rl_cfg.algorithm.class_name == auxiliary_rl_cfg.algorithm.class_name
  assert coadjust_rl_cfg.actor.class_name == auxiliary_rl_cfg.actor.class_name
  assert coadjust_rl_cfg.obs_groups["actor"] == auxiliary_rl_cfg.obs_groups["actor"]
  assert (
    coadjust_rl_cfg.actor.avoidance_planar_dim
    == auxiliary_rl_cfg.actor.avoidance_planar_dim
  )
  assert (
    coadjust_rl_cfg.actor.avoidance_joint_dim
    == auxiliary_rl_cfg.actor.avoidance_joint_dim
  )


def test_base_auxiliary_task_remains_unchanged() -> None:
  auxiliary_rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)
  assert auxiliary_rl_cfg.actor.adjust_command_with_joint_prediction is False
  assert auxiliary_rl_cfg.actor.command_joint_pos_offset == -1
  assert auxiliary_rl_cfg.actor.avoidance_joint_action_residual_gain == 0.5

  assert auxiliary_rl_cfg.algorithm.avoidance_teacher_mix_start == 0.1
  assert auxiliary_rl_cfg.algorithm.avoidance_teacher_mix_end == 0.0
  assert auxiliary_rl_cfg.algorithm.avoidance_teacher_mix_decay_updates == 5000

  auxiliary_env_cfg = load_env_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)
  assert auxiliary_env_cfg.commands["motion"].expose_filtered_command is False
