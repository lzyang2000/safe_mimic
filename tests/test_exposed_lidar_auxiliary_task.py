from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from safe_mimic.tasks import (
  LIDAR_AUXILIARY_AVOIDANCE_TASK_ID,
  LIDAR_AUXILIARY_EXPOSED_TASK_ID,
)
from safe_mimic.tasks.env_cfg import (
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
)


def test_expose_filtered_command_flag_defaults_to_false() -> None:
  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg()
  assert cfg.commands["motion"].expose_filtered_command is False


def test_expose_filtered_command_flag_can_be_enabled() -> None:
  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    expose_filtered_command=True
  )
  assert cfg.commands["motion"].expose_filtered_command is True

  play_cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
    play=True, expose_filtered_command=True
  )
  assert play_cfg.commands["motion"].expose_filtered_command is True


def test_exposed_task_is_registered_with_filtered_command_exposed() -> None:
  env_cfg = load_env_cfg(LIDAR_AUXILIARY_EXPOSED_TASK_ID)
  assert env_cfg.commands["motion"].expose_filtered_command is True

  play_env_cfg = load_env_cfg(LIDAR_AUXILIARY_EXPOSED_TASK_ID, play=True)
  assert play_env_cfg.commands["motion"].expose_filtered_command is True

  rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_EXPOSED_TASK_ID)
  assert rl_cfg.experiment_name == (
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_ttc_exposed_ft"
  )
  assert rl_cfg.logger == "tensorboard"
  assert rl_cfg.save_interval == 1000
  assert rl_cfg.upload_model is False


def test_exposed_task_otherwise_mirrors_auxiliary_task() -> None:
  exposed_env_cfg = load_env_cfg(LIDAR_AUXILIARY_EXPOSED_TASK_ID)
  auxiliary_env_cfg = load_env_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)
  assert (
    tuple(exposed_env_cfg.observations["avoidance_teacher"].terms)
    == tuple(auxiliary_env_cfg.observations["avoidance_teacher"].terms)
    == ("corrections",)
  )

  exposed_rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_EXPOSED_TASK_ID)
  auxiliary_rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_AVOIDANCE_TASK_ID)
  assert exposed_rl_cfg.algorithm.class_name == auxiliary_rl_cfg.algorithm.class_name
  assert exposed_rl_cfg.actor.class_name == auxiliary_rl_cfg.actor.class_name
  assert exposed_rl_cfg.obs_groups["actor"] == auxiliary_rl_cfg.obs_groups["actor"]

  # The base auxiliary task must remain untouched (default off).
  assert auxiliary_env_cfg.commands["motion"].expose_filtered_command is False
