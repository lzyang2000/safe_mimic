"""Leash-Ballet-Moves task: escape moves on the mirrored ballet library."""

from pathlib import Path

import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
)
from safe_mimic.tasks.env_cfg import (
  DEFAULT_G1_BALLET_ESCAPE_INDEX,
  DEFAULT_G1_BALLET_MANIFEST,
  DEFAULT_G1_BALLET_MIRROR_MANIFEST,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)

MOVES_ID = LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID
BALLET_ID = LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID


def test_constants_point_at_the_mirrored_library() -> None:
  assert DEFAULT_G1_BALLET_MIRROR_MANIFEST.name == "ballet.yaml"
  assert DEFAULT_G1_BALLET_MIRROR_MANIFEST.parent.name == "g1_ballet_v1_trim1s_mirror"
  assert DEFAULT_G1_BALLET_ESCAPE_INDEX == (
    DEFAULT_G1_BALLET_MIRROR_MANIFEST.with_name("escape_moves.json")
  )


def test_existing_ballet_task_is_unchanged() -> None:
  cfg = load_env_cfg(BALLET_ID)
  motion = cfg.commands["motion"]
  assert motion.escape_moves is None
  assert motion.motion_file == str(DEFAULT_G1_BALLET_MANIFEST)
  assert motion.disable_filters is False


def test_builder_flag_attaches_the_index_next_to_the_manifest() -> None:
  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
    active_joint_reward=True,
    root_lead_m=0.3,
    planar_filter_at_robot_root=True,
    motion_manifest=str(DEFAULT_G1_BALLET_MIRROR_MANIFEST),
    escape_moves=True,
  )
  moves = cfg.commands["motion"].escape_moves
  assert moves is not None
  assert Path(moves.index_file) == DEFAULT_G1_BALLET_ESCAPE_INDEX
  assert moves.trigger_source == "teacher"
  with pytest.raises(ValueError, match="escape_moves"):
    unitree_g1_lidar_unified_reference_tracking_env_cfg(escape_moves=True)


def test_moves_task_registered_for_train_and_play() -> None:
  for play in (False, True):
    cfg = load_env_cfg(MOVES_ID, play=play)
    motion = cfg.commands["motion"]
    assert motion.motion_file == str(DEFAULT_G1_BALLET_MIRROR_MANIFEST)
    assert motion.escape_moves is not None
    assert Path(motion.escape_moves.index_file) == DEFAULT_G1_BALLET_ESCAPE_INDEX
    assert motion.max_root_lead_m == 0.3 and motion.planar_filter_at_robot_root
    assert motion.disable_filters is False
    assert "motion_active_joint_pos" in cfg.rewards
  rl = load_rl_cfg(MOVES_ID)
  assert rl.experiment_name.endswith("coadjust_unified_joint_leash_ballet_moves")
  assert rl.actor.adjust_command_with_joint_prediction is True
