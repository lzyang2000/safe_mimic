import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat

from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
  mdp,
)
from safe_mimic.tasks.env_cfg import (
  _NOMINAL_TRACKING_REWARD_NAMES,
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)

_EXPECTED_ACTOR_TERM_ORDER = (
  "command",
  "motion_anchor_ori_b",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)


def test_reward_set_is_exactly_the_nominal_tracking_set() -> None:
  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  nominal = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)
  assert set(cfg.rewards) == set(nominal.rewards)
  for name, term in nominal.rewards.items():
    assert cfg.rewards[name].weight == term.weight, name
    assert cfg.rewards[name].params == term.params, name
    assert cfg.rewards[name].func is term.func, name


def test_terminations_are_nominal_plus_human_collisions() -> None:
  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  nominal = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)
  assert set(cfg.terminations) == set(nominal.terminations) | {
    "crowd_collision",
    "primary_human_collision",
  }
  for name, term in nominal.terminations.items():
    assert cfg.terminations[name].func is term.func, name
    assert cfg.terminations[name].params == term.params, name
    assert cfg.terminations[name].time_out == term.time_out, name


def test_filtered_reference_flags_and_actor_layout() -> None:
  for play in (False, True):
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(play=play)
    motion = cfg.commands["motion"]
    assert motion.propagate_joint_corrections_to_body_targets is True
    assert motion.propagate_arm_corrections_to_body_targets is False
    assert motion.closed_loop_root_target is True
    assert motion.expose_filtered_command is False
    assert motion.align_reference_to_robot_each_step is True
    assert motion.sampling_mode == ("start" if play else "uniform")
    assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERM_ORDER
    assert "avoidance_teacher" in cfg.observations
    assert "avoidance_robustness" in cfg.observations


def test_existing_tasks_untouched() -> None:
  for task_id in (
    LIDAR_AUXILIARY_COADJUST_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  ):
    motion = load_env_cfg(task_id).commands["motion"]
    assert motion.propagate_joint_corrections_to_body_targets is False
    assert motion.closed_loop_root_target is False
  aux = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg()
  assert "safe_planar_velocity" in aux.rewards
  assert "urgent_escape_progress" in aux.rewards


def test_motion_terminations_consume_the_filtered_reference() -> None:
  # User requirement (2026-09-03): motion-based terminations must evaluate the
  # filtered motion. mjlab's stock terminations read command.anchor_pos_w,
  # command.anchor_quat_w and command.body_pos_relative_w (recomputed from
  # body_pos_w + anchor pose in update_relative_body_poses); pin that the
  # unified task uses the stock functions AND that the filtered command class
  # overrides every property they consume.
  from safe_mimic.tasks.kinematic_replay_command import (
    PlanarFilteredReplayMotionCommand,
  )

  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  assert cfg.terminations["anchor_pos"].func is tracking_mdp.bad_anchor_pos_z_only
  assert cfg.terminations["anchor_ori"].func is tracking_mdp.bad_anchor_ori
  assert cfg.terminations["ee_body_pos"].func is tracking_mdp.bad_motion_body_pos_z_only
  overridden = PlanarFilteredReplayMotionCommand.__dict__
  for name in ("anchor_pos_w", "anchor_quat_w", "body_pos_w", "body_quat_w"):
    assert isinstance(overridden[name], property), name


def test_unified_task_registered_like_coadjust() -> None:
  env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID)
  assert env_cfg.commands["motion"].closed_loop_root_target is True
  # The REGISTERED cfgs (train and play) must come from the unified builder,
  # not from the auxiliary builder plus manual flag patches.
  for play in (False, True):
    registered = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID, play=play)
    assert set(registered.rewards) == set(_NOMINAL_TRACKING_REWARD_NAMES)
    assert registered.commands["motion"].propagate_joint_corrections_to_body_targets
    assert (
      registered.observations["actor"].terms["motion_anchor_ori_b"].func
      is mdp.raw_motion_anchor_ori_b
    )
  rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID)
  base = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  assert rl_cfg.actor.adjust_command_with_joint_prediction is True
  assert rl_cfg.actor.command_joint_pos_offset == 0
  assert rl_cfg.actor.avoidance_joint_action_residual_gain == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_start == 1.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_end == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_decay_updates == 8000
  assert rl_cfg.experiment_name == (
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified"
  )
  assert rl_cfg.actor.__class__ is base.actor.__class__


def test_actor_observes_the_raw_anchor_orientation_not_the_filtered_one() -> None:
  # 2026-09-03 follow-up: in whole-body mode command.anchor_quat_w carries the
  # privileged waist correction. At deployment only the raw reference and the
  # learned adjuster exist, so the actor must observe the raw anchor
  # orientation; the critic, rewards, and terminations keep the filtered one.
  aux = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg()
  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  actor_term = cfg.observations["actor"].terms["motion_anchor_ori_b"]
  critic_term = cfg.observations["critic"].terms["motion_anchor_ori_b"]
  assert actor_term.func is mdp.raw_motion_anchor_ori_b
  assert critic_term.func is tracking_mdp.motion_anchor_ori_b
  assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERM_ORDER
  aux_actor_term = aux.observations["actor"].terms["motion_anchor_ori_b"]
  assert actor_term.noise == aux_actor_term.noise
  assert actor_term.params == aux_actor_term.params


def test_raw_motion_anchor_ori_b_ignores_the_whole_body_correction(monkeypatch) -> None:
  # Pin that raw_motion_anchor_ori_b reads the RAW anchor pose even when the
  # command's anchor_quat_w property carries a whole-body FK correction.
  from safe_mimic.tasks.kinematic_replay_command import (
    PlanarFilteredReplayMotionCommand,
  )

  num_envs, num_bodies, anchor = 2, 4, 2
  identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  raw_quat = identity_quat.expand(num_envs, num_bodies, 4).clone()
  raw_pos = torch.zeros(num_envs, num_bodies, 3)
  yaw90 = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])

  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command.motion_anchor_body_index = anchor
  command._arm_body_target_quat_delta_w = torch.zeros(num_envs, num_bodies, 4)
  command._arm_body_target_quat_delta_w[..., 0] = 1.0
  command._arm_body_target_quat_delta_w[:, anchor] = yaw90
  command._anchor_target_corrected = True
  monkeypatch.setattr(command, "_raw_body_pos_w", lambda: raw_pos)
  monkeypatch.setattr(command, "_raw_body_quat_w", lambda: raw_quat)
  # robot_anchor_pos_w/quat_w are read-only mjlab base-class properties;
  # replace them on the class for this instance-only stub.
  monkeypatch.setattr(
    PlanarFilteredReplayMotionCommand, "robot_anchor_pos_w", torch.zeros(num_envs, 3)
  )
  monkeypatch.setattr(
    PlanarFilteredReplayMotionCommand,
    "robot_anchor_quat_w",
    identity_quat.expand(num_envs, 4).clone(),
  )

  # Sanity: the whole-body-corrected anchor orientation differs from raw.
  torch.testing.assert_close(command.anchor_quat_w, yaw90.expand(num_envs, 4))

  class _StubCommandManager:
    def get_term(self, name: str) -> PlanarFilteredReplayMotionCommand:
      assert name == "motion"
      return command

  class _StubEnv:
    command_manager = _StubCommandManager()

  result = mdp.raw_motion_anchor_ori_b(_StubEnv(), "motion")
  expected_mat = matrix_from_quat(identity_quat.expand(num_envs, 4))
  expected = expected_mat[..., :2].reshape(num_envs, -1)
  torch.testing.assert_close(result, expected)


def test_active_joint_variant_adds_exactly_one_joint_term() -> None:
  base = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  assert "motion_active_joint_pos" not in base.rewards
  for play in (False, True):
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=play, active_joint_reward=True
    )
    assert set(cfg.rewards) == set(_NOMINAL_TRACKING_REWARD_NAMES) | {
      "motion_active_joint_pos"
    }
    term = cfg.rewards["motion_active_joint_pos"]
    assert term.func is mdp.active_correction_joint_tracking_exp
    assert term.weight == 1.0
    assert term.params == {
      "command_name": "motion",
      "std": 0.4,
      "activation_threshold_rad": 0.05,
    }
    # Everything else is the unified task.
    motion = cfg.commands["motion"]
    assert motion.propagate_joint_corrections_to_body_targets is True
    assert motion.closed_loop_root_target is True
    assert (
      cfg.observations["actor"].terms["motion_anchor_ori_b"].func
      is mdp.raw_motion_anchor_ori_b
    )
    assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERM_ORDER


def test_unified_joint_task_registered() -> None:
  for play in (False, True):
    env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID, play=play)
    assert "motion_active_joint_pos" in env_cfg.rewards
    assert env_cfg.commands["motion"].closed_loop_root_target is True
  plain = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID)
  assert "motion_active_joint_pos" not in plain.rewards
  rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID)
  base = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID)
  assert rl_cfg.actor.adjust_command_with_joint_prediction is True
  assert rl_cfg.actor.command_joint_pos_offset == 0
  assert rl_cfg.actor.avoidance_joint_action_residual_gain == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_start == 1.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_end == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_decay_updates == 8000
  assert rl_cfg.actor.__class__ is base.actor.__class__


def test_leash_variant_changes_only_the_reference_generator() -> None:
  joint = unitree_g1_lidar_unified_reference_tracking_env_cfg(active_joint_reward=True)
  assert joint.commands["motion"].max_root_lead_m is None
  assert joint.commands["motion"].planar_filter_at_robot_root is False
  for play in (False, True):
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=play,
      active_joint_reward=True,
      root_lead_m=0.3,
      planar_filter_at_robot_root=True,
    )
    motion = cfg.commands["motion"]
    assert motion.max_root_lead_m == 0.3
    assert motion.planar_filter_at_robot_root is True
    assert motion.closed_loop_root_target is True
    assert motion.propagate_joint_corrections_to_body_targets is True
    reference = unitree_g1_lidar_unified_reference_tracking_env_cfg(
      play=play, active_joint_reward=True
    )
    assert set(cfg.rewards) == set(reference.rewards)
    for name, term in cfg.rewards.items():
      assert term.weight == reference.rewards[name].weight
      assert term.params == reference.rewards[name].params
    assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERM_ORDER
    assert set(cfg.terminations) == set(reference.terminations)


def test_unified_joint_leash_task_registered() -> None:
  for play in (False, True):
    env_cfg = load_env_cfg(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID, play=play
    )
    motion = env_cfg.commands["motion"]
    assert motion.max_root_lead_m == 0.3
    assert motion.planar_filter_at_robot_root is True
    assert motion.closed_loop_root_target is True
    assert "motion_active_joint_pos" in env_cfg.rewards
  # Existing variants keep the unbounded, target-evaluated filter.
  for task_id in (
    LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
  ):
    motion = load_env_cfg(task_id).commands["motion"]
    assert motion.max_root_lead_m is None
    assert motion.planar_filter_at_robot_root is False
  rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID)
  base = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID)
  assert rl_cfg.experiment_name == (
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified_joint_leash"
  )
  assert rl_cfg.actor.adjust_command_with_joint_prediction is True
  assert rl_cfg.actor.command_joint_pos_offset == 0
  assert rl_cfg.actor.avoidance_joint_action_residual_gain == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_start == 1.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_end == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_decay_updates == 8000
  assert rl_cfg.actor.__class__ is base.actor.__class__
