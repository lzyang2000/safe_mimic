import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)

from safe_mimic.assets.g1 import (
  SHOULDER_PLANK_HALF_SIZE_M,
  SHOULDER_PLANK_POSITION_M,
)
from safe_mimic.sensing.held_scan import HeldScanRayCastSensorCfg
from safe_mimic.tasks.env_cfg import (
  HUMAN_MOTION_EVENT_NAME,
  LIDAR_AZIMUTH_SAMPLES,
  LIDAR_ELEVATIONS_DEG,
  LIDAR_MAX_DISTANCE_M,
  LIDAR_MIN_DISTANCE_M,
  LIDAR_SCAN_PERIOD_S,
  LIDAR_SCAN_PHASES,
  LIDAR_SENSOR_NAME,
  LIVOX_SNAPSHOT_AZIMUTH_SAMPLES,
  LIVOX_SNAPSHOT_ELEVATION_SAMPLES,
  PRIMARY_HUMAN_CONTACT_SENSOR_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  PROTOTYPE_LIDAR_AZIMUTH_SAMPLES,
  PROTOTYPE_LIDAR_ELEVATIONS_DEG,
  unitree_g1_kinematic_reference_lidar_demo_env_cfg,
  unitree_g1_lidar_avoidance_tracking_env_cfg,
  unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg,
  unitree_g1_nominal_lidar_debug_env_cfg,
  unitree_g1_obstacle_aware_tracking_env_cfg,
  unitree_g1_reference_filter_lidar_demo_env_cfg,
  unitree_g1_reference_filter_policy_lidar_demo_env_cfg,
  unitree_g1_sparse_lidar_avoidance_tracking_env_cfg,
)
from safe_mimic.tasks.human_capsule_event import HumanCapsuleMotion
from safe_mimic.tasks.kinematic_replay_command import (
  KinematicReplayMotionCommand,
  KinematicReplayMotionCommandCfg,
  PlanarFilteredReplayMotionCommand,
  PlanarFilteredReplayMotionCommandCfg,
  _apply_planar_reference_alignment,
  _planar_reference_alignment,
)
from safe_mimic.tasks.packed_human_event import (
  PackedHumanCapsuleCrowdMotion,
  PackedHumanCapsuleMotion,
)

EXPECTED_HEAD_SIDE_HALF_SIZE = (0.011, 0.0065, 0.0375)
EXPECTED_HEAD_SIDE_X = 0.0402835
EXPECTED_HEAD_SIDE_Z = 0.37868


def test_primary_human_velocity_diagnostic_uses_pose_update_interval(
  capsys: pytest.CaptureFixture[str],
) -> None:
  motion = HumanCapsuleMotion.__new__(HumanCapsuleMotion)
  motion.print_velocity = True
  motion.velocity_print_interval_s = 0.5
  motion._velocity_previous_root_w = torch.zeros(1, 3)
  motion._velocity_previous_time_s = torch.zeros(1)
  motion._velocity_valid = torch.zeros(1, dtype=torch.bool)
  motion._velocity_measurement_valid = torch.zeros(1, dtype=torch.bool)
  motion._velocity_w = torch.zeros(1, 3)
  motion._velocity_closing_speed_mps = torch.zeros(1)
  motion._velocity_distance_m = torch.zeros(1)
  motion._velocity_last_print_s = -float("inf")
  motion.robot_entity_name = "robot"
  poses = SimpleNamespace(
    root_positions_w=torch.zeros(1, 3),
    active=torch.ones(1, dtype=torch.bool),
  )
  motion.sampler = SimpleNamespace(
    _poses=poses,
    playback_speed=torch.tensor([1.25]),
  )
  robot_data = SimpleNamespace(
    root_link_pos_w=torch.tensor([[2.0, 0.0, 0.0]]),
    root_link_lin_vel_w=torch.zeros(1, 3),
  )
  motion._env = SimpleNamespace(scene={"robot": SimpleNamespace(data=robot_data)})
  env_ids = torch.tensor([0])

  motion._update_velocity_diagnostic(env_ids, 0.0)
  assert capsys.readouterr().out == ""
  poses.root_positions_w[0, 0] = 0.2
  motion._update_velocity_diagnostic(env_ids, 0.1)

  output = capsys.readouterr().out
  assert "v_w=(+2.00, +0.00, +0.00) m/s" in output
  assert "speed_xy=2.00 m/s" in output
  assert "closing=+2.00 m/s" in output
  assert "playback=1.25x" in output
  assert motion._velocity_label_text(0) == (
    "human v=(+2.00, +0.00, +0.00) m/s\n"
    "speed=2.00  closing=+2.00  distance=1.80 m"
  )


def _pillar_quat(*, roll_deg: float) -> tuple[float, float, float, float]:
  roll = math.radians(roll_deg)
  pitch = math.radians(-5.0)
  cos_roll = math.cos(roll / 2.0)
  sin_roll = math.sin(roll / 2.0)
  cos_pitch = math.cos(pitch / 2.0)
  sin_pitch = math.sin(pitch / 2.0)
  return (
    cos_pitch * cos_roll,
    cos_pitch * sin_roll,
    sin_pitch * cos_roll,
    -sin_pitch * sin_roll,
  )


def test_nominal_debug_task_preserves_policy_observation_contract() -> None:
  upstream = unitree_g1_flat_tracking_env_cfg(play=True)
  debug = unitree_g1_nominal_lidar_debug_env_cfg(play=True)

  assert debug.observations.keys() == upstream.observations.keys()
  crowd = debug.events["animate_human"]
  assert crowd.params["inward_facing_probability"] == 1.0
  assert crowd.params["inward_facing_jitter_rad"] == 0.0
  assert crowd.params["mesh_update_hz"] == 5.0
  assert crowd.params["mesh_voxel_size_m"] == 0.02
  assert crowd.params["update_hz"] == 5.0
  assert crowd.params["facing_yaw_offset_rad"] == pytest.approx(-math.pi / 2)
  assert crowd.params["min_count"] == 30
  assert crowd.params["randomize_density"] is False
  assert crowd.params["min_radius_m"] == 3.0
  assert crowd.params["max_radius_m"] == 3.001
  assert crowd.params["radial_jitter_m"] == 0.0
  assert crowd.params["min_shape_exponent"] == 2.0
  assert crowd.params["max_shape_exponent"] == 8.0
  assert crowd.params["min_human_height_m"] == 1.3
  assert crowd.params["max_human_height_m"] == 1.9
  assert PRIMARY_HUMAN_ENTITY_NAME in debug.scene.entities
  assert PRIMARY_HUMAN_EVENT_NAME in debug.events
  primary = debug.events[PRIMARY_HUMAN_EVENT_NAME]
  assert primary.params["min_initial_spawn_radius_m"] == 3.0
  assert primary.params["max_initial_spawn_radius_m"] == 3.0
  assert primary.params["min_intersection_delay_s"] == 3.0
  assert primary.params["max_intersection_delay_s"] == 3.0
  assert primary.params["update_hz"] == 50.0
  assert primary.params["min_human_height_m"] == 1.3
  assert primary.params["max_human_height_m"] == 1.9
  assert primary.params["print_velocity"] is True
  assert primary.params["velocity_print_interval_s"] == 0.5
  for group_name in upstream.observations:
    assert (
      debug.observations[group_name].terms.keys()
      == upstream.observations[group_name].terms.keys()
    )

  lidar = next(
    sensor for sensor in debug.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )
  assert lidar.frame.type == "site"
  assert lidar.min_distance == LIDAR_MIN_DISTANCE_M == 0.3
  assert lidar.max_distance == LIDAR_MAX_DISTANCE_M == 5.0
  assert lidar.pattern.num_rays == 4_995
  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.rays_per_phase == 999
  assert lidar.scan_period == LIDAR_SCAN_PERIOD_S == 0.1
  assert lidar.pattern.phases == LIDAR_SCAN_PHASES == 5
  assert LIVOX_SNAPSHOT_AZIMUTH_SAMPLES == 185
  assert LIVOX_SNAPSHOT_ELEVATION_SAMPLES == 27
  assert lidar.pattern.elevation_angles_deg[0] == 0.0
  assert lidar.pattern.elevation_angles_deg[-1] == -52.0
  assert LIDAR_AZIMUTH_SAMPLES == 180
  assert LIDAR_ELEVATIONS_DEG == (0.0, -10.0, -20.0, -30.0, -40.0, -50.0)
  assert not lidar.exclude_parent_body
  assert lidar.include_geom_groups == (0, 3)
  assert lidar.debug_vis
  assert not lidar.viz.show_rays
  assert lidar.viz.hit_sphere_color == (1.0, 0.0, 0.0, 1.0)
  assert lidar.viz.hit_sphere_radius == 0.3

  robot_spec = debug.scene.entities["robot"].spec_fn()
  for side in ("left", "right"):
    visual = robot_spec.geom(f"lidar_head_side_{side}_visual")
    occluder = robot_spec.geom(f"lidar_head_side_{side}_occluder")
    assert visual is not None and visual.group == 2
    assert occluder is not None and occluder.group == 3
    assert occluder.contype == 0 and occluder.conaffinity == 0
    assert tuple(visual.size) == EXPECTED_HEAD_SIDE_HALF_SIZE
    assert tuple(occluder.size) == EXPECTED_HEAD_SIDE_HALF_SIZE
    assert visual.pos[0] == pytest.approx(EXPECTED_HEAD_SIDE_X)
    assert visual.pos[2] == pytest.approx(EXPECTED_HEAD_SIDE_Z)

  left = robot_spec.geom("lidar_head_side_left_visual")
  right = robot_spec.geom("lidar_head_side_right_visual")
  assert left is not None and right is not None
  assert left.pos[1] == pytest.approx(0.03503)
  assert right.pos[1] == pytest.approx(-0.03497)
  assert tuple(left.quat) == pytest.approx(_pillar_quat(roll_deg=-15.0))
  assert tuple(right.quat) == pytest.approx(_pillar_quat(roll_deg=15.0))

  torso = robot_spec.body("torso_link")
  assert torso is not None
  torso_geom_names = {geom.name for geom in torso.geoms}
  for suffix, expected_group in (("visual", 2), ("occluder", 3)):
    name = f"lidar_shoulder_plank_{suffix}"
    assert name in torso_geom_names
    plank = robot_spec.geom(name)
    assert plank is not None and plank.group == expected_group
    assert plank.contype == 0 and plank.conaffinity == 0
    assert tuple(plank.pos) == SHOULDER_PLANK_POSITION_M
    assert tuple(plank.size) == SHOULDER_PLANK_HALF_SIZE_M


def test_training_task_interleaves_sparse_pattern_at_10_hz() -> None:
  cfg = unitree_g1_obstacle_aware_tracking_env_cfg(play=True)
  lidar = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )
  assert lidar.pattern.num_rays == 1080
  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.rays_per_phase == 216
  assert lidar.scan_period == 0.1


def test_new_avoidance_task_uses_pooled_dense_dual_scan_and_raw_command() -> None:
  cfg = unitree_g1_lidar_avoidance_tracking_env_cfg()
  upstream = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)
  motion = cfg.commands["motion"]
  assert isinstance(motion, PlanarFilteredReplayMotionCommandCfg)
  assert motion.write_reference_to_sim is False
  assert motion.align_reference_to_robot_each_step is True
  assert motion.expose_filtered_command is False
  assert motion.planar_filter.safe_clearance_m == 0.8
  assert motion.link_filter.safe_clearance_m == 0.8
  assert motion.link_filter_capsules_per_group == (5, None)
  assert motion.link_filter_nearest_groups == (8, None)

  actor_terms = cfg.observations["actor"].terms
  assert "motion_anchor_pos_b" not in actor_terms
  assert "base_lin_vel" not in actor_terms
  assert LIDAR_SENSOR_NAME not in actor_terms
  assert tuple(cfg.observations["lidar"].terms) == (
    "directional_scan_pair",
    "scan_age",
  )
  actor_lidar = cfg.observations["lidar"].terms["directional_scan_pair"]
  assert actor_lidar.params["azimuth_bins"] == 120
  assert actor_lidar.params["elevation_bins"] == 9
  critic_terms = cfg.observations["critic"].terms
  critic_lidar = critic_terms[LIDAR_SENSOR_NAME]
  assert critic_lidar.func.__name__ == "CachedDirectionalLidarRanges"
  assert critic_lidar.params["azimuth_bins"] == 24
  assert critic_lidar.params["elevation_bins"] == 3
  assert "filtered_planar_velocity_b" in critic_terms
  assert "filtered_joint_command" in critic_terms
  assert "crowd_capsule_vectors_b" in critic_terms

  lidar = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )
  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.num_rays == 4_995
  assert lidar.pattern.rays_per_phase == 999
  assert lidar.pattern.phases == 5
  assert lidar.scan_period == 0.1

  assert cfg.events[HUMAN_MOTION_EVENT_NAME].params["obstacle_free_probability"] == 0.25
  assert cfg.events[HUMAN_MOTION_EVENT_NAME].func is PackedHumanCapsuleCrowdMotion
  assert cfg.events[PRIMARY_HUMAN_EVENT_NAME].func is PackedHumanCapsuleMotion
  assert (
    cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["use_shared_obstacle_free_mask"] is True
  )
  assert cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["update_hz"] == 10.0
  primary_params = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
  assert primary_params["encounter_sampling"] == "ttc"
  assert primary_params["min_initial_spawn_radius_m"] == 0.75
  assert primary_params["max_initial_spawn_radius_m"] == 4.0
  assert primary_params["min_intersection_delay_s"] == 0.5
  assert primary_params["max_intersection_delay_s"] == 4.0
  for name, reward in upstream.rewards.items():
    assert cfg.rewards[name].weight == reward.weight
  assert cfg.rewards["human_proximity"].params["safe_clearance"] == 0.8
  assert cfg.rewards["primary_human_proximity"].params["safe_clearance"] == 0.8
  assert "safe_planar_velocity" in cfg.rewards
  assert "filtered_joint_position" in cfg.rewards
  assert "safe_planar_freeze" in cfg.rewards
  assert "safe_planar_progress" in cfg.rewards
  primary_human = cfg.scene.entities[PRIMARY_HUMAN_ENTITY_NAME].spec_fn().compile()
  assert primary_human.nmocap == 1
  assert all(contype == 0 for contype in primary_human.geom_contype)
  assert all(
    sensor.name != PRIMARY_HUMAN_CONTACT_SENSOR_NAME
    for sensor in (cfg.scene.sensors or ())
  )
  assert "primary_human_collision" not in cfg.rewards
  assert cfg.terminations["crowd_collision"].params["collision_margin"] == 0.1
  assert cfg.terminations["primary_human_collision"].params == {
    "robot_entity": "robot",
    "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
    "robot_link_names": motion.link_filter.body_names,
    "link_radius": motion.link_filter.link_radius_m,
    "collision_margin": 0.1,
  }


def test_sparse_avoidance_prototype_uses_120_by_4_dual_scan() -> None:
  cfg = unitree_g1_sparse_lidar_avoidance_tracking_env_cfg()
  lidar = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )
  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.azimuth_samples == PROTOTYPE_LIDAR_AZIMUTH_SAMPLES == 120
  assert lidar.pattern.elevation_angles_deg == PROTOTYPE_LIDAR_ELEVATIONS_DEG
  assert lidar.pattern.num_rays == 480
  assert lidar.pattern.rays_per_phase == 96
  actor_lidar = cfg.observations["lidar"].terms["directional_scan_pair"]
  assert actor_lidar.params["azimuth_bins"] == 24
  assert actor_lidar.params["elevation_bins"] == 3
  noise = actor_lidar.params["noise_cfg"]
  assert noise.azimuth_samples == 120
  assert noise.sector_width_samples == 30


def test_range_rate_avoidance_adds_closing_speed_and_link_reward() -> None:
  cfg = unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg()
  lidar_terms = cfg.observations["lidar"].terms

  assert tuple(lidar_terms) == ("directional_range_rate", "scan_age")
  range_rate = lidar_terms["directional_range_rate"]
  assert range_rate.func.__name__ == "CachedDirectionalHeldLidarRangeRate"
  assert range_rate.params["max_abs_range_rate_mps"] == 5.0
  assert range_rate.clip == (-1.0, 1.0)
  link_reward = cfg.rewards["link_proximity"]
  assert link_reward.weight == -3.0
  assert link_reward.params == {
    "command_name": "motion",
    "safe_clearance_m": 0.8,
    "metric_name": "link_filter_minimum_clearance_m",
  }


def test_raw_actor_command_does_not_leak_filtered_joint_teacher() -> None:
  raw_pos = torch.tensor([[1.0, 2.0]])
  raw_vel = torch.tensor([[3.0, 4.0]])
  filtered_pos = torch.tensor([[10.0, 20.0]])
  filtered_vel = torch.tensor([[30.0, 40.0]])
  command = SimpleNamespace(
    cfg=SimpleNamespace(expose_filtered_command=False),
    joint_pos=filtered_pos,
    joint_vel=filtered_vel,
    _raw_joint_pos=lambda: raw_pos,
    _raw_joint_vel=lambda: raw_vel,
  )

  actor_command = PlanarFilteredReplayMotionCommand.command.fget(command)
  assert actor_command is not None
  torch.testing.assert_close(actor_command, torch.cat((raw_pos, raw_vel), dim=1))

  command.cfg.expose_filtered_command = True
  teacher_command = PlanarFilteredReplayMotionCommand.command.fget(command)
  assert teacher_command is not None
  torch.testing.assert_close(
    teacher_command, torch.cat((filtered_pos, filtered_vel), dim=1)
  )


def test_kinematic_replay_demo_is_exact_and_defaults_to_dense_5_hz() -> None:
  nominal = unitree_g1_nominal_lidar_debug_env_cfg()
  replay = unitree_g1_kinematic_reference_lidar_demo_env_cfg()

  assert replay.observations.keys() == nominal.observations.keys()
  for group_name in nominal.observations:
    assert (
      replay.observations[group_name].terms.keys()
      == nominal.observations[group_name].terms.keys()
    )
  assert replay.actions.keys() == nominal.actions.keys()

  motion = replay.commands["motion"]
  assert isinstance(motion, KinematicReplayMotionCommandCfg)
  assert motion.sampling_mode == "start"
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.joint_position_range == (0.0, 0.0)

  assert replay.sim.mujoco.gravity == (0.0, 0.0, 0.0)
  assert "contact" in replay.sim.mujoco.disableflags
  assert "actuation" in replay.sim.mujoco.disableflags
  assert "push_robot" not in replay.events
  assert replay.terminations == {}

  lidar = next(
    sensor for sensor in replay.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )
  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.num_rays == 4_995
  assert lidar.pattern.phases == 10
  assert lidar.pattern.rays_per_phase == 513
  assert lidar.scan_period == 0.2


def test_kinematic_replay_demo_keeps_dense_10_hz_option() -> None:
  replay = unitree_g1_kinematic_reference_lidar_demo_env_cfg(scan_hz=10)
  lidar = next(
    sensor for sensor in replay.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )

  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.num_rays == 4_995
  assert lidar.pattern.phases == 5
  assert lidar.pattern.rays_per_phase == 999
  assert lidar.scan_period == 0.1

  with pytest.raises(ValueError, match="scan_hz must be 5 or 10"):
    unitree_g1_kinematic_reference_lidar_demo_env_cfg(scan_hz=20)


def test_kinematic_replay_writes_advanced_frame_before_forwarding() -> None:
  command = object.__new__(KinematicReplayMotionCommand)
  command._all_env_ids = torch.tensor([0, 1])
  command.time_steps = torch.tensor([1, 2])
  command.motion = SimpleNamespace(
    time_step_total=3,
    body_pos_w=torch.arange(9, dtype=torch.float32).reshape(3, 1, 3),
    body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 3, dtype=torch.float32),
    body_lin_vel_w=torch.arange(9, dtype=torch.float32).reshape(3, 1, 3),
    body_ang_vel_w=torch.arange(9, dtype=torch.float32).reshape(3, 1, 3),
    joint_pos=torch.arange(6, dtype=torch.float32).reshape(3, 2),
    joint_vel=torch.arange(6, dtype=torch.float32).reshape(3, 2),
  )

  call_order: list[str] = []
  written: list[torch.Tensor] = []

  def record_write(*args: torch.Tensor) -> None:
    call_order.append("write")
    written.extend(args)

  command._write_reference_state_to_sim = record_write
  command.update_relative_body_poses = lambda: call_order.append("relative")
  command._env = SimpleNamespace(
    scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
    sim=SimpleNamespace(forward=lambda: call_order.append("forward")),
  )

  command._update_command()

  assert command.time_steps.tolist() == [2, 0]
  assert call_order == ["write", "forward", "relative"]
  assert torch.equal(written[0], command._all_env_ids)
  assert torch.equal(written[1], torch.tensor([[6.0, 7.0, 8.0], [0.0, 1.0, 2.0]]))
  assert torch.equal(written[5], torch.tensor([[4.0, 5.0], [0.0, 1.0]]))


def test_reference_filter_demo_wraps_replay_without_learning_contract_changes() -> None:
  replay = unitree_g1_kinematic_reference_lidar_demo_env_cfg()
  filtered = unitree_g1_reference_filter_lidar_demo_env_cfg()

  motion = filtered.commands["motion"]
  assert isinstance(motion, PlanarFilteredReplayMotionCommandCfg)
  assert motion.obstacle_entity_names == (
    "human",
    PRIMARY_HUMAN_ENTITY_NAME,
  )
  assert motion.planar_filter.safe_clearance_m == 0.1
  assert motion.link_filter.safe_clearance_m == 0.1
  assert motion.link_filter.max_joint_velocity_correction_rps == 1.5
  assert motion.link_filter.max_joint_position_residual_rad == 0.75
  assert motion.link_filter.max_arm_joint_position_residual_rad == 1.5
  assert "left_wrist_yaw_link" in motion.link_filter.body_names
  assert "right_ankle_roll_link" in motion.link_filter.body_names
  assert filtered.observations.keys() == replay.observations.keys()
  assert filtered.actions.keys() == replay.actions.keys()
  assert filtered.rewards.keys() == replay.rewards.keys()
  assert filtered.terminations == {}
  assert filtered.sim.mujoco.disableflags == replay.sim.mujoco.disableflags
  crowd = filtered.events["animate_human"]
  assert crowd.params["min_count"] == 0
  assert crowd.params["randomize_density"] is True
  assert crowd.params["min_radius_m"] == 2.0
  assert crowd.params["max_radius_m"] == 4.0
  assert crowd.params["radial_jitter_m"] == 0.25
  assert crowd.params["min_human_height_m"] == 1.3
  assert crowd.params["max_human_height_m"] == 1.9
  primary = filtered.events[PRIMARY_HUMAN_EVENT_NAME]
  assert primary.params["min_initial_spawn_radius_m"] == 2.0
  assert primary.params["max_initial_spawn_radius_m"] == 4.0
  assert primary.params["min_intersection_delay_s"] == 1.0
  assert primary.params["max_intersection_delay_s"] == 3.0
  assert primary.params["min_human_height_m"] == 1.3
  assert primary.params["max_human_height_m"] == 1.9
  assert set(filtered.metrics) == {
    "filter_cbf_violation_mps",
    "filter_clearance_violation_m",
    "filter_intervention_speed_mps",
    "filter_reference_offset_m",
    "joint_filter_intervention_rps",
    "joint_filter_reference_residual_rad",
    "joint_filter_standing_pull_rps",
    "link_filter_cbf_violation_mps",
    "link_filter_clearance_violation_m",
  }


def test_reference_filter_policy_demo_keeps_physics_and_checkpoint_contract() -> None:
  nominal = unitree_g1_nominal_lidar_debug_env_cfg(has_state_estimation=False)
  policy = unitree_g1_reference_filter_policy_lidar_demo_env_cfg()

  motion = policy.commands["motion"]
  assert isinstance(motion, PlanarFilteredReplayMotionCommandCfg)
  assert motion.write_reference_to_sim is False
  assert motion.align_reference_to_robot_each_step is True
  assert motion.sampling_mode == "start"
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.joint_position_range == (0.0, 0.0)
  assert policy.observations.keys() == nominal.observations.keys()
  for group_name in nominal.observations:
    assert (
      policy.observations[group_name].terms.keys()
      == nominal.observations[group_name].terms.keys()
    )
  assert policy.actions.keys() == nominal.actions.keys()
  assert policy.rewards.keys() == nominal.rewards.keys()
  assert policy.terminations.keys() == nominal.terminations.keys()
  assert policy.sim.mujoco.gravity == (0.0, 0.0, -9.81)
  assert "contact" not in policy.sim.mujoco.disableflags
  assert "actuation" not in policy.sim.mujoco.disableflags
  assert policy.sim.nconmax == 70
  assert policy.sim.njmax == 500
  assert "motion_anchor_pos_b" not in policy.observations["actor"].terms
  assert "base_lin_vel" not in policy.observations["actor"].terms
  assert "motion_anchor_pos_b" in policy.observations["critic"].terms
  assert "base_lin_vel" in policy.observations["critic"].terms

  lidar = next(
    sensor for sensor in policy.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )
  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.phases == 10
  assert lidar.scan_period == 0.2


def test_live_reference_alignment_matches_robot_xy_and_yaw() -> None:
  half_sqrt_two = math.sqrt(0.5)
  reference_anchor_pos = torch.tensor([[1.0, 2.0, 0.8]])
  reference_anchor_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  robot_anchor_pos = torch.tensor([[10.0, 20.0, 1.1]])
  robot_anchor_quat = torch.tensor([[half_sqrt_two, 0.0, 0.0, half_sqrt_two]])
  body_positions = torch.tensor([[[1.0, 2.0, 0.8], [2.0, 2.0, 0.4], [1.0, 3.0, 1.2]]])

  yaw_delta, aligned_anchor_xy = _planar_reference_alignment(
    reference_anchor_quat,
    robot_anchor_pos,
    robot_anchor_quat,
  )
  aligned = _apply_planar_reference_alignment(
    body_positions,
    reference_anchor_pos,
    aligned_anchor_xy,
    yaw_delta,
  )

  torch.testing.assert_close(
    aligned,
    torch.tensor([[[10.0, 20.0, 0.8], [10.0, 21.0, 0.4], [9.0, 20.0, 1.2]]]),
    atol=1e-6,
    rtol=1e-6,
  )
