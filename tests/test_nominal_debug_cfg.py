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
  LIDAR_AZIMUTH_SAMPLES,
  LIDAR_ELEVATIONS_DEG,
  LIDAR_MAX_DISTANCE_M,
  LIDAR_MIN_DISTANCE_M,
  LIDAR_SCAN_PERIOD_S,
  LIDAR_SCAN_PHASES,
  LIDAR_SENSOR_NAME,
  LIVOX_SNAPSHOT_AZIMUTH_SAMPLES,
  LIVOX_SNAPSHOT_ELEVATION_SAMPLES,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  unitree_g1_kinematic_reference_lidar_demo_env_cfg,
  unitree_g1_nominal_lidar_debug_env_cfg,
  unitree_g1_obstacle_aware_tracking_env_cfg,
  unitree_g1_reference_filter_lidar_demo_env_cfg,
  unitree_g1_reference_filter_policy_lidar_demo_env_cfg,
)
from safe_mimic.tasks.kinematic_replay_command import (
  KinematicReplayMotionCommand,
  KinematicReplayMotionCommandCfg,
  PlanarFilteredReplayMotionCommandCfg,
)

EXPECTED_HEAD_SIDE_HALF_SIZE = (0.011, 0.0065, 0.0375)
EXPECTED_HEAD_SIDE_X = 0.0402835
EXPECTED_HEAD_SIDE_Z = 0.37868


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
  nominal = unitree_g1_nominal_lidar_debug_env_cfg()
  policy = unitree_g1_reference_filter_policy_lidar_demo_env_cfg()

  motion = policy.commands["motion"]
  assert isinstance(motion, PlanarFilteredReplayMotionCommandCfg)
  assert motion.write_reference_to_sim is False
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

  lidar = next(
    sensor for sensor in policy.scene.sensors if sensor.name == LIDAR_SENSOR_NAME
  )
  assert isinstance(lidar, HeldScanRayCastSensorCfg)
  assert lidar.pattern.phases == 10
  assert lidar.scan_period == 0.2
