"""Unitree G1 obstacle-aware motion imitation with perceptive sensing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from functools import partial
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)
from mjlab.tasks.velocity.mdp.terminations import illegal_contact
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from safe_mimic.assets import (
  HUMAN_CAPSULE_BODY_PREFIX,
  HUMAN_CROWD_BODY_PREFIX,
  HUMAN_CROWD_CAPACITY,
  MID360_SITE_NAME,
  get_g1_with_mid360_spec,
  get_soma_capsule_crowd_spec,
  get_soma_capsule_human_spec,
)
from safe_mimic.motions.human_capsules import SOMA_CROWD_PROXY_SPECS
from safe_mimic.sensing.held_scan import (
  HeldScanRayCastSensorCfg,
  InterleavedSphericalLidarPatternCfg,
)
from safe_mimic.sensing.observations import LidarNoiseCfg, normalized_lidar_ranges
from safe_mimic.tasks import mdp
from safe_mimic.tasks.human_capsule_event import HumanCapsuleMotion
from safe_mimic.tasks.human_crowd_event import HumanCapsuleCrowdMotion
from safe_mimic.tasks.kinematic_replay_command import (
  KinematicReplayMotionCommandCfg,
  PlanarFilteredReplayMotionCommandCfg,
)
from safe_mimic.tasks.packed_motion_command import PackedMotionCommandCfg

LIDAR_SENSOR_NAME = "lidar_360"
HUMAN_CONTACT_SENSOR_NAME = "human_contact"
HUMAN_ENTITY_NAME = "human"
HUMAN_MOTION_EVENT_NAME = "animate_human"
PRIMARY_HUMAN_ENTITY_NAME = "primary_human"
PRIMARY_HUMAN_EVENT_NAME = "animate_primary_human"
PRIMARY_HUMAN_CONTACT_SENSOR_NAME = "primary_human_contact"
LIDAR_AZIMUTH_SAMPLES = 180
# The physical Mid-360 is mounted inverted. Its useful obstacle-facing field is
# horizontal at the top and extends downward toward the robot's feet.
LIDAR_ELEVATIONS_DEG = (0.0, -10.0, -20.0, -30.0, -40.0, -50.0)
LIVOX_SNAPSHOT_AZIMUTH_SAMPLES = 185
LIVOX_SNAPSHOT_ELEVATION_SAMPLES = 27
LIVOX_SNAPSHOT_ELEVATIONS_DEG = tuple(
  -52.0 * index / (LIVOX_SNAPSHOT_ELEVATION_SAMPLES - 1)
  for index in range(LIVOX_SNAPSHOT_ELEVATION_SAMPLES)
)
LIDAR_MIN_DISTANCE_M = 0.3
LIDAR_MAX_DISTANCE_M = 5.0
LIDAR_SCAN_PERIOD_S = 0.1
LIDAR_SCAN_PHASES = 5
CROWD_CAPSULES_PER_PERSON = len(SOMA_CROWD_PROXY_SPECS)
CROWD_PRIVILEGED_NEAREST_PEOPLE = 8
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPSULE_BANK_PATH = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/capsule_path_bank_50hz_6s"
)
DEFAULT_TRANSITION_INDEX_PATH = DEFAULT_CAPSULE_BANK_PATH / "transition_index_v3"
DEFAULT_SKELETON_BANK_PATH = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/skeleton_path_bank_transition_v3"
)
DEFAULT_SOMA_MESH_SKIN_PATH = (
  _PROJECT_ROOT
  / "artifacts/bones-seed/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd"
)
DEFAULT_STANDING_ACTION_INDEX_PATH = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/standing_arm_actions_100"
)
DEFAULT_STANDING_SKELETON_BANK_PATH = (
  _PROJECT_ROOT
  / "artifacts/bones-seed/datasets/skeleton_path_bank_standing_arm_actions_100"
)
DEFAULT_G1_MOTION_LIBRARY_MANIFEST = (
  _PROJECT_ROOT
  / "artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1/"
  "conversion_manifest.jsonl"
)
DEFAULT_EXAMPLE_DANCE_MOTION_FILE = Path(
  "/tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz"
)
IMPLICIT_STATE_HISTORY_STEPS = 10


def unitree_g1_example_dance_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build example-dance tracking without state estimation."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
  )
  motion = cfg.commands["motion"]
  assert isinstance(motion, tracking_mdp.MotionCommandCfg)
  motion.motion_file = str(DEFAULT_EXAMPLE_DANCE_MOTION_FILE)
  return cfg


def unitree_g1_motion_library_tracking_env_cfg(
  play: bool = False,
  has_state_estimation: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build raw G1 mimic tracking over the packed general-motion library."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
  )
  base = cfg.commands["motion"]
  assert isinstance(base, tracking_mdp.MotionCommandCfg)
  packed = PackedMotionCommandCfg(
    **{field.name: getattr(base, field.name) for field in fields(base)},
    manifest_splits=("train",),
    adaptive_bin_duration_s=1.0,
    adaptive_group_key="sampling_pool",
    adaptive_pair_key="pair_id",
    adaptive_couple_pairs=True,
  )
  packed.motion_file = str(DEFAULT_G1_MOTION_LIBRARY_MANIFEST)
  # Unlike mjlab's flat-timeline failure-count EMA, this coefficient updates a
  # per-completed-episode failure-rate EMA, so it can react at clip-library scale.
  packed.adaptive_alpha = 0.05
  cfg.commands["motion"] = packed
  return cfg


def unitree_g1_motion_library_implicit_state_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the motion-library tracker with learned implicit state."""

  cfg = unitree_g1_motion_library_tracking_env_cfg(
    play=play,
    has_state_estimation=False,
  )
  return _add_implicit_state_estimator_observations(cfg)


def unitree_g1_example_dance_implicit_state_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the single example-dance tracker with learned implicit state."""

  cfg = unitree_g1_example_dance_tracking_env_cfg(play=play)
  return _add_implicit_state_estimator_observations(cfg)


def _add_implicit_state_estimator_observations(
  cfg: ManagerBasedRlEnvCfg,
) -> ManagerBasedRlEnvCfg:
  """Add training-only targets and deployable proprioceptive history.

  The actor retains the exact no-state-estimation reference contract: it does
  not observe the reference-anchor position or measured base linear velocity.
  A history encoder predicts body linear velocity, reference-root position
  error, and a dynamics latent trained against actual successor
  proprioception. Simulator values and the target encoder are training-only.
  No local reference velocity is added in this ablation.
  """
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  proprio_terms = {
    "base_ang_vel": deepcopy(actor_terms["base_ang_vel"]),
    "projected_gravity": ObservationTermCfg(
      func=tracking_mdp.projected_gravity_from_sensor,
      params={"sensor_name": "robot/imu_upvector"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": deepcopy(actor_terms["joint_pos"]),
    "joint_vel": deepcopy(actor_terms["joint_vel"]),
    "actions": deepcopy(actor_terms["actions"]),
  }
  cfg.observations["proprio_history"] = ObservationGroupCfg(
    terms=deepcopy(proprio_terms),
    concatenate_terms=True,
    enable_corruption=True,
    history_length=IMPLICIT_STATE_HISTORY_STEPS,
    flatten_history_dim=True,
  )
  cfg.observations["implicit_proprio_target"] = ObservationGroupCfg(
    terms=deepcopy(proprio_terms),
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["implicit_state_target"] = ObservationGroupCfg(
    terms={
      "base_lin_vel": deepcopy(critic_terms["base_lin_vel"]),
      "motion_anchor_pos_b": deepcopy(critic_terms["motion_anchor_pos_b"]),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


def _human_motion_event_cfg(*, show_mesh: bool = False) -> EventTermCfg:
  return EventTermCfg(
    func=HumanCapsuleCrowdMotion,
    mode="step",
    params={
      "human_entity": HUMAN_ENTITY_NAME,
      "command_name": "motion",
      "robot_entity": "robot",
      "capacity": HUMAN_CROWD_CAPACITY,
      "min_count": 0,
      "max_count": HUMAN_CROWD_CAPACITY,
      "min_radius_m": 2.0,
      "max_radius_m": 4.0,
      "target_arc_spacing_m": 0.62,
      "randomize_density": True,
      "radial_jitter_m": 0.25,
      "min_shape_exponent": 2.0,
      "max_shape_exponent": 8.0,
      "angular_jitter_fraction": 0.0,
      "inward_facing_probability": 1.0,
      "inward_facing_jitter_rad": 0.0,
      "min_human_height_m": 1.3,
      "max_human_height_m": 1.9,
      "min_playback_speed": 0.8,
      "max_playback_speed": 1.2,
      "skeleton_bank_path": DEFAULT_STANDING_SKELETON_BANK_PATH,
      "transition_index_path": DEFAULT_STANDING_ACTION_INDEX_PATH,
      "update_hz": 10.0,
      "mesh_update_hz": 5.0,
      "mesh_voxel_size_m": 0.02,
      "transition_duration_s": 0.2,
      "min_intersection_delay_s": 1.0,
      "max_intersection_delay_s": 3.0,
      "show_mesh": show_mesh,
      "mesh_skin_path": DEFAULT_SOMA_MESH_SKIN_PATH,
    },
  )


def _primary_human_motion_event_cfg(*, show_mesh: bool) -> EventTermCfg:
  return EventTermCfg(
    func=HumanCapsuleMotion,
    mode="step",
    params={
      "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
      "command_name": "motion",
      "skeleton_bank_path": DEFAULT_SKELETON_BANK_PATH,
      "transition_index_path": DEFAULT_TRANSITION_INDEX_PATH,
      "update_hz": 50.0,
      "transition_duration_s": 0.2,
      "min_intersection_delay_s": 1.0,
      "max_intersection_delay_s": 3.0,
      "min_initial_spawn_radius_m": 2.0,
      "max_initial_spawn_radius_m": 4.0,
      "min_human_height_m": 1.3,
      "max_human_height_m": 1.9,
      "show_mesh": show_mesh,
      "mesh_skin_path": DEFAULT_SOMA_MESH_SKIN_PATH,
    },
  )


def _full_scene_demo_event_cfgs(
  *, show_mesh: bool
) -> tuple[EventTermCfg, EventTermCfg]:
  """Build debug events with the complete training scene distribution."""

  crowd_event = _human_motion_event_cfg(show_mesh=show_mesh)
  crowd_event.params["update_hz"] = 5.0
  crowd_event.params["mesh_update_hz"] = 5.0
  # The SOMA skin's apparent forward axis is 90 degrees counterclockwise from
  # the placement convention used by the sampler.
  crowd_event.params["facing_yaw_offset_rad"] = -1.5707963267948966
  return crowd_event, _primary_human_motion_event_cfg(show_mesh=show_mesh)


def _primary_human_contact_cfg() -> ContactSensorCfg:
  return ContactSensorCfg(
    name=PRIMARY_HUMAN_CONTACT_SENSOR_NAME,
    primary=ContactMatch(
      mode="body",
      pattern=rf"{HUMAN_CAPSULE_BODY_PREFIX}.*",
      entity=PRIMARY_HUMAN_ENTITY_NAME,
    ),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )


def _mid360_lidar_cfg(
  *,
  debug_vis: bool = False,
  livox_snapshot_density: bool = False,
  scan_period: float = LIDAR_SCAN_PERIOD_S,
  scan_phases: int = LIDAR_SCAN_PHASES,
) -> RayCastSensorCfg:
  """Build the shared URDF-mounted Mid-360 ray sensor."""
  azimuth_samples = (
    LIVOX_SNAPSHOT_AZIMUTH_SAMPLES if livox_snapshot_density else LIDAR_AZIMUTH_SAMPLES
  )
  elevation_angles_deg = (
    LIVOX_SNAPSHOT_ELEVATIONS_DEG if livox_snapshot_density else LIDAR_ELEVATIONS_DEG
  )
  lidar_pattern = InterleavedSphericalLidarPatternCfg(
    azimuth_samples=azimuth_samples,
    elevation_angles_deg=elevation_angles_deg,
    phases=scan_phases,
    # Origins are exactly at the attached Mid-360 site.
    origin_offset=(0.0, 0.0, 0.0),
  )
  return HeldScanRayCastSensorCfg(
    name=LIDAR_SENSOR_NAME,
    frame=ObjRef(type="site", name=MID360_SITE_NAME, entity="robot"),
    # Match a physical body-mounted sensor. "yaw" would hide roll and pitch.
    ray_alignment="base",
    pattern=lidar_pattern,
    scan_period=scan_period,
    min_distance=LIDAR_MIN_DISTANCE_M,
    max_distance=LIDAR_MAX_DISTANCE_M,
    # Do not exclude torso_link: it, the arms, and the legs must cast the same
    # self-shadow seen by the hardware.  The coarse head sphere containing the
    # optical origin is placed in a separate group by get_g1_with_mid360_spec.
    exclude_parent_body=False,
    # Obstacles/ground are group 0; G1 collision proxies are group 3.  Primitive
    # proxies are both cleaner and cheaper to ray cast than the visual meshes.
    include_geom_groups=(0, 3),
    debug_vis=debug_vis,
    viz=RayCastSensorCfg.VizCfg(
      # Hit markers are enough for visual calibration; thousands of arrows
      # obscure the robot and its self-occlusion proxies.
      show_rays=False,
      hit_sphere_color=(1.0, 0.0, 0.0, 1.0),
      hit_sphere_radius=0.3,
    ),
  )


def unitree_g1_obstacle_aware_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build flat-ground G1 imitation where global motion may avoid obstacles.

  Relative body-pose rewards retain the reference motion's style while weakened
  global-anchor rewards let the root translate and yaw around an obstacle. This
  is a new actor input contract: an unmodified upstream checkpoint cannot consume
  the LiDAR term directly.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_g1_with_mid360_spec
  cfg.scene.entities[HUMAN_ENTITY_NAME] = EntityCfg(
    spec_fn=get_soma_capsule_crowd_spec,
  )
  cfg.scene.extent = 5.0

  lidar = _mid360_lidar_cfg()
  human_contact = ContactSensorCfg(
    name=HUMAN_CONTACT_SENSOR_NAME,
    primary=ContactMatch(
      mode="body",
      pattern=rf"{HUMAN_CROWD_BODY_PREFIX}.*",
      entity=HUMAN_ENTITY_NAME,
    ),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (lidar, human_contact)

  actor_lidar = ObservationTermCfg(
    func=normalized_lidar_ranges,
    params={"sensor_name": LIDAR_SENSOR_NAME},
    noise=LidarNoiseCfg(
      azimuth_samples=LIDAR_AZIMUTH_SAMPLES,
      sector_width_samples=LIDAR_AZIMUTH_SAMPLES // 4,
    ),
    clip=(0.0, 1.0),
    # The tracking loop is 50 Hz: 20-60 ms of simulated end-to-end latency.
    delay_min_lag=1,
    delay_max_lag=3,
    delay_hold_prob=0.8,
    history_length=2,
  )
  critic_lidar = ObservationTermCfg(
    func=normalized_lidar_ranges,
    params={"sensor_name": LIDAR_SENSOR_NAME},
    clip=(0.0, 1.0),
  )
  cfg.observations["actor"].terms[LIDAR_SENSOR_NAME] = actor_lidar
  cfg.observations["critic"].terms[LIDAR_SENSOR_NAME] = critic_lidar
  cfg.observations["critic"].terms["human_capsule_vectors_b"] = ObservationTermCfg(
    func=mdp.human_capsule_vectors_b,
    params={
      "robot_entity": "robot",
      "human_entity": HUMAN_ENTITY_NAME,
      "max_distance": LIDAR_MAX_DISTANCE_M,
      "capsules_per_group": CROWD_CAPSULES_PER_PERSON,
      "nearest_groups": CROWD_PRIVILEGED_NEAREST_PEOPLE,
    },
    clip=(-1.0, 1.0),
  )

  cfg.events[HUMAN_MOTION_EVENT_NAME] = _human_motion_event_cfg(show_mesh=play)

  # Preserve pose and rhythm strongly. Global root matching is deliberately soft:
  # it pulls the robot back toward the reference path only when clearance permits.
  cfg.rewards["motion_global_root_pos"].weight = 0.15
  cfg.rewards["motion_global_root_ori"].weight = 0.15
  cfg.rewards["human_proximity"] = RewardTermCfg(
    func=mdp.human_capsule_proximity_penalty,
    weight=-3.0,
    params={
      "robot_entity": "robot",
      "human_entity": HUMAN_ENTITY_NAME,
      "safe_clearance": 0.1,
      "robot_radius": 0.35,
    },
  )
  cfg.rewards["human_collision"] = RewardTermCfg(
    func=tracking_mdp.self_collision_cost,
    weight=-20.0,
    params={
      "sensor_name": HUMAN_CONTACT_SENSOR_NAME,
      "force_threshold": 15.0,
    },
  )
  cfg.terminations["human_collision"] = TerminationTermCfg(
    func=illegal_contact,
    params={
      "sensor_name": HUMAN_CONTACT_SENSOR_NAME,
      "force_threshold": 30.0,
    },
  )

  cfg.sim.nconmax = 70
  cfg.sim.njmax = 500
  cfg.sim.mujoco.ccd_iterations = 500
  return cfg


def unitree_g1_crowd_and_human_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the optimized dense crowd plus action-human training task.

  The five-capsule crowd remains identical for LiDAR and analytical proximity,
  but uses no physical contacts and stores all ray-only geoms under one mocap
  body. The separate 18-capsule crossing/fighting human remains collidable.
  """

  cfg = unitree_g1_obstacle_aware_tracking_env_cfg(play=play)
  cfg.scene.entities[HUMAN_ENTITY_NAME] = EntityCfg(
    spec_fn=partial(get_soma_capsule_crowd_spec, collidable=False)
  )
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != HUMAN_CONTACT_SENSOR_NAME
  )
  cfg.rewards.pop("human_collision")
  cfg.terminations.pop("human_collision")
  # The clean critic LiDAR already contains the crowd silhouette. Removing the
  # redundant per-capsule vectors reduces both observation assembly and network
  # input while leaving the policy's sensing contract unchanged.
  cfg.observations["critic"].terms.pop("human_capsule_vectors_b")
  crowd_event = cfg.events[HUMAN_MOTION_EVENT_NAME]
  crowd_event.params["update_hz"] = 5.0
  crowd_event.params["mesh_update_hz"] = 5.0

  cfg.scene.entities[PRIMARY_HUMAN_ENTITY_NAME] = EntityCfg(
    spec_fn=get_soma_capsule_human_spec
  )
  cfg.events[PRIMARY_HUMAN_EVENT_NAME] = _primary_human_motion_event_cfg(show_mesh=play)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (_primary_human_contact_cfg(),)
  cfg.observations["critic"].terms["primary_human_capsule_vectors_b"] = (
    ObservationTermCfg(
      func=mdp.human_capsule_vectors_b,
      params={
        "robot_entity": "robot",
        "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
        "max_distance": LIDAR_MAX_DISTANCE_M,
      },
      clip=(-1.0, 1.0),
    )
  )
  cfg.rewards["primary_human_proximity"] = RewardTermCfg(
    func=mdp.human_capsule_proximity_penalty,
    weight=-3.0,
    params={
      "robot_entity": "robot",
      "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
      "safe_clearance": 0.1,
      "robot_radius": 0.35,
    },
  )
  cfg.rewards["primary_human_collision"] = RewardTermCfg(
    func=tracking_mdp.self_collision_cost,
    weight=-20.0,
    params={
      "sensor_name": PRIMARY_HUMAN_CONTACT_SENSOR_NAME,
      "force_threshold": 15.0,
    },
  )
  return cfg


def unitree_g1_nominal_lidar_debug_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the upstream nominal tracker with a visualization-only LiDAR.

  Neither actor nor critic observations are changed, so upstream tracking
  checkpoints remain strictly input-compatible. The animated human only gives
  the debug rays useful geometry to hit; the nominal policy cannot react to it.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_g1_with_mid360_spec
  cfg.scene.entities[HUMAN_ENTITY_NAME] = EntityCfg(
    spec_fn=partial(get_soma_capsule_crowd_spec, collidable=False),
  )
  crowd_event, primary_event = _full_scene_demo_event_cfgs(show_mesh=play)
  # Preserve the fixed nominal calibration scene. The reference-filter demo
  # deliberately restores the full distribution below.
  crowd_event.params["min_count"] = 30
  crowd_event.params["randomize_density"] = False
  crowd_event.params["min_radius_m"] = 3.0
  crowd_event.params["max_radius_m"] = 3.001
  crowd_event.params["radial_jitter_m"] = 0.0
  cfg.events[HUMAN_MOTION_EVENT_NAME] = crowd_event
  cfg.scene.entities[PRIMARY_HUMAN_ENTITY_NAME] = EntityCfg(
    spec_fn=get_soma_capsule_human_spec
  )
  primary_event.params["min_initial_spawn_radius_m"] = 3.0
  primary_event.params["max_initial_spawn_radius_m"] = 3.0
  primary_event.params["min_intersection_delay_s"] = 3.0
  primary_event.params["max_intersection_delay_s"] = 3.0
  cfg.events[PRIMARY_HUMAN_EVENT_NAME] = primary_event
  cfg.scene.extent = LIDAR_MAX_DISTANCE_M
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    _mid360_lidar_cfg(debug_vis=True, livox_snapshot_density=True),
  )
  return cfg


def unitree_g1_kinematic_reference_lidar_demo_env_cfg(
  play: bool = False,
  scan_hz: int = 5,
) -> ManagerBasedRlEnvCfg:
  """Replay the exact reference in the dense human-and-LiDAR debug scene.

  Physics and policy actions remain in the environment interface for checkpoint
  compatibility, but the motion command overwrites the robot state immediately
  before sensing. ``scan_hz=5`` is the dense default; ``scan_hz=10`` retains the
  existing faster publication option.
  """
  if scan_hz not in (5, 10):
    raise ValueError(f"scan_hz must be 5 or 10, got {scan_hz}")

  cfg = unitree_g1_nominal_lidar_debug_env_cfg(play=play)

  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, tracking_mdp.MotionCommandCfg)
  replay_cfg = KinematicReplayMotionCommandCfg(
    **{field.name: getattr(motion_cfg, field.name) for field in fields(motion_cfg)}
  )
  replay_cfg.pose_range = {}
  replay_cfg.velocity_range = {}
  replay_cfg.joint_position_range = (0.0, 0.0)
  replay_cfg.sampling_mode = "start"
  cfg.commands["motion"] = replay_cfg

  # The 50 Hz control loop casts one interleaved phase per step. Ten phases
  # produce a complete 5 Hz scan; five phases produce the optional 10 Hz scan.
  scan_phases = 50 // scan_hz
  replay_lidar = _mid360_lidar_cfg(
    debug_vis=True,
    livox_snapshot_density=True,
    scan_period=1.0 / scan_hz,
    scan_phases=scan_phases,
  )
  cfg.scene.sensors = tuple(
    replay_lidar if sensor.name == LIDAR_SENSOR_NAME else sensor
    for sensor in (cfg.scene.sensors or ())
  )

  cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)
  cfg.sim.mujoco.disableflags = tuple(
    dict.fromkeys((*cfg.sim.mujoco.disableflags, "contact", "actuation"))
  )
  cfg.events.pop("push_robot", None)
  cfg.terminations.clear()
  return cfg


def unitree_g1_reference_filter_lidar_demo_env_cfg(
  play: bool = False,
  scan_hz: int = 5,
) -> ManagerBasedRlEnvCfg:
  """Build the privileged planar-reference-filter tuning environment."""
  cfg = unitree_g1_kinematic_reference_lidar_demo_env_cfg(play=play, scan_hz=scan_hz)
  crowd_event, primary_event = _full_scene_demo_event_cfgs(show_mesh=play)
  cfg.events[HUMAN_MOTION_EVENT_NAME] = crowd_event
  cfg.events[PRIMARY_HUMAN_EVENT_NAME] = primary_event
  replay_cfg = cfg.commands["motion"]
  assert isinstance(replay_cfg, KinematicReplayMotionCommandCfg)
  filtered_command_cfg = PlanarFilteredReplayMotionCommandCfg(
    **{field.name: getattr(replay_cfg, field.name) for field in fields(replay_cfg)},
    obstacle_entity_names=(HUMAN_ENTITY_NAME, PRIMARY_HUMAN_ENTITY_NAME),
  )
  cfg.commands["motion"] = filtered_command_cfg
  metric_params = {"command_name": "motion"}
  cfg.metrics["filter_intervention_speed_mps"] = MetricsTermCfg(
    func=mdp.command_metric,
    params={
      **metric_params,
      "metric_name": "filter_intervention_speed_mps",
    },
    reduce="mean",
  )
  cfg.metrics["filter_reference_offset_m"] = MetricsTermCfg(
    func=mdp.command_metric,
    params={**metric_params, "metric_name": "filter_reference_offset_m"},
    reduce="max",
  )
  cfg.metrics["filter_cbf_violation_mps"] = MetricsTermCfg(
    func=mdp.command_metric,
    params={**metric_params, "metric_name": "filter_cbf_violation_mps"},
    reduce="max",
  )
  cfg.metrics["filter_clearance_violation_m"] = MetricsTermCfg(
    func=mdp.reference_filter_clearance_violation,
    params={
      **metric_params,
      "safe_clearance_m": filtered_command_cfg.planar_filter.safe_clearance_m,
    },
    reduce="max",
  )
  cfg.metrics["joint_filter_intervention_rps"] = MetricsTermCfg(
    func=mdp.command_metric,
    params={
      **metric_params,
      "metric_name": "joint_filter_intervention_rps",
    },
    reduce="mean",
  )
  cfg.metrics["joint_filter_reference_residual_rad"] = MetricsTermCfg(
    func=mdp.command_metric,
    params={
      **metric_params,
      "metric_name": "joint_filter_reference_residual_rad",
    },
    reduce="max",
  )
  cfg.metrics["joint_filter_standing_pull_rps"] = MetricsTermCfg(
    func=mdp.command_metric,
    params={
      **metric_params,
      "metric_name": "joint_filter_standing_pull_rps",
    },
    reduce="mean",
  )
  cfg.metrics["link_filter_cbf_violation_mps"] = MetricsTermCfg(
    func=mdp.command_metric,
    params={
      **metric_params,
      "metric_name": "link_filter_cbf_violation_mps",
    },
    reduce="max",
  )
  cfg.metrics["link_filter_clearance_violation_m"] = MetricsTermCfg(
    func=mdp.reference_filter_clearance_violation,
    params={
      **metric_params,
      "safe_clearance_m": filtered_command_cfg.link_filter.safe_clearance_m,
      "metric_name": "link_filter_minimum_clearance_m",
    },
    reduce="max",
  )
  return cfg


def unitree_g1_reference_filter_policy_lidar_demo_env_cfg(
  play: bool = False,
  scan_hz: int = 5,
) -> ManagerBasedRlEnvCfg:
  """Track the privileged filtered reference with the nominal policy.

  This keeps the upstream actor's 160-value observation contract and all
  physics, contacts, actuation, and tracking terminations.  Unlike the exact
  replay demo, the filter changes only the desired motion command and never
  writes the filtered pose into MuJoCo during a policy step.
  """
  if scan_hz not in (5, 10):
    raise ValueError(f"scan_hz must be 5 or 10, got {scan_hz}")

  cfg = unitree_g1_nominal_lidar_debug_env_cfg(play=play)
  # The nominal flat task sizes these buffers for the robot alone.  The live
  # 18-capsule action human can transiently add roughly forty broadphase pairs.
  cfg.sim.nconmax = 70
  cfg.sim.njmax = 500
  crowd_event, primary_event = _full_scene_demo_event_cfgs(show_mesh=play)
  cfg.events[HUMAN_MOTION_EVENT_NAME] = crowd_event
  cfg.events[PRIMARY_HUMAN_EVENT_NAME] = primary_event

  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, tracking_mdp.MotionCommandCfg)
  filtered_command_cfg = PlanarFilteredReplayMotionCommandCfg(
    **{field.name: getattr(motion_cfg, field.name) for field in fields(motion_cfg)},
    obstacle_entity_names=(HUMAN_ENTITY_NAME, PRIMARY_HUMAN_ENTITY_NAME),
    write_reference_to_sim=False,
  )
  filtered_command_cfg.pose_range = {}
  filtered_command_cfg.velocity_range = {}
  filtered_command_cfg.joint_position_range = (0.0, 0.0)
  filtered_command_cfg.sampling_mode = "start"
  cfg.commands["motion"] = filtered_command_cfg

  scan_phases = 50 // scan_hz
  policy_lidar = _mid360_lidar_cfg(
    debug_vis=True,
    livox_snapshot_density=True,
    scan_period=1.0 / scan_hz,
    scan_phases=scan_phases,
  )
  cfg.scene.sensors = tuple(
    policy_lidar if sensor.name == LIDAR_SENSOR_NAME else sensor
    for sensor in (cfg.scene.sensors or ())
  )
  return cfg
