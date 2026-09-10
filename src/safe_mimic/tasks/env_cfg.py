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
from safe_mimic.sensing.observations import (
  BlindDirectionalHeldLidarRangeRate,
  CachedDirectionalHeldLidarRangeRate,
  CachedDirectionalHeldLidarScanPair,
  CachedDirectionalLidarRanges,
  LidarNoiseCfg,
  held_lidar_scan_age,
  normalized_lidar_ranges,
)
from safe_mimic.tasks import mdp
from safe_mimic.tasks.human_capsule_event import HumanCapsuleMotion
from safe_mimic.tasks.human_crowd_event import HumanCapsuleCrowdMotion
from safe_mimic.tasks.kinematic_replay_command import (
  KinematicReplayMotionCommandCfg,
  PlanarFilteredReplayMotionCommandCfg,
)
from safe_mimic.tasks.packed_human_event import (
  PackedHumanCapsuleCrowdMotion,
  PackedHumanCapsuleMotion,
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
PROTOTYPE_LIDAR_AZIMUTH_SAMPLES = 120
PROTOTYPE_LIDAR_ELEVATIONS_DEG = (0.0, -10.0, -25.0, -45.0)
LIDAR_MIN_DISTANCE_M = 0.3
LIDAR_MAX_DISTANCE_M = 5.0
LIDAR_SCAN_PERIOD_S = 0.1
LIDAR_SCAN_PHASES = 5
CROWD_CAPSULES_PER_PERSON = len(SOMA_CROWD_PROXY_SPECS)
CROWD_PRIVILEGED_NEAREST_PEOPLE = 8

# Slow-regime encounter distribution (user direction 2026-09-04): the walking
# human spawns outside the critical distance and approaches no faster than the
# robot's sustained planar speed. joint@16k slow-bin anatomy: every collision
# that started outside the 0.8 m safe clearance had a root distance >= 1.4 m
# and the human's swinging limbs reach ~0.5 m from its root. Spawn audit on the
# packed training events (1024 envs x 4 resets): 1.6 m leaves 97.2 % of spawns
# outside the 0.8 m clearance, 1.8 m leaves 99.7 % (min 0.54, p1 0.88 m) while
# 98.3 % of effective approach speeds stay <= 0.75 m/s (p99 0.76); 2.0 m gives
# 100 % clearance but the radius clamp pushes 7 % of speeds past 0.75. Speed bins stop
# at 0.75 m/s (robot cap median 0.60, sustained plateau ~0.6). TTC bins and the
# delay range let a 4 m spawn at 0.5 m/s still arrive inside the 10 s episode.
SLOW_REGIME_MIN_SPAWN_RADIUS_M = 1.8
SLOW_REGIME_SPEED_EDGES_MPS = (0.25, 0.5, 0.75)
SLOW_REGIME_MAX_SPEED_MPS = SLOW_REGIME_SPEED_EDGES_MPS[-1]
SLOW_REGIME_TTC_EDGES_S = (2.5, 4.0, 6.0, 8.0)
SLOW_REGIME_DELAY_RANGE_S = (SLOW_REGIME_TTC_EDGES_S[0], SLOW_REGIME_TTC_EDGES_S[-1])
# Dense variant (2026-09-04, after the slow@15k gate showed 0.7 training
# collisions per batch): a 5 s TTC cap fits two to three encounters into the
# 10 s episode instead of one. Radius clamp 1.8-4.0 keeps effective speeds
# under 0.75 m/s (0.25 m/s at 2.5 s -> 0.625 m clamped to 1.8 -> 0.72 m/s).
# The crowd's obstacle-free probability is deliberately NOT changed.
SLOW_REGIME_DENSE_TTC_EDGES_S = (2.5, 3.5, 5.0)
SLOW_REGIME_DENSE_DELAY_RANGE_S = (
  SLOW_REGIME_DENSE_TTC_EDGES_S[0],
  SLOW_REGIME_DENSE_TTC_EDGES_S[-1],
)
# ee_body_pos bound while the link filter is actively correcting that limb.
# NOTE (slow@15k gate): gating this termination on the corrected limb removed
# the main training pressure for arm compliance (state t 0.856 -> 0.618); the
# flag stays available but the registered tasks after Leash-Slow keep the
# strict stock termination.
EE_LOOSENED_THRESHOLD_M = 0.5
# Lag-aware ee_body_pos (2026-09-06): bound = 0.25 m + LAG_TIME * |reference
# vertical speed|, capped. 0.2 s is roughly the actor's observed arm latency;
# the cap keeps a runaway target from disabling the check.
EE_LAG_TIME_S = 0.2
EE_LAG_MAX_THRESHOLD_M = 0.6
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
DEFAULT_PACKED_PRIMARY_HUMAN_BANK_PATH = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/packed_primary_human_trajectory_v1"
)
DEFAULT_PACKED_CROWD_HUMAN_BANK_PATH = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/packed_crowd_human_trajectory_v1"
)
DEFAULT_G1_MOTION_LIBRARY_MANIFEST = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1/"
  "conversion_manifest.jsonl"
)
# G1-retargeted ballet library (348 clips, median 6 s). Used whole (every
# split) as the reference for the Ballet tasks; clips shorter than the episode
# chain into a fresh clip when they end. Since 2026-09-09 the STANCE-TRIMMED
# copy is the default: at most ~1 s of standing kept before the first and after
# the last active frame of each clip (scripts/trim_motion_stance.py, 2212 s ->
# 1942 s). The untrimmed original stays at g1_ballet_v1/ballet.yaml.
DEFAULT_G1_BALLET_MANIFEST = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/g1_ballet_v1_trim1s/ballet.yaml"
)
UNTRIMMED_G1_BALLET_MANIFEST = (
  _PROJECT_ROOT / "artifacts/bones-seed/datasets/g1_ballet_v1/ballet.yaml"
)
DEFAULT_EXAMPLE_DANCE_MOTION_FILE = (
  _PROJECT_ROOT / "artifacts/motions/lafan1_dance1_subject1_demo_motion.npz"
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


def _primary_human_motion_event_cfg(
  *, show_mesh: bool, print_velocity: bool = False
) -> EventTermCfg:
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
      # Console velocity diagnostic: only the full-scene demo tasks ask for
      # it; play of the avoidance tasks prints just the episode-end reasons.
      "print_velocity": print_velocity,
      "velocity_print_interval_s": 0.5,
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
  return crowd_event, _primary_human_motion_event_cfg(
    show_mesh=show_mesh, print_velocity=show_mesh
  )


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

  The upstream tracking rewards and their weights are retained. Human proximity
  and contact terms add obstacle awareness. This is a new actor input contract:
  an unmodified upstream checkpoint cannot consume the LiDAR term directly.
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


def unitree_g1_lidar_avoidance_tracking_env_cfg(
  play: bool = False,
  *,
  lidar_azimuth_samples: int = LIVOX_SNAPSHOT_AZIMUTH_SAMPLES,
  lidar_elevation_angles_deg: tuple[float, ...] = LIVOX_SNAPSHOT_ELEVATIONS_DEG,
  actor_azimuth_bins: int = 120,
  actor_elevation_bins: int = 9,
) -> ManagerBasedRlEnvCfg:
  """Build the from-scratch no-state LiDAR avoidance training task.

  The actor receives the nominal live-aligned mimic command plus current and
  previous 10 Hz quarter-resolution scans. A privileged CBF produces safe
  planar and joint targets only for the critic and rewards; it is deliberately
  hidden from the actor so avoidance must be inferred from LiDAR. Both human
  populations are ray-only; analytical capsule clearance terminates an episode
  before contact, without adding human contacts to MuJoCo's constraint solve.
  """
  cfg = unitree_g1_crowd_and_human_tracking_env_cfg(play=play)

  cfg.scene.entities[PRIMARY_HUMAN_ENTITY_NAME] = EntityCfg(
    spec_fn=partial(get_soma_capsule_human_spec, collidable=False)
  )
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != PRIMARY_HUMAN_CONTACT_SENSOR_NAME
  )
  cfg.rewards.pop("primary_human_collision")

  actor_terms = cfg.observations["actor"].terms
  actor_terms.pop("motion_anchor_pos_b")
  actor_terms.pop("base_lin_vel")
  actor_terms.pop(LIDAR_SENSOR_NAME)

  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, tracking_mdp.MotionCommandCfg)
  filtered_command_cfg = PlanarFilteredReplayMotionCommandCfg(
    **{field.name: getattr(motion_cfg, field.name) for field in fields(motion_cfg)},
    obstacle_entity_names=(HUMAN_ENTITY_NAME, PRIMARY_HUMAN_ENTITY_NAME),
    link_filter_capsules_per_group=(CROWD_CAPSULES_PER_PERSON, None),
    link_filter_nearest_groups=(CROWD_PRIVILEGED_NEAREST_PEOPLE, None),
    write_reference_to_sim=False,
    align_reference_to_robot_each_step=True,
    expose_filtered_command=False,
  )
  filtered_command_cfg.motion_file = str(DEFAULT_EXAMPLE_DANCE_MOTION_FILE)
  filtered_command_cfg.planar_filter.safe_clearance_m = 0.8
  filtered_command_cfg.link_filter.safe_clearance_m = 0.8
  filtered_command_cfg.planar_filter.max_intervention_speed_mps = 2.0
  filtered_command_cfg.planar_filter.max_planar_speed_mps = 2.3
  # Random start frames: episodes cover the whole clip instead of replaying
  # its first ``episode_length_s`` seconds from frame zero every reset.
  # Play mode pins "start" to match upstream mjlab's play convention.
  filtered_command_cfg.sampling_mode = "start" if play else "uniform"
  cfg.commands["motion"] = filtered_command_cfg

  lidar = _mid360_lidar_cfg(
    debug_vis=play,
    scan_period=LIDAR_SCAN_PERIOD_S,
    scan_phases=LIDAR_SCAN_PHASES,
  )
  lidar.pattern = InterleavedSphericalLidarPatternCfg(
    azimuth_samples=lidar_azimuth_samples,
    elevation_angles_deg=lidar_elevation_angles_deg,
    phases=LIDAR_SCAN_PHASES,
    origin_offset=(0.0, 0.0, 0.0),
  )
  cfg.scene.sensors = tuple(
    lidar if sensor.name == LIDAR_SENSOR_NAME else sensor
    for sensor in (cfg.scene.sensors or ())
  )

  cfg.observations["lidar"] = ObservationGroupCfg(
    terms={
      "directional_scan_pair": ObservationTermCfg(
        func=CachedDirectionalHeldLidarScanPair,
        params={
          "sensor_name": LIDAR_SENSOR_NAME,
          "azimuth_samples": lidar_azimuth_samples,
          "elevation_samples": len(lidar_elevation_angles_deg),
          "azimuth_bins": actor_azimuth_bins,
          "elevation_bins": actor_elevation_bins,
          "noise_cfg": None
          if play
          else LidarNoiseCfg(
            azimuth_samples=lidar_azimuth_samples,
            sector_width_samples=lidar_azimuth_samples // 4,
          ),
        },
        clip=(0.0, 1.0),
      ),
      "scan_age": ObservationTermCfg(
        func=held_lidar_scan_age,
        params={"sensor_name": LIDAR_SENSOR_NAME},
        clip=(0.0, 1.0),
      ),
    },
    concatenate_terms=True,
    enable_corruption=not play,
  )

  critic_terms = cfg.observations["critic"].terms
  critic_terms[LIDAR_SENSOR_NAME] = ObservationTermCfg(
    func=CachedDirectionalLidarRanges,
    params={
      "sensor_name": LIDAR_SENSOR_NAME,
      "azimuth_samples": lidar_azimuth_samples,
      "elevation_samples": len(lidar_elevation_angles_deg),
      "azimuth_bins": 24,
      "elevation_bins": 3,
    },
    clip=(0.0, 1.0),
  )
  critic_terms["filtered_planar_velocity_b"] = ObservationTermCfg(
    func=mdp.filtered_planar_velocity_b,
    params={"command_name": "motion"},
  )
  critic_terms["filtered_joint_command"] = ObservationTermCfg(
    func=mdp.filtered_joint_command,
    params={"command_name": "motion"},
  )
  critic_terms["crowd_capsule_vectors_b"] = ObservationTermCfg(
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

  crowd_event = cfg.events[HUMAN_MOTION_EVENT_NAME]
  crowd_event.params["obstacle_free_probability"] = 0.25
  primary_event = cfg.events[PRIMARY_HUMAN_EVENT_NAME]
  primary_event.params["update_hz"] = 10.0
  primary_event.params["use_shared_obstacle_free_mask"] = True
  # Sample encounters jointly by (TTC, approach speed) with curriculum-ramped,
  # failure-adaptive bins. The radius/delay bounds double as the sampler's
  # spawn clamp and as the independent-mode fallback used by evaluation.
  primary_event.params["min_initial_spawn_radius_m"] = 0.75
  primary_event.params["min_intersection_delay_s"] = 0.5
  primary_event.params["max_intersection_delay_s"] = 4.0
  primary_event.params["encounter_sampling"] = "ttc"
  if not play:
    cfg.events[HUMAN_MOTION_EVENT_NAME] = EventTermCfg(
      func=PackedHumanCapsuleCrowdMotion,
      mode="step",
      params={
        **crowd_event.params,
        "packed_bank_path": DEFAULT_PACKED_CROWD_HUMAN_BANK_PATH,
      },
    )
    cfg.events[PRIMARY_HUMAN_EVENT_NAME] = EventTermCfg(
      func=PackedHumanCapsuleMotion,
      mode="step",
      params={
        **primary_event.params,
        "packed_bank_path": DEFAULT_PACKED_PRIMARY_HUMAN_BANK_PATH,
      },
    )

  cfg.rewards["human_proximity"].params["safe_clearance"] = 0.8
  cfg.rewards["primary_human_proximity"].params["safe_clearance"] = 0.8
  cfg.rewards["safe_planar_velocity"] = RewardTermCfg(
    func=mdp.safe_planar_velocity_tracking_exp,
    weight=1.5,
    params={"command_name": "motion", "std": 0.5},
  )
  cfg.rewards["filtered_joint_position"] = RewardTermCfg(
    func=mdp.filtered_joint_position_tracking_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.35},
  )
  cfg.rewards["safe_planar_freeze"] = RewardTermCfg(
    func=mdp.safe_planar_freeze_penalty,
    weight=-1.0,
    params={
      "command_name": "motion",
      "minimum_target_speed": 0.1,
      "minimum_progress_fraction": 0.25,
    },
  )
  cfg.rewards["safe_planar_progress"] = RewardTermCfg(
    func=mdp.safe_planar_progress_reward,
    weight=1.0,
    params={
      "command_name": "motion",
      "minimum_target_speed": 0.1,
      "normalization_speed": 0.5,
    },
  )
  cfg.rewards["survival"] = RewardTermCfg(
    func=mdp.survival_reward,
    weight=0.1,
  )
  cfg.terminations["crowd_collision"] = TerminationTermCfg(
    func=mdp.HumanCapsuleLinkCollision,
    params={
      "robot_entity": "robot",
      "human_entity": HUMAN_ENTITY_NAME,
      "robot_link_names": filtered_command_cfg.link_filter.body_names,
      "link_radius": filtered_command_cfg.link_filter.link_radius_m,
      "collision_margin": 0.1,
      "capsules_per_group": CROWD_CAPSULES_PER_PERSON,
      "nearest_groups": CROWD_PRIVILEGED_NEAREST_PEOPLE,
    },
  )
  cfg.terminations["primary_human_collision"] = TerminationTermCfg(
    func=mdp.HumanCapsuleLinkCollision,
    params={
      "robot_entity": "robot",
      "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
      "robot_link_names": filtered_command_cfg.link_filter.body_names,
      "link_radius": filtered_command_cfg.link_filter.link_radius_m,
      "collision_margin": 0.1,
    },
  )
  return cfg


def unitree_g1_sparse_lidar_avoidance_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the directional 120 x 4 LiDAR prototype task."""

  return unitree_g1_lidar_avoidance_tracking_env_cfg(
    play=play,
    lidar_azimuth_samples=PROTOTYPE_LIDAR_AZIMUTH_SAMPLES,
    lidar_elevation_angles_deg=PROTOTYPE_LIDAR_ELEVATIONS_DEG,
    actor_azimuth_bins=24,
    actor_elevation_bins=3,
  )


def unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Add explicit 10 Hz closing speed and dense link-clearance supervision."""

  cfg = unitree_g1_lidar_avoidance_tracking_env_cfg(play=play)
  lidar_terms = cfg.observations["lidar"].terms
  scan_pair = lidar_terms.pop("directional_scan_pair")
  lidar_terms = {
    "directional_range_rate": ObservationTermCfg(
      func=CachedDirectionalHeldLidarRangeRate,
      params={
        **scan_pair.params,
        "max_abs_range_rate_mps": 5.0,
      },
      clip=(-1.0, 1.0),
    ),
    **lidar_terms,
  }
  cfg.observations["lidar"].terms = lidar_terms

  motion = cfg.commands["motion"]
  assert isinstance(motion, PlanarFilteredReplayMotionCommandCfg)
  cfg.rewards["link_proximity"] = RewardTermCfg(
    func=mdp.reference_filter_clearance_penalty,
    weight=-3.0,
    params={
      "command_name": "motion",
      "safe_clearance_m": motion.link_filter.safe_clearance_m,
      "metric_name": "link_filter_minimum_clearance_m",
    },
  )
  return cfg


def unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(
  play: bool = False,
  *,
  expose_filtered_command: bool = False,
  propagate_arm_corrections_to_body_targets: bool = False,
  active_correction_reward: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Add training-only planar and limb-filter targets for the actor loss.

  ``expose_filtered_command`` swaps the actor's observed joint-command
  channel from the raw reference to the filter-adjusted joint targets
  (``PlanarFilteredReplayMotionCommandCfg.expose_filtered_command``). It
  defaults to ``False`` so every existing task built on this function is
  unaffected.

  ``propagate_arm_corrections_to_body_targets`` forwards the filtered arm
  joint corrections into the command's body position/orientation targets via
  reference-frame FK
  (``PlanarFilteredReplayMotionCommandCfg.propagate_arm_corrections_to_body_targets``).
  It defaults to ``False`` so every existing task built on this function is
  unaffected.

  ``active_correction_reward`` adds ``"active_correction_joint_tracking"``
  (``mdp.active_correction_joint_tracking_exp``), which rewards joint
  tracking only where the CBF teacher's own residual is active, instead of
  diluting the signal over all 29 joints like ``"filtered_joint_position"``
  does. It defaults to ``False`` so every existing task built on this
  function is unaffected.
  """

  cfg = unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg(play=play)
  if expose_filtered_command:
    cfg.commands["motion"].expose_filtered_command = True
  if propagate_arm_corrections_to_body_targets:
    cfg.commands["motion"].propagate_arm_corrections_to_body_targets = True
  if active_correction_reward:
    cfg.rewards["active_correction_joint_tracking"] = RewardTermCfg(
      func=mdp.active_correction_joint_tracking_exp,
      weight=1.5,
      params={
        "command_name": "motion",
        "std": 0.2,
        "activation_threshold_rad": 0.05,
      },
    )
  cfg.observations["avoidance_teacher"] = ObservationGroupCfg(
    terms={
      "corrections": ObservationTermCfg(
        func=mdp.avoidance_teacher_corrections,
        params={"command_name": "motion"},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["avoidance_robustness"] = ObservationGroupCfg(
    terms={
      "conditioning_noise": ObservationTermCfg(
        func=mdp.avoidance_conditioning_noise,
        params={"size": 31},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.rewards["urgent_escape_progress"] = RewardTermCfg(
    func=mdp.urgent_escape_progress_reward,
    weight=1.0,
    params={
      "command_name": "motion",
      "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
    },
  )
  return cfg


_NOMINAL_TRACKING_REWARD_NAMES = (
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
  "action_rate_l2",
  "joint_limit",
  "self_collisions",
)


def unitree_g1_lidar_unified_reference_tracking_env_cfg(
  play: bool = False,
  *,
  active_joint_reward: bool = False,
  root_lead_m: float | None = None,
  planar_filter_at_robot_root: bool = False,
  slow_regime: bool = False,
  dense_encounters: bool = False,
  filter_gated_ee_termination: bool = False,
  lag_aware_ee_termination: bool = False,
  motion_manifest: str | None = None,
  blind_actor: bool = False,
  nominal_reference: bool = False,
  training_humans: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Nominal tracking rewards on one fully filtered reference.

  Pipeline: LiDAR of randomized humans -> privileged planar + link CBF filters
  -> FK of every joint correction (anchor included) and closed-loop
  integration of the filtered root -> the stock mjlab tracking reward set on
  that reference. Every bespoke avoidance reward (planar velocity / progress /
  freeze, filtered-joint, urgent-escape, survival, proximity penalties) is
  removed; the two human collision terminations stay because the humans are
  ray-only geometry with no physics contact. The auxiliary teacher observation
  groups stay: they supervise the co-adjust head, they are not rewards.

  ``active_joint_reward`` adds ONE joint-space tracking term,
  ``"motion_active_joint_pos"`` (``mdp.active_correction_joint_tracking_exp``,
  weight 1.0, std 0.4 rad per active joint), evaluated against the filtered
  joint targets on the joints the filter is actively correcting. The nominal
  set has no joint-space term and its body-position term averages over 14
  bodies, so an arm correction otherwise carries almost no gradient (unified
  10k gate: arm state compliance 0.32 and falling). Std sized from data: the
  per-active-joint rms error at that gate was ~0.33 rad, so std 0.4 yields
  ~0.5 reward today and 1.0 at compliance (FKC2's std 0.2 was inert, ~0.06).

  ``root_lead_m`` / ``planar_filter_at_robot_root`` change only the reference
  generator (joint@14k root-tracking diagnosis, 2026-09-03): the closed-loop
  target is leashed to that planar lead from the robot so the nominal
  root-position term never saturates, and the planar CBF is evaluated at the
  robot instead of at the target so the escape velocity persists while the
  robot itself is still inside the safe clearance. Rewards, observations and
  terminations are untouched.

  ``slow_regime`` re-parameterises ONLY the walking human's encounter
  sampler (see the ``SLOW_REGIME_*`` constants): spawn outside the critical
  distance, TTC x speed bins capped at ``SLOW_REGIME_MAX_SPEED_MPS``, no hard
  bins, matching delay range. The crowd is untouched (its annulus already
  starts at 2 m). ``filter_gated_ee_termination`` swaps ``ee_body_pos`` for
  :func:`mdp.bad_motion_body_pos_z_only_filter_gated` with the stock bodies
  and 0.25 m bound, loosened to ``EE_LOOSENED_THRESHOLD_M`` on a limb while
  the link filter is correcting it. ``dense_encounters`` (requires
  ``slow_regime``) swaps in the ``SLOW_REGIME_DENSE_*`` TTC edges and delay
  range so each episode holds two to three encounters. ``motion_manifest``
  replaces the single dance clip with a clip-library manifest (every split);
  the replay command chains to a fresh clip whenever one ends.
  ``blind_actor`` feeds the actor a constant "no returns" LiDAR term.
  ``nominal_reference`` turns BOTH CBF filters off (the reference is the raw
  live-aligned motion, teacher correction zero) so a blind baseline trained
  with it has no avoidance signal of any kind. ``training_humans=False``
  (training cfg only, ignored when ``play``) additionally removes the two
  human animation events and the two collision terminations, so the humans
  stay parked below the floor and never touch training; the play cfg keeps
  them so evaluation happens in the populated scene. Scene entities and the
  critic's privileged terms are untouched so network shapes match.
  """
  if dense_encounters and not slow_regime:
    raise ValueError("dense_encounters requires slow_regime=True")
  if filter_gated_ee_termination and lag_aware_ee_termination:
    raise ValueError("choose one ee_body_pos variant, not both")
  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=play)
  motion = cfg.commands["motion"]
  assert isinstance(motion, PlanarFilteredReplayMotionCommandCfg)
  motion.propagate_joint_corrections_to_body_targets = True
  motion.closed_loop_root_target = True
  motion.max_root_lead_m = root_lead_m
  motion.planar_filter_at_robot_root = planar_filter_at_robot_root
  motion.disable_filters = nominal_reference
  if not training_humans and not play:
    for name in (HUMAN_MOTION_EVENT_NAME, PRIMARY_HUMAN_EVENT_NAME):
      cfg.events.pop(name)
    for name in ("crowd_collision", "primary_human_collision"):
      cfg.terminations.pop(name)
  if motion_manifest is not None:
    motion.motion_file = str(motion_manifest)
    motion.manifest_splits = None
  if blind_actor:
    # No-perception baseline: the actor's LiDAR term reads "no returns" at
    # every step (same shape/params, so the network is identical); the critic
    # keeps its privileged LiDAR and human vectors.
    cfg.observations["lidar"].terms["directional_range_rate"].func = (
      BlindDirectionalHeldLidarRangeRate
    )
  if lag_aware_ee_termination:
    stock = cfg.terminations["ee_body_pos"]
    cfg.terminations["ee_body_pos"] = TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only_lag_aware,
      params={
        "command_name": "motion",
        "threshold": stock.params["threshold"],
        "lag_time_s": EE_LAG_TIME_S,
        "max_threshold": EE_LAG_MAX_THRESHOLD_M,
        "body_names": tuple(stock.params["body_names"]),
      },
    )
  if slow_regime:
    primary = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
    primary["min_initial_spawn_radius_m"] = SLOW_REGIME_MIN_SPAWN_RADIUS_M
    primary["encounter_sampling"] = "ttc"
    primary["speed_bin_edges_mps"] = SLOW_REGIME_SPEED_EDGES_MPS
    primary["ttc_bin_edges_s"] = SLOW_REGIME_TTC_EDGES_S
    # Every bin is "easy": disable the curriculum's hard-bin down-weighting.
    primary["hard_speed_above_mps"] = 10.0 * SLOW_REGIME_MAX_SPEED_MPS
    primary["hard_ttc_below_s"] = 0.1 * SLOW_REGIME_TTC_EDGES_S[0]
    primary["min_intersection_delay_s"] = SLOW_REGIME_DELAY_RANGE_S[0]
    primary["max_intersection_delay_s"] = SLOW_REGIME_DELAY_RANGE_S[1]
    if dense_encounters:
      primary["ttc_bin_edges_s"] = SLOW_REGIME_DENSE_TTC_EDGES_S
      primary["min_intersection_delay_s"] = SLOW_REGIME_DENSE_DELAY_RANGE_S[0]
      primary["max_intersection_delay_s"] = SLOW_REGIME_DENSE_DELAY_RANGE_S[1]
  if filter_gated_ee_termination:
    stock = cfg.terminations["ee_body_pos"]
    cfg.terminations["ee_body_pos"] = TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only_filter_gated,
      params={
        "command_name": "motion",
        "threshold": stock.params["threshold"],
        "loosened_threshold": EE_LOOSENED_THRESHOLD_M,
        "activation_threshold_rad": 0.05,
        "body_names": tuple(stock.params["body_names"]),
      },
    )
  # At deployment only the raw reference and the learned adjuster exist, so
  # the actor must observe the raw anchor orientation, not the privileged
  # whole-body FK correction carried by anchor_quat_w. Swap only the actor
  # group's term func; keep its name/params/noise/position (the co-adjust
  # command offset and 154-dim actor layout depend on the term order). The
  # critic keeps mjlab's filtered-reference term.
  cfg.observations["actor"].terms["motion_anchor_ori_b"].func = (
    mdp.raw_motion_anchor_ori_b
  )
  for name in tuple(cfg.rewards):
    if name not in _NOMINAL_TRACKING_REWARD_NAMES:
      cfg.rewards.pop(name)
  missing = set(_NOMINAL_TRACKING_REWARD_NAMES) - set(cfg.rewards)
  if missing:
    raise ValueError(f"nominal tracking rewards missing: {sorted(missing)}")
  if active_joint_reward:
    cfg.rewards["motion_active_joint_pos"] = RewardTermCfg(
      func=mdp.active_correction_joint_tracking_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.4,
        "activation_threshold_rad": 0.05,
      },
    )
  return cfg


def unitree_g1_nominal_lidar_debug_env_cfg(
  play: bool = False,
  has_state_estimation: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Build the upstream nominal tracker with a visualization-only LiDAR.

  Neither actor nor critic observations are changed, so upstream tracking
  checkpoints remain strictly input-compatible. The animated human only gives
  the debug rays useful geometry to hit; the nominal policy cannot react to it.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
  )
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
  """Track a live-aligned privileged reference with the no-state policy.

  This keeps the no-state actor's 154-value observation contract and all
  physics, contacts, actuation, and tracking terminations. Before filtering,
  each source frame is translated and yaw-aligned to the live robot. Unlike
  the exact replay demo, the filter changes only the desired motion command
  and never writes the filtered pose into MuJoCo during a policy step.
  """
  if scan_hz not in (5, 10):
    raise ValueError(f"scan_hz must be 5 or 10, got {scan_hz}")

  cfg = unitree_g1_nominal_lidar_debug_env_cfg(
    play=play,
    has_state_estimation=False,
  )
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
    align_reference_to_robot_each_step=True,
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
