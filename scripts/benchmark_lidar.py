"""Benchmark Safe Mimic environment stepping with and without LiDAR."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace

import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from safe_mimic.assets import (
  HUMAN_CAPSULE_BODY_PREFIX,
  get_soma_capsule_human_spec,
)
from safe_mimic.sensing.held_scan import HeldScanRayCastSensorCfg
from safe_mimic.tasks import mdp
from safe_mimic.tasks.env_cfg import (
  DEFAULT_SKELETON_BANK_PATH,
  DEFAULT_SOMA_MESH_SKIN_PATH,
  DEFAULT_TRANSITION_INDEX_PATH,
  HUMAN_CONTACT_SENSOR_NAME,
  HUMAN_ENTITY_NAME,
  HUMAN_MOTION_EVENT_NAME,
  LIDAR_SENSOR_NAME,
  LIVOX_SNAPSHOT_AZIMUTH_SAMPLES,
  LIVOX_SNAPSHOT_ELEVATIONS_DEG,
  unitree_g1_crowd_and_human_tracking_env_cfg,
  unitree_g1_lidar_avoidance_tracking_env_cfg,
  unitree_g1_nominal_lidar_debug_env_cfg,
  unitree_g1_obstacle_aware_tracking_env_cfg,
)
from safe_mimic.tasks.human_capsule_event import HumanCapsuleMotion

FULL_RESOLUTION_AZIMUTH_SAMPLES = 360
FULL_RESOLUTION_ELEVATION_SAMPLES = 52
FULL_RESOLUTION_ELEVATIONS_DEG = tuple(
  -52.0 * index / (FULL_RESOLUTION_ELEVATION_SAMPLES - 1)
  for index in range(FULL_RESOLUTION_ELEVATION_SAMPLES)
)
PRIMARY_HUMAN_ENTITY_NAME = "primary_human"
PRIMARY_HUMAN_EVENT_NAME = "animate_primary_human"
PRIMARY_HUMAN_CONTACT_SENSOR_NAME = "primary_human_contact"


def _full_human_event_cfg(entity_name: str) -> EventTermCfg:
  return EventTermCfg(
    func=HumanCapsuleMotion,
    mode="step",
    params={
      "human_entity": entity_name,
      "command_name": "motion",
      "skeleton_bank_path": DEFAULT_SKELETON_BANK_PATH,
      "transition_index_path": DEFAULT_TRANSITION_INDEX_PATH,
      "update_hz": 10.0,
      "transition_duration_s": 0.2,
      "min_intersection_delay_s": 1.0,
      "max_intersection_delay_s": 3.0,
      "show_mesh": False,
      "mesh_skin_path": DEFAULT_SOMA_MESH_SKIN_PATH,
    },
  )


def _full_human_contact_cfg(entity_name: str, sensor_name: str) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=sensor_name,
    primary=ContactMatch(
      mode="body",
      pattern=rf"{HUMAN_CAPSULE_BODY_PREFIX}.*",
      entity=entity_name,
    ),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )


def _single_human_obstacle_cfg():
  """Replace the dense crowd with one full 18-capsule crossing human."""

  cfg = unitree_g1_obstacle_aware_tracking_env_cfg(play=True)
  cfg.scene.entities[HUMAN_ENTITY_NAME] = EntityCfg(spec_fn=get_soma_capsule_human_spec)
  cfg.events[HUMAN_MOTION_EVENT_NAME] = _full_human_event_cfg(HUMAN_ENTITY_NAME)
  critic_params = cfg.observations["critic"].terms["human_capsule_vectors_b"].params
  reward_params = cfg.rewards["human_proximity"].params
  for params in (critic_params, reward_params):
    params.pop("capsules_per_group", None)
    params.pop("nearest_groups", None)
  contact = _full_human_contact_cfg(HUMAN_ENTITY_NAME, HUMAN_CONTACT_SENSOR_NAME)
  cfg.scene.sensors = tuple(
    contact if sensor.name == HUMAN_CONTACT_SENSOR_NAME else sensor
    for sensor in (cfg.scene.sensors or ())
  )
  return cfg


def _combined_human_obstacle_cfg():
  """Dense five-proxy crowd plus one full 18-capsule crossing human."""

  cfg = unitree_g1_obstacle_aware_tracking_env_cfg(play=True)
  cfg.scene.entities[PRIMARY_HUMAN_ENTITY_NAME] = EntityCfg(
    spec_fn=get_soma_capsule_human_spec
  )
  cfg.events[PRIMARY_HUMAN_EVENT_NAME] = _full_human_event_cfg(
    PRIMARY_HUMAN_ENTITY_NAME
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    _full_human_contact_cfg(
      PRIMARY_HUMAN_ENTITY_NAME, PRIMARY_HUMAN_CONTACT_SENSOR_NAME
    ),
  )
  cfg.observations["critic"].terms["primary_human_capsule_vectors_b"] = (
    ObservationTermCfg(
      func=mdp.human_capsule_vectors_b,
      params={
        "robot_entity": "robot",
        "human_entity": PRIMARY_HUMAN_ENTITY_NAME,
        "max_distance": 5.0,
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
      "safe_clearance": 0.65,
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


def _ray_only_combined_human_obstacle_cfg():
  """Mixed case with unchanged crowd rays but analytical-only crowd contact."""

  return unitree_g1_crowd_and_human_tracking_env_cfg(play=True)


def _lidar_only_obstacle_cfg():
  """Keep the obstacle-task LiDAR contract while removing all humans."""

  cfg = unitree_g1_obstacle_aware_tracking_env_cfg(play=True)
  del cfg.scene.entities[HUMAN_ENTITY_NAME]
  del cfg.events[HUMAN_MOTION_EVENT_NAME]
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != HUMAN_CONTACT_SENSOR_NAME
  )
  del cfg.observations["critic"].terms["human_capsule_vectors_b"]
  del cfg.rewards["human_proximity"]
  del cfg.rewards["human_collision"]
  del cfg.terminations["human_collision"]
  return cfg


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--case", choices=("no-lidar", "lidar"), required=True)
  parser.add_argument(
    "--task",
    choices=(
      "baseline",
      "lidar",
      "human",
      "combined",
      "combined-ray-only",
      "debug",
      "obstacle",
      "avoidance",
    ),
    default="debug",
  )
  parser.add_argument("--scan-hz", choices=(5, 10), default=10, type=int)
  parser.add_argument("--crowd-update-hz", choices=(5, 10), default=10, type=int)
  parser.add_argument(
    "--disable-crowd-critic",
    action="store_true",
    help="profile without the crowd privileged-observation term",
  )
  parser.add_argument(
    "--disable-crowd-proximity",
    action="store_true",
    help="profile without the crowd analytical-proximity reward",
  )
  parser.add_argument(
    "--resolution", choices=("sparse", "quarter", "full"), default="sparse"
  )
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--warmup-steps", type=int, default=10)
  parser.add_argument("--steps-per-trial", type=int, default=30)
  parser.add_argument("--trials", type=int, default=3)
  parser.add_argument("--verify-hold-steps", type=int, default=0)
  parser.add_argument("--verify-crowd-geometry", action="store_true")
  args = parser.parse_args()

  configure_torch_backends()
  if args.task == "baseline":
    if args.case != "no-lidar":
      raise ValueError("the baseline task intentionally has no LiDAR")
    cfg = unitree_g1_flat_tracking_env_cfg(play=True)
  elif args.task == "lidar":
    cfg = _lidar_only_obstacle_cfg()
  elif args.task == "human":
    cfg = _single_human_obstacle_cfg()
  elif args.task == "combined":
    cfg = _combined_human_obstacle_cfg()
  elif args.task == "combined-ray-only":
    cfg = _ray_only_combined_human_obstacle_cfg()
  elif args.task == "debug":
    cfg = unitree_g1_nominal_lidar_debug_env_cfg(play=True)
  elif args.task == "avoidance":
    if args.case != "lidar":
      raise ValueError("the avoidance task requires LiDAR")
    cfg = unitree_g1_lidar_avoidance_tracking_env_cfg(play=False)
  else:
    cfg = unitree_g1_obstacle_aware_tracking_env_cfg(play=True)
  if args.disable_crowd_critic:
    cfg.observations["critic"].terms.pop("human_capsule_vectors_b", None)
  if args.disable_crowd_proximity:
    del cfg.rewards["human_proximity"]
  crowd_event = cfg.events.get(HUMAN_MOTION_EVENT_NAME)
  if crowd_event is not None and "capacity" in crowd_event.params:
    # Headless stepping never consumes the Viser-only retained joint poses.
    crowd_event.params["show_mesh"] = False
    crowd_event.params["update_hz"] = float(args.crowd_update_hz)
    crowd_event.params["mesh_update_hz"] = min(
      float(crowd_event.params["mesh_update_hz"]),
      float(args.crowd_update_hz),
    )
  primary_event = cfg.events.get(PRIMARY_HUMAN_EVENT_NAME)
  if primary_event is not None:
    primary_event.params["show_mesh"] = False
  cfg.scene.num_envs = args.num_envs
  if args.task != "avoidance":
    cfg.terminations = {}

  motion = cfg.commands["motion"]
  if not isinstance(motion, MotionCommandCfg):
    raise TypeError("expected the nominal debug task to use MotionCommandCfg")
  motion.motion_file = args.motion_file

  sensors = cfg.scene.sensors or ()
  if args.case == "no-lidar":
    if args.task == "obstacle":
      raise ValueError("the obstacle task's observations require LiDAR")
    cfg.scene.sensors = tuple(
      sensor for sensor in sensors if sensor.name != LIDAR_SENSOR_NAME
    )
  else:
    for sensor in sensors:
      if sensor.name == LIDAR_SENSOR_NAME:
        if not isinstance(sensor, HeldScanRayCastSensorCfg):
          raise TypeError("expected a HeldScanRayCastSensorCfg")
        sensor.scan_period = 1.0 / args.scan_hz
        pattern_updates: dict[str, object] = {"phases": 50 // args.scan_hz}
        if args.resolution == "quarter":
          pattern_updates.update(
            azimuth_samples=LIVOX_SNAPSHOT_AZIMUTH_SAMPLES,
            elevation_angles_deg=LIVOX_SNAPSHOT_ELEVATIONS_DEG,
          )
        elif args.resolution == "full":
          pattern_updates.update(
            azimuth_samples=FULL_RESOLUTION_AZIMUTH_SAMPLES,
            elevation_angles_deg=FULL_RESOLUTION_ELEVATIONS_DEG,
          )
        sensor.pattern = replace(sensor.pattern, **pattern_updates)
        sensor.debug_vis = False

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  action = torch.zeros(env.action_space.shape, device=env.device)

  if args.verify_crowd_geometry:
    crowd_motion = env.event_manager.get_term_cfg(HUMAN_MOTION_EVENT_NAME).func
    expected_centers = crowd_motion.sampler._poses.centers_w.reshape(
      args.num_envs, -1, 3
    )
    expected_quaternions = crowd_motion.sampler._poses.quaternions_wxyz.reshape(
      args.num_envs, -1, 4
    )
    crowd = env.scene[HUMAN_ENTITY_NAME]
    center_error = (crowd.data.geom_pos_w - expected_centers).abs().max().item()
    quaternion_dot = (crowd.data.geom_quat_w * expected_quaternions).sum(dim=-1).abs()
    quaternion_error = (1.0 - quaternion_dot).abs().max().item()
    geom_ids = crowd.indexing.geom_ids.to(dtype=torch.long)
    expected_sizes = torch.stack(
      (
        crowd_motion.sampler._poses.radii_m,
        crowd_motion.sampler._poses.half_lengths_m,
        torch.zeros_like(crowd_motion.sampler._poses.radii_m),
      ),
      dim=-1,
    ).reshape(args.num_envs, -1, 3)
    size_error = (
      (env.sim.model.geom_size[:, geom_ids] - expected_sizes).abs().max().item()
    )
    print(
      "CROWD_GEOMETRY_CHECK "
      f"direct_geom_poses={crowd_motion._direct_geom_poses} "
      f"max_center_error_m={center_error:.9g} "
      f"max_quaternion_error={quaternion_error:.9g} "
      f"max_size_error_m={size_error:.9g}"
    )
    env.close()
    return

  if args.verify_hold_steps:
    if args.case != "lidar":
      raise ValueError("--verify-hold-steps requires --case lidar")
    sensor = env.scene[LIDAR_SENSOR_NAME]
    previous = sensor.data.distances.clone()
    changed_steps: list[int] = []
    for step in range(1, args.verify_hold_steps + 1):
      env.step(action)
      current = sensor.data.distances
      if torch.any(current != previous).item():
        changed_steps.append(step)
      previous.copy_(current)
    print(f"HOLD_CHECK steps={args.verify_hold_steps} changed_steps={changed_steps}")
    env.close()
    return

  for _ in range(args.warmup_steps):
    env.step(action)
  torch.cuda.synchronize()

  durations: list[float] = []
  for _ in range(args.trials):
    start = time.perf_counter()
    for _ in range(args.steps_per_trial):
      env.step(action)
    torch.cuda.synchronize()
    durations.append(time.perf_counter() - start)

  seconds_per_step = statistics.median(durations) / args.steps_per_trial
  transitions_per_second = args.num_envs / seconds_per_step
  print(
    f"RESULT task={args.task} case={args.case} scan_hz={args.scan_hz} "
    f"crowd_update_hz={args.crowd_update_hz} "
    f"envs={args.num_envs} "
    f"resolution={args.resolution} "
    f"ms_per_step={seconds_per_step * 1e3:.3f} "
    f"transitions_per_second={transitions_per_second:.0f} "
    f"trial_seconds={','.join(f'{value:.6f}' for value in durations)}"
  )
  env.close()


if __name__ == "__main__":
  main()
