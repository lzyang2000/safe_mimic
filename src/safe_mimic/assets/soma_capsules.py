"""Runtime SOMA human proxy built from independently driven mocap bodies."""

from __future__ import annotations

import mujoco

from safe_mimic.motions.human_capsules import (
  SOMA_CAPSULE_SPECS,
  SOMA_CROWD_PROXY_SPECS,
  CapsuleSpec,
)

HUMAN_CAPSULE_BODY_PREFIX = "human_capsule_"
HUMAN_CAPSULE_GEOM_PREFIX = "human_capsule_geom_"
HUMAN_CROWD_BODY_PREFIX = "human_crowd_"
HUMAN_CROWD_GEOM_PREFIX = "human_crowd_geom_"
HUMAN_CROWD_RAY_BODY_NAME = "human_crowd_ray_only_root"
# The densest configured ring can contain at most
# floor(2*pi*(6.0 m - 0.25 m)/0.62 m) == 58 people. Compiling more slots adds
# mocap bodies and ray-test geoms that can never become active.
HUMAN_CROWD_CAPACITY = 58
HUMAN_COLLISION_TYPE = 1 << 1
# Group 3 is included by the LiDAR sensor but hidden by default in both mjlab
# viewers.  This keeps the collision/raycast proxies out of the mesh-only
# visualization without relying on alpha, which MuJoCo Warp uses to decide
# whether a geom can be hit by a ray.
HUMAN_RAYCAST_GROUP = 3
HUMAN_INACTIVE_HEIGHT_M = -100.0
HUMAN_PROXY_RENDER_ALPHA = 1.0

# These sizes only provide a valid model before the first reset event. Runtime
# sizes come from the selected path and its per-environment shape randomization.
_DEFAULT_HALF_LENGTH_M = {
  "pelvis": 0.09,
  "torso_lower": 0.13,
  "torso_upper": 0.12,
  "shoulders": 0.18,
  "neck": 0.10,
  "head": 0.0,
  "left_upper_arm": 0.14,
  "left_forearm": 0.13,
  "left_hand": 0.0,
  "right_upper_arm": 0.14,
  "right_forearm": 0.13,
  "right_hand": 0.0,
  "left_thigh": 0.20,
  "left_shin": 0.20,
  "left_foot": 0.10,
  "right_thigh": 0.20,
  "right_shin": 0.20,
  "right_foot": 0.10,
  "body_head": 0.40,
  "left_arm": 0.28,
  "right_arm": 0.28,
  "left_leg": 0.40,
  "right_leg": 0.40,
}


def human_capsule_body_name(capsule_name: str) -> str:
  return f"{HUMAN_CAPSULE_BODY_PREFIX}{capsule_name}"


def human_capsule_geom_name(capsule_name: str) -> str:
  return f"{HUMAN_CAPSULE_GEOM_PREFIX}{capsule_name}"


def crowd_capsule_body_name(crowd_index: int, capsule_name: str) -> str:
  if not 0 <= crowd_index < HUMAN_CROWD_CAPACITY:
    raise ValueError("crowd index lies outside the configured capacity")
  return f"{HUMAN_CROWD_BODY_PREFIX}{crowd_index:02d}_capsule_{capsule_name}"


def crowd_capsule_geom_name(crowd_index: int, capsule_name: str) -> str:
  if not 0 <= crowd_index < HUMAN_CROWD_CAPACITY:
    raise ValueError("crowd index lies outside the configured capacity")
  return f"{HUMAN_CROWD_GEOM_PREFIX}{crowd_index:02d}_{capsule_name}"


def _add_capsule_member(
  spec: mujoco.MjSpec,
  *,
  body_name: str,
  geom_name: str,
  capsule: CapsuleSpec,
  color_index: int,
  collidable: bool = True,
) -> None:
  colors = (
    (0.80, 0.32, 0.22),
    (0.92, 0.58, 0.20),
    (0.24, 0.55, 0.82),
  )
  half_length = _DEFAULT_HALF_LENGTH_M[capsule.name]
  geom_type = (
    mujoco.mjtGeom.mjGEOM_SPHERE
    if capsule.end_joint is None
    else mujoco.mjtGeom.mjGEOM_CAPSULE
  )
  body = spec.worldbody.add_body(
    name=body_name,
    mocap=True,
    pos=(0.0, 0.0, HUMAN_INACTIVE_HEIGHT_M),
  )
  body.add_geom(
    name=geom_name,
    type=geom_type,
    size=(capsule.radius_m, half_length, 0.0),
    group=HUMAN_RAYCAST_GROUP,
    contype=HUMAN_COLLISION_TYPE if collidable else 0,
    conaffinity=0,
    friction=(0.8, 0.005, 0.0001),
    rgba=(*colors[color_index % len(colors)], HUMAN_PROXY_RENDER_ALPHA),
  )


def get_soma_capsule_human_spec() -> mujoco.MjSpec:
  """Create 18 root-level mocap bodies driven by the online path sampler.

  The human uses collision type bit 1 and no affinity. G1 collision geoms opt
  into that bit, so the human contacts the robot without contacting itself or
  the ground. Geom group 3 keeps every capsule visible to the LiDAR ray caster
  while the viewers hide the proxy geometry by default.
  """

  spec = mujoco.MjSpec()
  for index, capsule in enumerate(SOMA_CAPSULE_SPECS):
    _add_capsule_member(
      spec,
      body_name=human_capsule_body_name(capsule.name),
      geom_name=human_capsule_geom_name(capsule.name),
      capsule=capsule,
      color_index=index,
    )
  return spec


def get_soma_capsule_crowd_spec(*, collidable: bool = True) -> mujoco.MjSpec:
  """Create a fixed-capacity crowd of independently driven capsule humans.

  Ray-only crowds keep ``collidable=False`` geoms visible to LiDAR group 3 while
  avoiding physical contact generation. This is useful when analytical
  proximity/collision terms provide the training signal.
  """

  spec = mujoco.MjSpec()
  ray_only_root = None
  if not collidable:
    # Ray-only crowd geoms do not need independent kinematic bodies. Their
    # per-environment geom transforms are written directly by the motion event.
    ray_only_root = spec.worldbody.add_body(
      name=HUMAN_CROWD_RAY_BODY_NAME, mocap=True
    )
  for crowd_index in range(HUMAN_CROWD_CAPACITY):
    for capsule_index, capsule in enumerate(SOMA_CROWD_PROXY_SPECS):
      if ray_only_root is None:
        _add_capsule_member(
          spec,
          body_name=crowd_capsule_body_name(crowd_index, capsule.name),
          geom_name=crowd_capsule_geom_name(crowd_index, capsule.name),
          capsule=capsule,
          color_index=capsule_index,
          collidable=True,
        )
      else:
        half_length = _DEFAULT_HALF_LENGTH_M[capsule.name]
        geom_type = (
          mujoco.mjtGeom.mjGEOM_SPHERE
          if capsule.end_joint is None
          else mujoco.mjtGeom.mjGEOM_CAPSULE
        )
        ray_only_root.add_geom(
          name=crowd_capsule_geom_name(crowd_index, capsule.name),
          type=geom_type,
          pos=(0.0, 0.0, HUMAN_INACTIVE_HEIGHT_M),
          size=(capsule.radius_m, half_length, 0.0),
          group=HUMAN_RAYCAST_GROUP,
          contype=0,
          conaffinity=0,
          rgba=(0.8, 0.32, 0.22, HUMAN_PROXY_RENDER_ALPHA),
        )
  return spec
