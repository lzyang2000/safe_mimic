"""Unitree G1 asset extensions that remain local to Safe Mimic."""

from __future__ import annotations

import math

import mujoco

from safe_mimic.assets.soma_capsules import HUMAN_COLLISION_TYPE

# Fixed transform of mid360_link relative to torso_link in the deployment URDF.
MID360_SITE_NAME = "mid360_site"
LIDAR_IGNORE_GEOM_GROUP = 5
MID360_POSITION_M = (0.0002835, 0.00003, 0.41618)
MID360_PITCH_RAD = 0.04014257279586953
MID360_QUAT_WXYZ = (
  math.cos(MID360_PITCH_RAD / 2.0),
  0.0,
  math.sin(MID360_PITCH_RAD / 2.0),
  0.0,
)
SHOULDER_PLANK_POSITION_M = (
  MID360_POSITION_M[0],
  MID360_POSITION_M[1],
  0.27778,
)
SHOULDER_PLANK_HALF_SIZE_M = (0.045, 0.15, 0.005)


def get_g1_with_mid360_spec() -> mujoco.MjSpec:
  """Return mjlab's G1 spec with the URDF-defined Mid-360 frame added.

  The published mjlab G1 MJCF omits the fixed sensor link from the deployment
  URDF. Adding a massless site retains the upstream dynamics and lets sensors
  use MuJoCo's frame transform directly.
  """
  # Keep mjlab's task plug-in import out of module initialization. Importing it
  # here avoids a cycle when mjlab discovers Safe Mimic's entry point.
  from mjlab.asset_zoo.robots import get_g1_robot_cfg

  spec = get_g1_robot_cfg().spec_fn()
  torso = spec.body("torso_link")
  if torso is None:
    raise ValueError("mjlab G1 spec does not contain torso_link")

  torso.add_site(
    name=MID360_SITE_NAME,
    pos=MID360_POSITION_M,
    quat=MID360_QUAT_WXYZ,
    size=(0.01, 0.01, 0.01),
    rgba=(0.0, 0.0, 1.0, 1.0),
    group=5,
  )

  # Head-side pillars measured relative to the Mid-360 optical origin: 4 cm
  # forward, 3.5 cm to either side, and 3.75 cm down.  Their full XYZ size is
  # 2.2 x 1.3 x 7.5 cm (MuJoCo stores box half-sizes).  Group 2 copies make the
  # proxies visible in the default viewer; coincident group 3 copies are
  # LiDAR-only and have no contact or inertial effect.
  head_side_pos_x = MID360_POSITION_M[0] + 0.04
  head_side_pos_z = MID360_POSITION_M[2] - 0.0375
  head_side_center_y = 0.035
  head_side_half_size = (0.011, 0.0065, 0.0375)
  for side_name, side_sign in (("left", 1.0), ("right", -1.0)):
    # Mirrored X-axis roll makes the lower ends lean toward y=0.  A shared
    # negative Y-axis pitch makes both lower ends lean forward in +X.  Compose
    # q = q_pitch * q_roll so both directions refer to the torso frame.
    roll_rad = -side_sign * math.radians(15.0)
    pitch_rad = math.radians(-5.0)
    cos_roll = math.cos(roll_rad / 2.0)
    sin_roll = math.sin(roll_rad / 2.0)
    cos_pitch = math.cos(pitch_rad / 2.0)
    sin_pitch = math.sin(pitch_rad / 2.0)
    quat_wxyz = (
      cos_pitch * cos_roll,
      cos_pitch * sin_roll,
      sin_pitch * cos_roll,
      -sin_pitch * sin_roll,
    )
    pos = (
      head_side_pos_x,
      MID360_POSITION_M[1] + side_sign * head_side_center_y,
      head_side_pos_z,
    )
    torso.add_geom(
      name=f"lidar_head_side_{side_name}_visual",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      quat=quat_wxyz,
      size=head_side_half_size,
      group=2,
      contype=0,
      conaffinity=0,
      density=0.0,
      rgba=(1.0, 0.35, 0.05, 0.8),
    )
    torso.add_geom(
      name=f"lidar_head_side_{side_name}_occluder",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      quat=quat_wxyz,
      size=head_side_half_size,
      group=3,
      contype=0,
      conaffinity=0,
      density=0.0,
      rgba=(1.0, 0.35, 0.05, 0.25),
    )

  # A temporary flat calibration plank is rigidly attached to torso_link at
  # shoulder height, directly below the Mid-360 origin.  Its full XYZ size is
  # 9 x 30 x 1 cm.  As with the head pillars, it has no contacts or inertia.
  for suffix, group, alpha in (("visual", 2, 1.0), ("occluder", 3, 0.25)):
    torso.add_geom(
      name=f"lidar_shoulder_plank_{suffix}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=SHOULDER_PLANK_POSITION_M,
      size=SHOULDER_PLANK_HALF_SIZE_M,
      group=group,
      contype=0,
      conaffinity=0,
      density=0.0,
      rgba=(0.72, 0.38, 0.12, alpha),
    )

  # The upstream G1 uses a coarse 6 cm ``head_collision`` sphere centered
  # almost exactly on the Mid-360 optical origin.  It is useful for contact
  # physics, but it is not a valid LiDAR occluder: ray casting against it would
  # make every beam hit immediately from inside the sphere.  Move only its
  # visualization/raycast group out of the sensor mask.  MuJoCo collision
  # behavior is controlled by contype/conaffinity and is therefore unchanged.
  head_collision = spec.geom("head_collision")
  if head_collision is None:
    raise ValueError("mjlab G1 spec does not contain head_collision")
  head_collision.group = LIDAR_IGNORE_GEOM_GROUP

  # The animated human uses a dedicated collision type and zero affinity so
  # its overlapping mocap capsules neither self-collide nor hit the terrain.
  # Opt only physical G1 collision geoms into contacts with that type.
  for geom in spec.geoms:
    if geom.contype:
      geom.conaffinity |= HUMAN_COLLISION_TYPE
  return spec
