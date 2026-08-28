"""Local extensions to published mjlab assets."""

from safe_mimic.assets.g1 import (
  MID360_PITCH_RAD,
  MID360_POSITION_M,
  MID360_SITE_NAME,
  get_g1_with_mid360_spec,
)
from safe_mimic.assets.soma_capsules import (
  HUMAN_CAPSULE_BODY_PREFIX,
  HUMAN_CAPSULE_GEOM_PREFIX,
  HUMAN_COLLISION_TYPE,
  HUMAN_CROWD_BODY_PREFIX,
  HUMAN_CROWD_CAPACITY,
  HUMAN_CROWD_GEOM_PREFIX,
  HUMAN_INACTIVE_HEIGHT_M,
  HUMAN_RAYCAST_GROUP,
  crowd_capsule_body_name,
  crowd_capsule_geom_name,
  get_soma_capsule_crowd_spec,
  get_soma_capsule_human_spec,
  human_capsule_body_name,
  human_capsule_geom_name,
)
from safe_mimic.assets.soma_human import (
  get_soma_debug_mesh_cfg,
  get_soma_debug_mesh_spec,
)

__all__ = [
  "MID360_PITCH_RAD",
  "MID360_POSITION_M",
  "MID360_SITE_NAME",
  "HUMAN_CAPSULE_BODY_PREFIX",
  "HUMAN_CAPSULE_GEOM_PREFIX",
  "HUMAN_COLLISION_TYPE",
  "HUMAN_CROWD_BODY_PREFIX",
  "HUMAN_CROWD_CAPACITY",
  "HUMAN_CROWD_GEOM_PREFIX",
  "HUMAN_INACTIVE_HEIGHT_M",
  "HUMAN_RAYCAST_GROUP",
  "crowd_capsule_body_name",
  "crowd_capsule_geom_name",
  "get_g1_with_mid360_spec",
  "get_soma_capsule_crowd_spec",
  "get_soma_capsule_human_spec",
  "get_soma_debug_mesh_cfg",
  "get_soma_debug_mesh_spec",
  "human_capsule_body_name",
  "human_capsule_geom_name",
]
