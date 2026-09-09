import mujoco
import numpy as np

from safe_mimic.assets.g1 import get_g1_with_mid360_spec
from safe_mimic.assets.soma_capsules import (
  HUMAN_COLLISION_TYPE,
  HUMAN_CROWD_CAPACITY,
  HUMAN_INACTIVE_HEIGHT_M,
  HUMAN_PROXY_RENDER_ALPHA,
  HUMAN_RAYCAST_GROUP,
  crowd_capsule_body_name,
  crowd_capsule_geom_name,
  get_soma_capsule_crowd_spec,
  get_soma_capsule_human_spec,
)
from safe_mimic.motions.human_capsules import (
  SOMA_CAPSULE_SPECS,
  SOMA_CROWD_PROXY_SPECS,
)
from safe_mimic.tasks.env_cfg import (
  HUMAN_CONTACT_SENSOR_NAME,
  HUMAN_ENTITY_NAME,
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_CONTACT_SENSOR_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  unitree_g1_crowd_and_human_tracking_env_cfg,
  unitree_g1_nominal_lidar_debug_env_cfg,
  unitree_g1_obstacle_aware_tracking_env_cfg,
)


def test_soma_capsule_crowd_has_independent_mocap_geometries() -> None:
  model = get_soma_capsule_crowd_spec().compile()
  expected = HUMAN_CROWD_CAPACITY * len(SOMA_CROWD_PROXY_SPECS)

  assert model.nmocap == expected
  assert model.ngeom == expected
  for crowd_index in (0, HUMAN_CROWD_CAPACITY - 1):
    for capsule in SOMA_CROWD_PROXY_SPECS:
      body = model.body(crowd_capsule_body_name(crowd_index, capsule.name))
      geom = model.geom(crowd_capsule_geom_name(crowd_index, capsule.name))
      assert body.mocapid >= 0
      assert geom.group == 3
      assert geom.contype == HUMAN_COLLISION_TYPE

  assert len(SOMA_CROWD_PROXY_SPECS) == 5
  assert SOMA_CROWD_PROXY_SPECS[0].name == "body_head"
  assert SOMA_CROWD_PROXY_SPECS[0].radius_m == 0.165


def test_ray_only_crowd_remains_lidar_visible_without_contacts() -> None:
  model = get_soma_capsule_crowd_spec(collidable=False).compile()

  assert model.nmocap == 1
  assert model.nbody == 2
  assert model.ngeom == HUMAN_CROWD_CAPACITY * len(SOMA_CROWD_PROXY_SPECS)
  assert np.all(model.geom_group == HUMAN_RAYCAST_GROUP)
  assert np.all(model.geom_contype == 0)


def test_soma_proxy_has_18_independent_lidar_visible_mocap_capsules() -> None:
  model = get_soma_capsule_human_spec().compile()

  assert model.nmocap == model.ngeom == len(SOMA_CAPSULE_SPECS) == 18
  assert np.array_equal(model.body_mocapid[1:], np.arange(18))
  assert np.all(model.geom_group == HUMAN_RAYCAST_GROUP)
  assert np.all(model.geom_contype == HUMAN_COLLISION_TYPE)
  assert np.all(model.geom_conaffinity == 0)
  assert np.all(model.geom_rgba[:, 3] == HUMAN_PROXY_RENDER_ALPHA)
  assert model.geom("human_capsule_geom_head").type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
  assert (
    model.geom("human_capsule_geom_left_forearm").type[0]
    == mujoco.mjtGeom.mjGEOM_CAPSULE
  )


def test_ray_only_primary_human_remains_lidar_visible_without_contacts() -> None:
  model = get_soma_capsule_human_spec(collidable=False).compile()

  assert model.nmocap == 1
  assert model.nbody == 2
  assert model.ngeom == len(SOMA_CAPSULE_SPECS)
  assert np.all(model.geom_group == HUMAN_RAYCAST_GROUP)
  assert np.all(model.geom_contype == 0)
  assert np.all(model.geom_conaffinity == 0)


def test_every_capsule_follows_its_mocap_pose() -> None:
  model = get_soma_capsule_human_spec().compile()
  data = mujoco.MjData(model)
  expected = np.column_stack(
    (
      np.linspace(0.0, 1.7, model.nmocap),
      np.linspace(-0.5, 0.5, model.nmocap),
      np.linspace(0.2, 1.9, model.nmocap),
    )
  )
  data.mocap_pos[:] = expected
  data.mocap_quat[:, 0] = 1.0
  mujoco.mj_forward(model, data)

  assert np.allclose(data.geom_xpos, expected)
  assert not np.any(data.geom_xpos[:, 2] == HUMAN_INACTIVE_HEIGHT_M)


def test_g1_collides_with_human_collision_bit() -> None:
  model = get_g1_with_mid360_spec().compile()
  physical = model.geom_contype != 0

  assert np.all((model.geom_conaffinity[physical] & HUMAN_COLLISION_TYPE) != 0)


def test_human_replaces_static_obstacle_field_in_both_tasks() -> None:
  for (
    cfg,
    expected_update_hz,
    expected_min_count,
    expected_min_radius,
    expected_max_radius,
    expected_radial_jitter,
    expected_randomize_density,
  ) in (
    (
      unitree_g1_obstacle_aware_tracking_env_cfg(play=True),
      10.0,
      0,
      2.0,
      4.0,
      0.25,
      True,
    ),
    (
      unitree_g1_nominal_lidar_debug_env_cfg(play=True),
      5.0,
      30,
      3.0,
      3.001,
      0.0,
      False,
    ),
  ):
    assert HUMAN_ENTITY_NAME in cfg.scene.entities
    assert "obstacles" not in cfg.scene.entities
    assert HUMAN_MOTION_EVENT_NAME in cfg.events
    event = cfg.events[HUMAN_MOTION_EVENT_NAME]
    assert event.mode == "step"
    assert event.params["update_hz"] == expected_update_hz
    assert event.params["transition_duration_s"] == 0.2
    assert event.params["min_count"] == expected_min_count
    assert event.params["max_count"] == HUMAN_CROWD_CAPACITY
    assert event.params["min_radius_m"] == expected_min_radius
    assert event.params["max_radius_m"] == expected_max_radius
    assert event.params["target_arc_spacing_m"] == 0.62
    assert event.params["randomize_density"] is expected_randomize_density
    assert event.params["radial_jitter_m"] == expected_radial_jitter
    assert event.params["angular_jitter_fraction"] == 0.0
    assert event.params["min_human_height_m"] == 1.3
    assert event.params["max_human_height_m"] == 1.9
    assert event.params["show_mesh"]
    assert str(event.params["mesh_skin_path"]).endswith("soma_base_skel_minimal.usd")
    assert "skeleton_path_bank_standing_arm_actions_100" in str(
      event.params["skeleton_bank_path"]
    )


def test_combined_task_uses_ray_only_crowd_and_collidable_primary_human() -> None:
  cfg = unitree_g1_crowd_and_human_tracking_env_cfg(play=False)

  crowd_model = cfg.scene.entities[HUMAN_ENTITY_NAME].spec_fn().compile()
  primary_model = cfg.scene.entities[PRIMARY_HUMAN_ENTITY_NAME].spec_fn().compile()
  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}

  assert crowd_model.nmocap == 1
  assert np.all(crowd_model.geom_contype == 0)
  assert primary_model.nmocap == len(SOMA_CAPSULE_SPECS)
  assert np.all(primary_model.geom_contype == HUMAN_COLLISION_TYPE)
  assert cfg.events[HUMAN_MOTION_EVENT_NAME].params["update_hz"] == 5.0
  assert PRIMARY_HUMAN_EVENT_NAME in cfg.events
  primary_event = cfg.events[PRIMARY_HUMAN_EVENT_NAME]
  assert primary_event.params["min_initial_spawn_radius_m"] == 2.0
  assert primary_event.params["max_initial_spawn_radius_m"] == 4.0
  assert primary_event.params["min_human_height_m"] == 1.3
  assert primary_event.params["max_human_height_m"] == 1.9
  assert primary_event.params["update_hz"] == 50.0
  assert HUMAN_CONTACT_SENSOR_NAME not in sensor_names
  assert PRIMARY_HUMAN_CONTACT_SENSOR_NAME in sensor_names
  assert "human_capsule_vectors_b" not in cfg.observations["critic"].terms
  assert "primary_human_capsule_vectors_b" in cfg.observations["critic"].terms
  assert "human_proximity" in cfg.rewards
  assert "human_collision" not in cfg.rewards
  assert "primary_human_collision" in cfg.rewards
