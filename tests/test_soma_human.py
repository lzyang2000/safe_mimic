from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from safe_mimic.assets.soma_human import get_soma_debug_mesh_spec
from safe_mimic.motions.human_capsules import (
  MAX_CROSS_SECTION_PROPORTION,
  MIN_CROSS_SECTION_PROPORTION,
  SOMA_BASE_MESH_HEIGHT_M,
  SOMA_CAPSULE_SPECS,
  fit_soma_capsules,
  sample_body_scales_for_height,
  sample_capsule_domain_randomization,
)
from safe_mimic.motions.soma_mesh import (
  SomaMeshSkin,
  SomaViserSkin,
  cluster_soma_viser_skin,
  prepare_soma_viser_skin,
)


def _joint_positions() -> tuple[tuple[str, ...], np.ndarray]:
  points = {
    "Hips": (0.0, 1.0, 0.0),
    "Spine2": (0.0, 1.25, 0.0),
    "Chest": (0.0, 1.5, 0.0),
    "Head": (0.0, 1.72, 0.0),
    "LeftArm": (-0.2, 1.48, 0.0),
    "LeftForeArm": (-0.45, 1.35, 0.0),
    "LeftHand": (-0.65, 1.22, 0.0),
    "RightArm": (0.2, 1.48, 0.0),
    "RightForeArm": (0.45, 1.35, 0.0),
    "RightHand": (0.65, 1.22, 0.0),
    "LeftLeg": (-0.1, 0.95, 0.0),
    "LeftShin": (-0.1, 0.52, 0.0),
    "LeftFoot": (-0.1, 0.08, 0.0),
    "LeftToeBase": (-0.1, 0.04, 0.18),
    "RightLeg": (0.1, 0.95, 0.0),
    "RightShin": (0.1, 0.52, 0.0),
    "RightFoot": (0.1, 0.08, 0.0),
    "RightToeBase": (0.1, 0.04, 0.18),
  }
  names = tuple(points)
  positions = np.asarray([[points[name] for name in names]], dtype=np.float64)
  return names, positions


def test_capsule_fit_and_domain_randomization() -> None:
  names, positions = _joint_positions()
  randomization = sample_capsule_domain_randomization(np.random.default_rng(7))
  fit = fit_soma_capsules(names, positions, randomization)

  assert fit.centers_m.shape == (1, len(SOMA_CAPSULE_SPECS), 3)
  assert fit.quaternions_wxyz.shape == (1, len(SOMA_CAPSULE_SPECS), 4)
  assert np.allclose(np.linalg.norm(fit.quaternions_wxyz, axis=-1), 1.0)
  assert np.all(fit.radii_m > 0.0)
  assert np.all(fit.half_lengths_m >= 0.0)
  sampled_height = randomization.body_scale_xyz[2] * SOMA_BASE_MESH_HEIGHT_M
  assert 1.3 <= sampled_height <= 1.9
  proportions = randomization.body_scale_xyz[:2] / randomization.body_scale_xyz[2]
  assert np.all(proportions >= MIN_CROSS_SECTION_PROPORTION)
  assert np.all(proportions <= MAX_CROSS_SECTION_PROPORTION)
  assert 0.0 <= randomization.radius_margin_m <= 0.025


def test_torch_body_scales_cover_explicit_height_range() -> None:
  scales = sample_body_scales_for_height(
    4096,
    "cpu",
    min_height_m=1.3,
    max_height_m=1.9,
    generator=torch.Generator().manual_seed(29),
  )
  heights = scales[:, 2] * SOMA_BASE_MESH_HEIGHT_M
  proportions = scales[:, :2] / scales[:, 2, None]

  assert heights.min() == pytest.approx(1.3, abs=5e-4)
  assert heights.max() == pytest.approx(1.9, abs=5e-4)
  assert torch.all(proportions >= MIN_CROSS_SECTION_PROPORTION)
  assert torch.all(proportions <= MAX_CROSS_SECTION_PROPORTION)


def test_soma_debug_mesh_loads_as_noncolliding_mjlab_asset(
  tmp_path: Path,
) -> None:
  mesh_path = tmp_path / "triangle.obj"
  mesh_path.write_text(
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 4 2\nf 2 4 3\nf 3 4 1\n"
  )
  spec = get_soma_debug_mesh_spec(mesh_path)
  model = spec.compile()
  geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "soma_debug_mesh")

  assert geom_id >= 0
  assert model.geom_contype[geom_id] == 0
  assert model.geom_conaffinity[geom_id] == 0


def test_viser_skin_collapses_omitted_joint_weights_to_retained_ancestor() -> None:
  bind = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
  bind[1, 3, :3] = (1.0, 2.0, 3.0)
  bind[2, 3, :3] = (4.0, 5.0, 6.0)
  skin = SomaMeshSkin(
    points_cm=np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
    triangles=np.asarray(((0, 1, 1),), dtype=np.int32),
    joint_names=("Root", "Hand", "Finger"),
    joint_paths=(("Root",), ("Root", "Hand"), ("Root", "Hand", "Finger")),
    joint_indices=np.asarray(((0,), (2,)), dtype=np.int32),
    joint_weights=np.ones((2, 1)),
    bind_transforms=bind,
    geom_bind_transform=np.eye(4),
  )

  prepared = prepare_soma_viser_skin(skin, ("Root", "Hand"))

  np.testing.assert_allclose(
    prepared.vertices_m, ((0.01, 0.03, 0.02), (0.04, 0.06, 0.05))
  )
  np.testing.assert_allclose(prepared.skin_weights, ((1.0, 0.0), (0.0, 1.0)))
  np.testing.assert_allclose(prepared.bind_positions_m[1], (0.01, 0.03, 0.02))
  np.testing.assert_allclose(
    prepared.bind_quaternions_wxyz,
    np.broadcast_to((1.0, 0.0, 0.0, 0.0), (2, 4)),
  )

  posed = prepared.skin_vertices_mujoco(
    prepared.bind_positions_m,
    prepared.bind_quaternions_wxyz,
    body_scale_xyz=np.asarray((2.0, 3.0, 4.0)),
    placement_yaw=np.pi / 2.0,
    translation_w=np.asarray((10.0, 20.0, 30.0)),
  )
  expected = prepared.vertices_m * (2.0, 3.0, 4.0)
  expected = np.column_stack((-expected[:, 1], expected[:, 0], expected[:, 2]))
  expected += (10.0, 20.0, 30.0)
  np.testing.assert_allclose(posed, expected, atol=1e-6)


def test_viser_skin_vertex_clustering_preserves_articulation_weights() -> None:
  skin = SomaViserSkin(
    vertices_m=np.asarray(
      ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
      dtype=np.float32,
    ),
    triangles=np.asarray(((0, 2, 3), (1, 2, 3)), dtype=np.uint32),
    joint_names=("Root",),
    bind_positions_m=np.zeros((1, 3), dtype=np.float32),
    bind_quaternions_wxyz=np.asarray(((1.0, 0.0, 0.0, 0.0),), dtype=np.float32),
    bind_inverse_matrices=np.asarray((np.eye(4),), dtype=np.float64),
    skin_weights=np.ones((4, 1), dtype=np.float32),
  )

  clustered = cluster_soma_viser_skin(skin, 0.01)

  assert clustered.vertices_m.shape == (3, 3)
  assert clustered.triangles.shape == (1, 3)
  np.testing.assert_allclose(clustered.skin_weights.sum(axis=1), 1.0)
