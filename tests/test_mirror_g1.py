"""Left/right mirroring of G1 tracker clips."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from safe_mimic.motions.mirror_g1 import (
  G1MirrorSpec,
  mirror_motion_arrays,
  mirror_name,
)

CLIP = Path(
  "artifacts/bones-seed/datasets/g1_ballet_v1_trim1s/npz_50hz/231006/"
  "dance_pirouette_001__A464.npz"
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(G1_XML))


@pytest.fixture(scope="module")
def spec(model) -> G1MirrorSpec:
  return G1MirrorSpec.from_model(model)


def test_mirror_name_swaps_sides() -> None:
  assert mirror_name("left_hip_roll_joint") == "right_hip_roll_joint"
  assert mirror_name("right_wrist_yaw_link") == "left_wrist_yaw_link"
  assert mirror_name("waist_yaw_joint") == "waist_yaw_joint"


def test_spec_permutes_sides_and_flips_roll_yaw(spec) -> None:
  j = spec.joint_names
  assert spec.joint_perm[j.index("left_knee_joint")] == j.index("right_knee_joint")
  assert spec.joint_perm[j.index("waist_pitch_joint")] == j.index("waist_pitch_joint")
  assert spec.joint_sign[j.index("left_knee_joint")] == 1.0  # pitch keeps sign
  assert spec.joint_sign[j.index("left_hip_roll_joint")] == -1.0
  assert spec.joint_sign[j.index("waist_yaw_joint")] == -1.0
  b = spec.body_names
  assert b[0] == "pelvis" and "world" not in b
  assert spec.body_perm[b.index("left_elbow_link")] == b.index("right_elbow_link")
  assert spec.body_perm[b.index("torso_link")] == b.index("torso_link")


def test_mirror_is_an_involution(spec) -> None:
  with np.load(CLIP) as data:
    arrays = {k: data[k] for k in data.files}
  twice = mirror_motion_arrays(mirror_motion_arrays(arrays, spec), spec)
  for key, value in arrays.items():
    np.testing.assert_allclose(twice[key], value, atol=1e-6)
    assert twice[key].dtype == value.dtype


def _fk_body_positions(model, root_pos, root_quat, joint_pos) -> np.ndarray:
  data = mujoco.MjData(model)
  data.qpos[:3] = root_pos
  data.qpos[3:7] = root_quat
  data.qpos[7:] = joint_pos
  mujoco.mj_kinematics(model, data)
  return data.xpos[1:].copy()  # bodies minus world, model order


def test_fk_reproduces_original_and_mirrored_clouds(model, spec) -> None:
  with np.load(CLIP) as data:
    arrays = {k: data[k] for k in data.files}
  mirrored = mirror_motion_arrays(arrays, spec)
  for clip in (arrays, mirrored):
    for frame in (0, 50, 150):
      fk = _fk_body_positions(
        model,
        clip["body_pos_w"][frame, 0],
        clip["body_quat_w"][frame, 0],
        clip["joint_pos"][frame],
      )
      np.testing.assert_allclose(fk, clip["body_pos_w"][frame], atol=5e-3)


def test_mirrored_clip_travels_the_other_way(spec) -> None:
  with np.load(CLIP) as data:
    arrays = {k: data[k] for k in data.files}
  mirrored = mirror_motion_arrays(arrays, spec)
  source = arrays["body_pos_w"][:, spec.body_perm]
  np.testing.assert_allclose(mirrored["body_pos_w"][..., 0], source[..., 0])
  np.testing.assert_allclose(mirrored["body_pos_w"][..., 1], -source[..., 1])
  np.testing.assert_allclose(mirrored["body_pos_w"][..., 2], source[..., 2])
