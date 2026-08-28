import csv
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.tracking.mdp.commands import MotionLoader

from safe_mimic.motions.g1_tracker_npz import (
  TRACKER_NPZ_FIELDS,
  convert_g1_csv_to_tracker_npz,
  quaternion_angular_velocity_w,
  validate_tracker_npz,
)


def _write_motion(path: Path) -> None:
  frame_count = 121
  raw = np.zeros((frame_count, 36), dtype=np.float64)
  raw[:, 0] = np.arange(frame_count)
  raw[:, 1] = np.linspace(0.0, 100.0, frame_count)
  raw[:, 3] = 80.0
  raw[:, 6] = np.linspace(0.0, 90.0, frame_count)
  header = ["Frame", "root_tX", "root_tY", "root_tZ"]
  header += ["root_rX", "root_rY", "root_rZ"]
  header += [f"joint_{index}" for index in range(29)]
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(raw)


def test_tracker_npz_matches_mjlab_schema_at_50_hz(tmp_path: Path) -> None:
  csv_path = tmp_path / "motion.csv"
  output_path = tmp_path / "motion.npz"
  _write_motion(csv_path)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)

  arrays = convert_g1_csv_to_tracker_npz(model, data, csv_path, output_path)

  assert tuple(arrays) == TRACKER_NPZ_FIELDS
  assert arrays["fps"] == 50.0
  assert arrays["joint_pos"].shape == (51, 29)
  assert arrays["body_pos_w"].shape == (51, 30, 3)
  assert arrays["body_quat_w"].shape == (51, 30, 4)
  assert arrays["body_lin_vel_w"].shape == (51, 30, 3)
  assert arrays["body_ang_vel_w"].shape == (51, 30, 3)
  validate_tracker_npz(output_path)

  with np.load(output_path) as saved:
    assert tuple(saved.files) == TRACKER_NPZ_FIELDS
    assert saved["fps"] == 50.0

  loader = MotionLoader(str(output_path), torch.arange(30), device="cpu")
  assert loader.time_step_total == 51
  assert loader.joint_pos.shape == (51, 29)
  assert loader.body_pos_w.shape == (51, 30, 3)


def test_world_angular_velocity_from_known_yaw() -> None:
  frame_count = 11
  yaw = np.linspace(0.0, 1.0, frame_count)
  xyzw = np.zeros((frame_count, 1, 4), dtype=np.float64)
  xyzw[:, 0, 2] = np.sin(yaw / 2.0)
  xyzw[:, 0, 3] = np.cos(yaw / 2.0)
  wxyz = np.roll(xyzw, shift=1, axis=-1)

  velocity = quaternion_angular_velocity_w(wxyz, fps=10.0)

  assert np.allclose(velocity[..., :2], 0.0, atol=1e-8)
  assert np.allclose(velocity[..., 2], 1.0, atol=1e-6)
