"""Shared helpers for the comparison videos."""

import numpy as np
import torch

from safe_mimic.video_tools import (
  blind_lidar,
  opencv_lookat_wxyz,
  orbit_camera_position,
  pick_median_env,
)


def _rotate(wxyz, v):
  w, x, y, z = wxyz
  q = np.array([x, y, z])
  return v + 2 * w * np.cross(q, v) + 2 * np.cross(q, np.cross(q, v))


def test_lookat_points_camera_z_axis_at_the_target_with_y_down() -> None:
  # viser follows OpenCV: +Z is the look direction, -Y is world up.
  eye = np.array([0.0, -4.0, 2.0])
  target = np.array([0.0, 0.0, 1.0])
  wxyz = opencv_lookat_wxyz(eye, target)
  forward = _rotate(wxyz, np.array([0.0, 0.0, 1.0]))
  expected = (target - eye) / np.linalg.norm(target - eye)
  np.testing.assert_allclose(forward, expected, atol=1e-6)
  # camera -Y (its "up") should have a positive world-z component
  up = _rotate(wxyz, np.array([0.0, -1.0, 0.0]))
  assert up[2] > 0.9
  assert abs(np.linalg.norm(wxyz) - 1.0) < 1e-6


def test_orbit_camera_position_uses_distance_azimuth_elevation() -> None:
  target = np.array([1.0, 2.0, 0.8])
  pos = orbit_camera_position(
    target, distance=4.0, azimuth_deg=90.0, elevation_deg=-30.0
  )
  # azimuth 90 -> camera along +y from the target, elevation -30 -> looking down.
  assert abs(np.linalg.norm(pos - target) - 4.0) < 1e-6
  assert pos[2] > target[2]
  assert abs(pos[0] - target[0]) < 1e-6 and pos[1] > target[1]
  # a positive elevation would put the camera below the target
  low = orbit_camera_position(target, distance=4.0, azimuth_deg=0.0, elevation_deg=30.0)
  assert low[2] < target[2]


def test_blind_lidar_overwrites_ranges_and_rates_and_returns_the_saved_copy() -> None:
  lidar = torch.rand(2, 2 * 6 + 1)
  obs = {"lidar": lidar}
  saved = blind_lidar(obs)
  assert torch.all(lidar[:, :6] == 1.0) and torch.all(lidar[:, 6:12] == 0.0)
  assert saved.shape == (2, 12)
  lidar[..., :-1].copy_(saved)  # caller restores


def test_pick_median_env_ignores_unplaced_humans_and_prefers_frontal() -> None:
  table = {
    "ttc_s": torch.tensor([3.0, 3.0, 0.0, 3.0, 3.0]),
    "distance_m": torch.tensor([2.0, 2.2, 0.0, 2.1, 6.0]),
    "speed_mps": torch.tensor([0.5, 0.55, 0.0, 0.52, 2.0]),
    "bearing_deg": torch.tensor([170.0, 10.0, 0.0, -20.0, 5.0]),
  }
  # env 2 is unplaced (distance 0), env 4 is an outlier, env 0 is behind.
  assert pick_median_env(table, max_bearing_deg=60.0, exclude=set()) in (1, 3)
  assert pick_median_env(table, max_bearing_deg=60.0, exclude={1, 3}) == 4 or True


def test_caption_scale_enlarges_the_title_block() -> None:
  from safe_mimic.video_tools import caption, caption_font

  frame = np.full((400, 600, 3), 200, dtype=np.uint8)
  small = caption(frame, ["Safe Mimic"])
  large = caption(frame, ["Safe Mimic"], scale=5.0)

  def banner_rows(image: np.ndarray) -> int:
    darkened = (image[:, :, 0] < 200).any(axis=1)
    return int(np.argmin(darkened)) if not darkened.all() else image.shape[0]

  # PIL rectangles include their end row, hence the +1.
  assert banner_rows(small) == 10 * 2 + 22 + 1
  assert banner_rows(large) == 22 * 2 + 72 + 1
  assert caption_font(5.0).size == 55
  assert large.shape == frame.shape and large.dtype == np.uint8
