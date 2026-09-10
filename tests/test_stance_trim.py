"""Trim leading/trailing standing stance from reference clips (2026-09-09)."""

import numpy as np
import pytest

from safe_mimic.motions.stance_trim import (
  StanceTrimCfg,
  active_frame_mask,
  trim_bounds,
  trim_motion_arrays,
)

FPS = 50.0


def _clip(lead_s: float, active_s: float, trail_s: float) -> dict[str, np.ndarray]:
  n_lead, n_act, n_trail = (int(round(x * FPS)) for x in (lead_s, active_s, trail_s))
  n = n_lead + n_act + n_trail
  joint_vel = np.zeros((n, 29), dtype=np.float32)
  joint_vel[n_lead : n_lead + n_act] = 0.8  # norm ~4.3 rad/s: clearly moving
  body_lin_vel = np.zeros((n, 30, 3), dtype=np.float32)
  return {
    "fps": np.array([FPS]),
    "joint_pos": np.arange(n, dtype=np.float32)[:, None].repeat(29, 1),
    "joint_vel": joint_vel,
    "body_pos_w": np.zeros((n, 30, 3), dtype=np.float32),
    "body_quat_w": np.zeros((n, 30, 4), dtype=np.float32),
    "body_lin_vel_w": body_lin_vel,
    "body_ang_vel_w": np.zeros((n, 30, 3), dtype=np.float32),
  }


def test_active_mask_uses_joint_speed_or_root_speed() -> None:
  joint_vel = np.zeros((10, 29))
  root_xy = np.zeros((10, 2))
  joint_vel[2] = 0.5  # norm 2.7 > 1.0
  root_xy[7] = (0.2, 0.0)  # 0.2 m/s > 0.15
  mask = active_frame_mask(joint_vel, root_xy, StanceTrimCfg(smoothing_frames=1))
  assert mask.tolist() == [i in (2, 7) for i in range(10)]


def test_trim_bounds_keep_one_second_each_side() -> None:
  clip = _clip(lead_s=3.0, active_s=4.0, trail_s=2.5)
  cfg = StanceTrimCfg(keep_s=1.0)
  start, end = trim_bounds(
    clip["joint_vel"], clip["body_lin_vel_w"][:, 0, :2], FPS, cfg
  )
  # 3 s of lead -> keep the last 1 s of it; 2.5 s of trail -> keep the first 1 s.
  assert start == pytest.approx(2.0 * FPS, abs=cfg.smoothing_frames)
  assert end == pytest.approx((3.0 + 4.0 + 1.0) * FPS, abs=cfg.smoothing_frames)


def test_trim_bounds_leave_short_stance_alone() -> None:
  clip = _clip(lead_s=0.4, active_s=3.0, trail_s=0.2)
  n = clip["joint_vel"].shape[0]
  assert trim_bounds(
    clip["joint_vel"], clip["body_lin_vel_w"][:, 0, :2], FPS, StanceTrimCfg()
  ) == (0, n)


def test_trim_bounds_untouched_when_nothing_moves() -> None:
  clip = _clip(lead_s=3.0, active_s=0.0, trail_s=0.0)
  n = clip["joint_vel"].shape[0]
  assert trim_bounds(
    clip["joint_vel"], clip["body_lin_vel_w"][:, 0, :2], FPS, StanceTrimCfg()
  ) == (0, n)


def test_trim_bounds_respects_minimum_length() -> None:
  clip = _clip(lead_s=3.0, active_s=0.3, trail_s=3.0)
  cfg = StanceTrimCfg(keep_s=0.5, min_length_s=3.0)
  start, end = trim_bounds(
    clip["joint_vel"], clip["body_lin_vel_w"][:, 0, :2], FPS, cfg
  )
  assert end - start >= int(round(3.0 * FPS))
  assert 0 <= start and end <= clip["joint_vel"].shape[0]
  # still centred on the active burst
  burst = int(round(3.0 * FPS))
  assert start <= burst and end >= burst + int(round(0.3 * FPS))


def test_trim_motion_arrays_slices_every_time_major_key() -> None:
  clip = _clip(lead_s=2.0, active_s=1.0, trail_s=2.0)
  out, (start, end) = trim_motion_arrays(clip, StanceTrimCfg(keep_s=1.0))
  n_out = end - start
  for key in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    assert out[key].shape[0] == n_out
    assert out[key].dtype == clip[key].dtype
  np.testing.assert_array_equal(
    out["joint_pos"][:, 0], np.arange(start, end, dtype=np.float32)
  )
  assert float(out["fps"][0]) == FPS
  assert n_out == pytest.approx(3.0 * FPS, abs=2 * StanceTrimCfg().smoothing_frames)
