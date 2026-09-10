"""Trim leading and trailing standing stance from tracker NPZ clips.

Motion-capture takes start and end with the performer standing still, often
for several seconds. For reference tracking that idle stance is dead time:
the robot stands, the crowd walks. :func:`trim_bounds` finds the first and
last "active" frame (joint-speed norm or root planar speed above threshold,
majority-smoothed over a short window) and keeps at most ``keep_s`` of stance
on each side. Clips with no active frame, or whose stance is already short,
come back untouched; the result never drops below ``min_length_s``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

TIME_MAJOR_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


@dataclass(frozen=True)
class StanceTrimCfg:
  keep_s: float = 1.0
  """Stance retained before the first / after the last active frame."""

  joint_speed_rps: float = 1.0
  """Joint-velocity norm (all joints) above which a frame counts as moving."""

  root_speed_mps: float = 0.15
  """Root planar speed above which a frame counts as moving."""

  smoothing_frames: int = 5
  """Majority vote window (frames) applied to the raw activity mask."""

  min_length_s: float = 2.0
  """Never trim a clip shorter than this (expanded symmetrically if needed)."""


def active_frame_mask(
  joint_vel: np.ndarray, root_lin_vel_xy: np.ndarray, cfg: StanceTrimCfg
) -> np.ndarray:
  """Boolean per-frame mask: True where the performer is moving."""
  joint_speed = np.linalg.norm(np.asarray(joint_vel, dtype=np.float64), axis=-1)
  root_speed = np.linalg.norm(np.asarray(root_lin_vel_xy, dtype=np.float64), axis=-1)
  raw = (joint_speed > cfg.joint_speed_rps) | (root_speed > cfg.root_speed_mps)
  k = max(1, int(cfg.smoothing_frames))
  if k == 1:
    return raw
  kernel = np.ones(k) / k
  return np.convolve(raw.astype(np.float64), kernel, mode="same") > 0.5


def trim_bounds(
  joint_vel: np.ndarray,
  root_lin_vel_xy: np.ndarray,
  fps: float,
  cfg: StanceTrimCfg,
) -> tuple[int, int]:
  """Return ``(start, end)`` frame bounds (end exclusive) after trimming."""
  n = int(joint_vel.shape[0])
  active = np.nonzero(active_frame_mask(joint_vel, root_lin_vel_xy, cfg))[0]
  if active.size == 0:
    return 0, n
  keep = int(round(cfg.keep_s * fps))
  start = max(0, int(active[0]) - keep)
  end = min(n, int(active[-1]) + 1 + keep)
  min_frames = min(n, int(round(cfg.min_length_s * fps)))
  short = min_frames - (end - start)
  if short > 0:
    # Expand symmetrically around the active span, clamped to the clip.
    grow_front = short // 2
    start = max(0, start - grow_front)
    end = min(n, end + (min_frames - (end - start)))
    start = max(0, end - min_frames)
  return start, end


def trim_motion_arrays(
  arrays: Mapping[str, np.ndarray], cfg: StanceTrimCfg
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
  """Slice every time-major array of a tracker NPZ to the trimmed bounds."""
  fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
  start, end = trim_bounds(
    arrays["joint_vel"], arrays["body_lin_vel_w"][:, 0, :2], fps, cfg
  )
  out: dict[str, np.ndarray] = {}
  for key, value in arrays.items():
    out[key] = value[start:end] if key in TIME_MAJOR_KEYS else np.asarray(value)
  return out, (start, end)


__all__ = [
  "StanceTrimCfg",
  "TIME_MAJOR_KEYS",
  "active_frame_mask",
  "trim_bounds",
  "trim_motion_arrays",
]
