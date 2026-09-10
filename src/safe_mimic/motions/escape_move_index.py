"""Index each clip's travel phase so a planner can pick an escape move.

For every clip the fastest ``window_s`` window of root travel is located. A
clip is an escape candidate when that window covers at least ``min_travel_m``.
The planner enters the clip ``lead_s`` before the window (so the step reads as
a step), and leaves it at the first frame after the window where the root
planar speed drops below ``exit_speed_mps`` (the natural end of the travel),
clamped strictly inside the clip. The travel direction is expressed in the
anchor body's heading frame at the entry frame; the live alignment glues that
heading to the robot's, so the direction is directly comparable with an
escape direction in the robot frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class EscapeMoveIndexCfg:
  window_s: float = 0.6
  min_travel_m: float = 0.3
  lead_s: float = 0.1
  exit_speed_mps: float = 0.3


@dataclass(frozen=True)
class ClipTravel:
  entry_frame: int
  exit_frame: int
  direction_b: tuple[float, float]
  speed_mps: float
  travel_m: float
  candidate: bool


def quat_yaw(q: np.ndarray) -> np.ndarray:
  """Yaw angle of a (w, x, y, z) quaternion array."""
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clip_travel(
  root_xy: np.ndarray,
  anchor_quat_wxyz: np.ndarray,
  fps: float,
  cfg: EscapeMoveIndexCfg,
) -> ClipTravel:
  root_xy = np.asarray(root_xy, dtype=np.float64)
  n = int(root_xy.shape[0])
  window = max(1, int(round(cfg.window_s * fps)))
  if n <= window:
    return ClipTravel(0, max(0, n - 1), (1.0, 0.0), 0.0, 0.0, False)
  disp = root_xy[window:] - root_xy[:-window]
  travel = np.linalg.norm(disp, axis=1)
  i = int(np.argmax(travel))
  travel_m = float(travel[i])
  if travel_m < 1e-6:
    return ClipTravel(0, n - 1, (1.0, 0.0), 0.0, 0.0, False)
  entry = max(0, i - int(round(cfg.lead_s * fps)))
  yaw = float(quat_yaw(np.asarray(anchor_quat_wxyz, dtype=np.float64)[entry]))
  c, s = np.cos(-yaw), np.sin(-yaw)
  d = disp[i] / travel_m
  direction_b = (float(c * d[0] - s * d[1]), float(s * d[0] + c * d[1]))
  speed = np.linalg.norm(np.diff(root_xy, axis=0), axis=1) * fps  # k -> k + 1
  exit_frame = n - 1
  for k in range(i + window, n - 1):
    if speed[k] < cfg.exit_speed_mps:
      exit_frame = k
      break
  return ClipTravel(
    entry_frame=int(entry),
    exit_frame=int(min(exit_frame, n - 1)),
    direction_b=direction_b,
    speed_mps=travel_m / cfg.window_s,
    travel_m=travel_m,
    candidate=travel_m >= cfg.min_travel_m,
  )


def build_escape_move_index(
  manifest: str | Path, anchor_body_index: int, cfg: EscapeMoveIndexCfg
) -> dict:
  manifest = Path(manifest)
  data = yaml.safe_load(manifest.read_text())
  root = Path(data["root_path"])
  if not root.is_absolute():
    root = (manifest.parent / root).resolve()
  fps = float(data["fps"])
  clips = []
  for entry in data["motions"]:
    with np.load(root / entry["file"]) as arrays:
      travel = clip_travel(
        arrays["body_pos_w"][:, 0, :2],
        arrays["body_quat_w"][:, anchor_body_index],
        fps,
        cfg,
      )
    clips.append({"file": entry["file"], **asdict(travel)})
  return {
    "fps": fps,
    "anchor_body_index": int(anchor_body_index),
    "window_s": cfg.window_s,
    "clips": clips,
  }


def load_escape_move_index(path: str | Path) -> dict:
  return json.loads(Path(path).read_text())


__all__ = [
  "ClipTravel",
  "EscapeMoveIndexCfg",
  "build_escape_move_index",
  "clip_travel",
  "load_escape_move_index",
  "quat_yaw",
]
