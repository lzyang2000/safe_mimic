"""Travel-phase index used to pick escape moves."""

import json

import numpy as np

from safe_mimic.motions.escape_move_index import (
  ClipTravel,
  EscapeMoveIndexCfg,
  clip_travel,
  load_escape_move_index,
  quat_yaw,
)

FPS = 50.0


def _yaw_quat(yaw: float) -> np.ndarray:
  return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def test_quat_yaw_roundtrip() -> None:
  assert np.isclose(quat_yaw(_yaw_quat(0.7)), 0.7)


def test_static_clip_is_not_a_candidate() -> None:
  n = 100
  travel = clip_travel(
    np.zeros((n, 2)), np.tile(_yaw_quat(0.0), (n, 1)), FPS, EscapeMoveIndexCfg()
  )
  assert travel.candidate is False and travel.travel_m == 0.0


def test_travel_segment_is_indexed_in_the_anchor_frame() -> None:
  # 1 s still, 1 s travelling 1.0 m along world +y while heading +90 deg
  # (so the travel is straight ahead in the body frame), 1 s still.
  still = int(FPS)
  xy = np.zeros((3 * still, 2))
  xy[still : 2 * still, 1] = np.linspace(0.0, 1.0, still)
  xy[2 * still :, 1] = 1.0
  quat = np.tile(_yaw_quat(np.pi / 2), (3 * still, 1))
  t = clip_travel(xy, quat, FPS, EscapeMoveIndexCfg())
  assert isinstance(t, ClipTravel) and t.candidate
  assert t.travel_m > 0.55
  assert abs(t.speed_mps - t.travel_m / 0.6) < 1e-6
  assert np.allclose(t.direction_b, (1.0, 0.0), atol=1e-6)
  # entry = fastest-window start minus the 0.1 s lead; window starts inside
  # the travel segment.
  assert still - 5 <= t.entry_frame <= still + 15
  # exit: first frame after the window where the speed drops below 0.3 m/s,
  # i.e. the trailing stance.
  assert 2 * still - 2 <= t.exit_frame <= 2 * still + 3
  assert t.exit_frame <= 3 * still - 1


def test_exit_frame_stays_inside_the_clip() -> None:
  n = 40  # 0.8 s clip that travels to its very end
  xy = np.stack([np.linspace(0.0, 1.0, n), np.zeros(n)], axis=1)
  t = clip_travel(xy, np.tile(_yaw_quat(0.0), (n, 1)), FPS, EscapeMoveIndexCfg())
  assert t.candidate and t.exit_frame == n - 1


def test_index_roundtrip(tmp_path) -> None:
  payload = {"fps": 50.0, "anchor_body_index": 15, "window_s": 0.6, "clips": []}
  path = tmp_path / "escape_moves.json"
  path.write_text(json.dumps(payload))
  assert load_escape_move_index(path) == payload
