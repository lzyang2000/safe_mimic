"""Pure planner logic for escape moves (cfg, table, trigger, selection, blend)."""

import math
from pathlib import Path

import pytest
import torch

from safe_mimic.tasks.escape_moves import (
  EscapeMoveCfg,
  EscapeMoveTable,
  blend_alpha,
  body_frame_planar,
  nlerp,
  select_escape_moves,
  update_trigger_count,
)


def test_cfg_defaults_and_validation() -> None:
  cfg = EscapeMoveCfg(index_file="x.json")
  assert (cfg.trigger_speed_mps, cfg.trigger_steps, cfg.min_alignment) == (0.4, 3, 0.7)
  assert (cfg.speed_cap_mps, cfg.pose_distance_weight) == (1.0, 0.5)
  assert (cfg.blend_s, cfg.cooldown_s, cfg.trigger_source) == (0.3, 1.0, "teacher")
  with pytest.raises(ValueError, match="trigger_source"):
    EscapeMoveCfg(index_file="x.json", trigger_source="oracle")
  with pytest.raises(ValueError):
    EscapeMoveCfg(index_file="x.json", trigger_steps=0)


def _yaw_quat(yaw: float) -> torch.Tensor:
  return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def test_body_frame_planar_rotates_by_minus_yaw() -> None:
  # Robot heading +90 deg: world +y is straight ahead (+x in the body frame).
  out = body_frame_planar(torch.tensor([[0.0, 1.0]]), _yaw_quat(math.pi / 2)[None])
  torch.testing.assert_close(out, torch.tensor([[1.0, 0.0]]), atol=1e-6, rtol=0)


def test_trigger_count_increments_and_resets() -> None:
  count = torch.tensor([0, 2, 5])
  speed = torch.tensor([0.3, 0.1, 0.25])
  torch.testing.assert_close(
    update_trigger_count(count, speed, 0.25), torch.tensor([1, 0, 6])
  )


def _table() -> EscapeMoveTable:
  # Three candidates: straight ahead (fast), left (slow), back (very fast).
  return EscapeMoveTable(
    entry_frames=torch.tensor([12, 32, 52]),
    exit_frames=torch.tensor([18, 38, 58]),
    clip_starts=torch.tensor([10, 30, 50]),
    clip_ends=torch.tensor([20, 40, 60]),
    direction_b=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    speed_mps=torch.tensor([0.9, 0.5, 2.0]),
    entry_joint_pos=torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
  )


def test_selection_picks_the_aligned_move_and_gates_on_alignment() -> None:
  cfg = EscapeMoveCfg(index_file="x.json")
  joints = torch.zeros(3, 2)
  escape = torch.tensor([[0.5, 0.0], [0.0, 0.4], [0.3, 0.3]])  # ahead, left, 45 deg
  chosen = select_escape_moves(_table(), escape, joints, cfg)
  # 45 deg: cos to ahead and to left are both 0.707 >= 0.7; ahead is faster.
  assert chosen.tolist() == [0, 1, 0]


def test_selection_returns_minus_one_without_a_qualifying_move() -> None:
  cfg = EscapeMoveCfg(index_file="x.json")
  joints = torch.zeros(2, 2)
  escape = torch.tensor([[0.0, -1.0], [0.0, 0.0]])  # right (no candidate), no push
  assert select_escape_moves(_table(), escape, joints, cfg).tolist() == [-1, -1]


def test_selection_caps_speed_and_penalises_pose_distance() -> None:
  cfg = EscapeMoveCfg(index_file="x.json", speed_cap_mps=1.0, pose_distance_weight=0.5)
  # Escape backwards: the only aligned move is the fast "back" clip, chosen
  # even though its entry pose is 1 rad away (score 1.0 - 0.5 > -inf).
  chosen = select_escape_moves(
    _table(), torch.tensor([[-1.0, 0.0]]), torch.zeros(1, 2), cfg
  )
  assert chosen.tolist() == [2]
  # Two aligned moves with equal (capped) speed: the closer pose wins.
  table = _table()
  table.direction_b[2] = torch.tensor([1.0, 0.0])  # now also "ahead", speed 2 -> cap 1
  table.speed_mps[0] = 1.0
  chosen = select_escape_moves(
    table, torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2), cfg
  )
  assert chosen.tolist() == [0]


def test_blend_alpha_schedule() -> None:
  alpha = blend_alpha(torch.tensor([4, 2, 0, -1]), 4)
  torch.testing.assert_close(alpha, torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_nlerp_endpoints_and_sign_alignment() -> None:
  q0 = _yaw_quat(0.0)[None]
  q1 = _yaw_quat(1.0)[None]
  torch.testing.assert_close(nlerp(q0, q1, torch.tensor([[0.0]])), q0)
  torch.testing.assert_close(nlerp(q0, q1, torch.tensor([[1.0]])), q1)
  mid = nlerp(q0, -q1, torch.tensor([[0.5]]))  # antipodal input, same rotation
  torch.testing.assert_close(mid, nlerp(q0, q1, torch.tensor([[0.5]])))
  assert torch.allclose(torch.linalg.vector_norm(mid, dim=-1), torch.ones(1))


def test_table_from_index_maps_files_and_gathers_entry_poses() -> None:
  index = {
    "fps": 50.0,
    "clips": [
      {
        "file": "a/one.npz",
        "entry_frame": 2,
        "exit_frame": 4,
        "direction_b": [1.0, 0.0],
        "speed_mps": 0.8,
        "travel_m": 0.5,
        "candidate": True,
      },
      {
        "file": "a/two.npz",
        "entry_frame": 0,
        "exit_frame": 3,
        "direction_b": [0.0, 1.0],
        "speed_mps": 0.1,
        "travel_m": 0.05,
        "candidate": False,
      },
    ],
  }
  # Library order differs from index order: two.npz first.
  paths = [Path("/lib/a/two.npz"), Path("/lib/a/one.npz")]
  clip_start = torch.tensor([0, 5])
  clip_frames = torch.tensor([5, 6])
  joint_pos = torch.arange(11.0)[:, None].repeat(1, 2)
  table = EscapeMoveTable.from_index(
    index, paths, clip_start, clip_frames, joint_pos, "cpu"
  )
  assert table.entry_frames.tolist() == [7]  # 5 + 2, global
  assert table.exit_frames.tolist() == [9]
  assert table.clip_starts.tolist() == [5] and table.clip_ends.tolist() == [11]
  torch.testing.assert_close(table.entry_joint_pos, torch.tensor([[7.0, 7.0]]))
  with pytest.raises(ValueError, match="not in the library"):
    EscapeMoveTable.from_index(
      index, paths[:1], clip_start, clip_frames, joint_pos, "cpu"
    )
