"""Escape moves: answer a predicted collision with a travelling ballet step.

Pure logic for the planner that lives in the filtered replay command. The
command owns the per-env state; these functions decide WHEN to switch (a
sustained planar correction from the CBF filter, or the actor's own planar
prediction at deployment), WHICH indexed move to enter (travel direction
aligned with the escape direction, fast, close in joint space) and HOW the raw
reference is blended across the switch and the resume.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

TRIGGER_SOURCES = ("teacher", "actor")


@dataclass
class EscapeMoveCfg:
  index_file: str
  """Travel index JSON (``build_escape_move_index.py``) for the manifest."""

  trigger_speed_mps: float = 0.4
  """Planar trigger norm at or above which a step counts towards a switch.

  0.4 m/s (was 0.25 in the first smoke): the lower value fired on 42 % of
  env-steps next to the dense crowd ring, so moves were not reserved for real
  threats (2026-09-10 smoke with ballet@30k driving).
  """

  trigger_steps: int = 3
  """Consecutive triggering steps before a move is entered."""

  min_alignment: float = 0.7
  """Minimum cosine between the move's travel and the escape direction."""

  speed_cap_mps: float = 1.0
  """The speed term of the score saturates here."""

  pose_distance_weight: float = 0.5
  """Weight on the RMS joint distance to the move's entry pose (rad)."""

  blend_s: float = 0.3
  """Linear blend duration at entry and at resume."""

  cooldown_s: float = 1.0
  """No new trigger for this long after a resume."""

  trigger_source: str = "teacher"
  """``teacher``: privileged CBF intervention; ``actor``: the head's planar hint."""

  def __post_init__(self) -> None:
    if self.trigger_source not in TRIGGER_SOURCES:
      raise ValueError(f"trigger_source must be one of {TRIGGER_SOURCES}")
    if self.trigger_speed_mps <= 0.0 or self.trigger_steps < 1:
      raise ValueError("trigger_speed_mps must be positive and trigger_steps >= 1")
    if self.blend_s < 0.0 or self.cooldown_s < 0.0:
      raise ValueError("blend_s and cooldown_s must be non-negative")
    if not -1.0 <= self.min_alignment <= 1.0:
      raise ValueError("min_alignment must be a cosine in [-1, 1]")


@dataclass
class EscapeMoveTable:
  """Escape candidates with GLOBAL frame indices into the flat library."""

  entry_frames: torch.Tensor
  exit_frames: torch.Tensor
  clip_starts: torch.Tensor
  clip_ends: torch.Tensor
  direction_b: torch.Tensor
  speed_mps: torch.Tensor
  entry_joint_pos: torch.Tensor

  def __len__(self) -> int:
    return int(self.entry_frames.shape[0])

  @classmethod
  def from_index(
    cls,
    index: dict,
    source_paths: Sequence[Path],
    clip_start_idx: torch.Tensor,
    clip_num_frames: torch.Tensor,
    joint_pos: torch.Tensor,
    device: str | torch.device,
  ) -> EscapeMoveTable:
    """Map index rows to library clips by file name; keep candidates only."""
    by_name = {Path(p).name: i for i, p in enumerate(source_paths)}
    starts = clip_start_idx.to("cpu", torch.long)
    frames = clip_num_frames.to("cpu", torch.long)
    entry, exit_, cstart, cend, direction, speed = [], [], [], [], [], []
    for row in index["clips"]:
      name = Path(row["file"]).name
      if name not in by_name:
        raise ValueError(f"escape index row {row['file']!r} is not in the library")
      if not row["candidate"]:
        continue
      clip = by_name[name]
      start = int(starts[clip])
      end = start + int(frames[clip])
      entry.append(start + int(row["entry_frame"]))
      exit_.append(min(start + int(row["exit_frame"]), end - 1))
      cstart.append(start)
      cend.append(end)
      direction.append([float(row["direction_b"][0]), float(row["direction_b"][1])])
      speed.append(float(row["speed_mps"]))
    entry_t = torch.tensor(entry, dtype=torch.long)
    return cls(
      entry_frames=entry_t.to(device),
      exit_frames=torch.tensor(exit_, dtype=torch.long, device=device),
      clip_starts=torch.tensor(cstart, dtype=torch.long, device=device),
      clip_ends=torch.tensor(cend, dtype=torch.long, device=device),
      direction_b=torch.tensor(direction, dtype=torch.float32, device=device).reshape(
        -1, 2
      ),
      speed_mps=torch.tensor(speed, dtype=torch.float32, device=device),
      entry_joint_pos=joint_pos.detach().to("cpu")[entry_t].to(device),
    )


def body_frame_planar(vec_w: torch.Tensor, anchor_quat_w: torch.Tensor) -> torch.Tensor:
  """Rotate world-frame planar vectors by minus the quaternions' yaw."""
  w, x, y, z = anchor_quat_w.unbind(-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  c, s = torch.cos(-yaw), torch.sin(-yaw)
  return torch.stack(
    (c * vec_w[..., 0] - s * vec_w[..., 1], s * vec_w[..., 0] + c * vec_w[..., 1]),
    dim=-1,
  )


def update_trigger_count(
  count: torch.Tensor, speed: torch.Tensor, threshold: float
) -> torch.Tensor:
  """Consecutive-step counter: +1 while at/above ``threshold``, else 0."""
  return torch.where(speed >= threshold, count + 1, torch.zeros_like(count))


def select_escape_moves(
  table: EscapeMoveTable,
  escape_dir_b: torch.Tensor,
  joint_pos: torch.Tensor,
  cfg: EscapeMoveCfg,
) -> torch.Tensor:
  """Best table row per env (``-1`` when nothing qualifies).

  ``score = cos(direction, escape) * min(speed, cap) - w * RMS(entry - joints)``
  over rows with ``cos >= min_alignment``; a zero escape direction selects
  nothing.
  """
  if len(table) == 0:
    return torch.full(
      (escape_dir_b.shape[0],), -1, dtype=torch.long, device=joint_pos.device
    )
  norm = torch.linalg.vector_norm(escape_dir_b, dim=-1, keepdim=True)
  unit = escape_dir_b / norm.clamp_min(1e-9)
  cosine = unit @ table.direction_b.T  # (N, K)
  speed = table.speed_mps.clamp(max=cfg.speed_cap_mps)[None, :]
  pose = torch.sqrt(
    ((table.entry_joint_pos[None, :, :] - joint_pos[:, None, :]) ** 2).mean(dim=-1)
  )
  score = cosine * speed - cfg.pose_distance_weight * pose
  allowed = (cosine >= cfg.min_alignment) & (norm > 1e-6)
  score = torch.where(allowed, score, torch.full_like(score, -math.inf))
  best = score.argmax(dim=-1)
  return torch.where(allowed.any(dim=-1), best, torch.full_like(best, -1))


def blend_alpha(steps_left: torch.Tensor, total_steps: int) -> torch.Tensor:
  """Blend weight of the NEW frame: 0 at the switch, 1 once the blend ends."""
  total = max(1, int(total_steps))
  return (1.0 - steps_left.to(torch.float32) / total).clamp(0.0, 1.0)


def nlerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
  """Normalised linear quaternion interpolation with hemisphere alignment."""
  sign = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
  out = (1.0 - alpha) * q0 + alpha * sign * q1
  return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(1e-9)


__all__ = [
  "EscapeMoveCfg",
  "EscapeMoveTable",
  "TRIGGER_SOURCES",
  "blend_alpha",
  "body_frame_planar",
  "nlerp",
  "select_escape_moves",
  "update_trigger_count",
]
