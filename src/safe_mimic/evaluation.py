"""Shared evaluation summaries for the benchmark scripts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from safe_mimic.tasks.env_cfg import (
  PRIMARY_HUMAN_EVENT_NAME,
  SLOW_REGIME_MIN_SPAWN_RADIUS_M,
  SLOW_REGIME_SPEED_EDGES_MPS,
  SLOW_REGIME_TTC_EDGES_S,
)

_COLLISION_CAUSES = ("primary_human_collision", "crowd_collision")

# Evaluation encounter presets. "standard" is the frozen envelope every gate
# since Phase 2 used (spawn 0.75-4 m, intercept 0.5-4 s, independent draws,
# 6 s episodes). "slow" is the slow training regime replayed as a benchmark:
# the same TTC x speed bins, but with a huge base weight so the sampler is
# uniform over bins instead of re-weighting toward the policy's own failures,
# no curriculum ramp, and episodes long enough for the longest intercept.
ENCOUNTER_PRESETS: dict[str, dict[str, Any]] = {
  "standard": {
    "episode_length_s": 6.0,
    "params": {
      "encounter_sampling": "independent",
      "min_initial_spawn_radius_m": 0.75,
      "max_initial_spawn_radius_m": 4.0,
      "min_intersection_delay_s": 0.5,
      "max_intersection_delay_s": 4.0,
    },
  },
  "slow": {
    "episode_length_s": 10.0,
    "params": {
      "encounter_sampling": "ttc",
      "min_initial_spawn_radius_m": SLOW_REGIME_MIN_SPAWN_RADIUS_M,
      "max_initial_spawn_radius_m": 4.0,
      "min_intersection_delay_s": SLOW_REGIME_TTC_EDGES_S[0],
      "max_intersection_delay_s": SLOW_REGIME_TTC_EDGES_S[-1],
      "speed_bin_edges_mps": SLOW_REGIME_SPEED_EDGES_MPS,
      "ttc_bin_edges_s": SLOW_REGIME_TTC_EDGES_S,
      "hard_speed_above_mps": 10.0 * SLOW_REGIME_SPEED_EDGES_MPS[-1],
      "hard_ttc_below_s": 0.1 * SLOW_REGIME_TTC_EDGES_S[0],
      "bin_base_weight": 1.0e6,
      "curriculum_ramp_s": 1.0e-3,
    },
  },
}


def apply_encounter_preset(
  cfg: Any, preset: str, *, keep_episode_length: bool = False
) -> Any:
  """Overwrite the primary human's encounter parameters with a named preset.

  ``keep_episode_length`` applies only the encounter parameters and leaves the
  config's own episode length alone (play keeps mjlab's effectively infinite
  episode so a session only ends on a failure termination).
  """
  if preset not in ENCOUNTER_PRESETS:
    raise ValueError(
      f"unknown encounter preset {preset!r}; expected one of {tuple(ENCOUNTER_PRESETS)}"
    )
  spec = ENCOUNTER_PRESETS[preset]
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params.update(spec["params"])
  if not keep_episode_length:
    cfg.episode_length_s = spec["episode_length_s"]
  return cfg


# Quadrants of the approach bearing in the robot's heading frame: 0 deg is
# straight ahead, +90 deg is the robot's left, +-180 deg is behind.
BEARING_REGIONS = ("front", "left", "back", "right")


def bearing_deg(
  robot_xy_w: torch.Tensor, robot_quat_wxyz: torch.Tensor, target_xy_w: torch.Tensor
) -> torch.Tensor:
  """Signed bearing of ``target`` from the robot, in the robot's heading frame.

  Heading is the yaw of the root quaternion (its body x-axis projected onto
  the ground). Positive angles are to the robot's left, ``+-180`` is behind.
  Returned in degrees, shape ``(N,)``.
  """
  w, x, y, z = robot_quat_wxyz.unbind(-1)
  # Yaw of the body x-axis: R[1,0] / R[0,0] of the rotation matrix.
  heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  delta = target_xy_w - robot_xy_w
  world_bearing = torch.atan2(delta[..., 1], delta[..., 0])
  relative = torch.atan2(
    torch.sin(world_bearing - heading), torch.cos(world_bearing - heading)
  )
  return torch.rad2deg(relative)


def bearing_region(bearing: float) -> str:
  """Map a bearing in degrees onto one of :data:`BEARING_REGIONS`."""
  angle = (bearing + 180.0) % 360.0 - 180.0  # wrap into [-180, 180)
  if -45.0 < angle < 45.0:
    return "front"
  if 45.0 <= angle < 135.0:
    return "left"
  if -135.0 < angle <= -45.0:
    return "right"
  return "back"


def fair_regime_summary(
  cases: Iterable[Mapping[str, Any]],
  *,
  max_speed_mps: float,
  min_clearance_m: float,
) -> dict[str, Any]:
  """Summarise the encounters a ~0.6 m/s robot can physically win.

  Selects episodes whose nominal approach speed is at most ``max_speed_mps``
  AND whose human starts at least ``min_clearance_m`` of surface clearance
  away. Every non-timeout ending counts as a failure: the joint@16k gate
  showed the wrist-height termination ending doomed episodes before the
  collision could register, which flattered collision-only rates (leash@16k
  read 14.7 % collisions but 27.3 % failures in this regime).
  """
  selected = [
    case
    for case in cases
    if case["nominal_approach_speed_mps"] <= max_speed_mps
    and case["initial_clearance_m"] >= min_clearance_m
  ]
  episodes = len(selected)
  collisions = sum(1 for c in selected if c["termination_cause"] in _COLLISION_CAUSES)
  ee_trips = sum(1 for c in selected if c["termination_cause"] == "ee_body_pos")
  timeouts = sum(1 for c in selected if c["termination_cause"] == "time_out")
  other = episodes - collisions - ee_trips - timeouts

  def rate(count: int) -> float:
    return count / episodes if episodes else math.nan

  # Safety failures are contacts and falls (collisions, anchor height/tilt);
  # the end-effector height check is a TRACKING termination: it fires on a
  # 25 cm wrist/ankle lag behind the reference and never involves contact.
  # ``failure_rate`` still counts both, for continuity with earlier gates.
  safety = collisions + other
  # Escape moves (optional per-case keys from the envelope benchmark): the
  # share of selected episodes that entered a move AND kept the planar CBF
  # correction under the resolved threshold while it played.
  with_escape = [c for c in selected if "escape_resolved" in c]
  escape_resolved = sum(1 for c in with_escape if c["escape_resolved"])
  escape_moves_mean = (
    sum(float(c.get("escape_moves", 0.0)) for c in with_escape) / len(with_escape)
    if with_escape
    else math.nan
  )
  return {
    "definition": {"max_speed_mps": max_speed_mps, "min_clearance_m": min_clearance_m},
    "escape_resolved_rate": (
      escape_resolved / len(with_escape) if with_escape else math.nan
    ),
    "escape_moves_mean": escape_moves_mean,
    "episodes": episodes,
    "collisions": collisions,
    "ee_body_pos": ee_trips,
    "other_terminations": other,
    "timeouts": timeouts,
    "safety_failures": safety,
    "tracking_terminations": ee_trips,
    "collision_rate": rate(collisions),
    "safety_failure_rate": rate(safety),
    "tracking_termination_rate": rate(ee_trips),
    "failure_rate": rate(episodes - timeouts),
    "survival_rate": rate(timeouts),
  }


def fair_regime_by_region(
  cases: Iterable[Mapping[str, Any]],
  *,
  max_speed_mps: float,
  min_clearance_m: float,
  bearing_key: str,
) -> dict[str, dict[str, Any]]:
  """:func:`fair_regime_summary` split by the approach region under ``bearing_key``.

  Episodes without a finite bearing are skipped rather than filed anywhere.
  """
  buckets: dict[str, list[Mapping[str, Any]]] = {r: [] for r in BEARING_REGIONS}
  for case in cases:
    bearing = case.get(bearing_key)
    if bearing is None or not math.isfinite(float(bearing)):
      continue
    buckets[bearing_region(float(bearing))].append(case)
  return {
    region: fair_regime_summary(
      subset, max_speed_mps=max_speed_mps, min_clearance_m=min_clearance_m
    )
    for region, subset in buckets.items()
  }


__all__ = [
  "BEARING_REGIONS",
  "ENCOUNTER_PRESETS",
  "apply_encounter_preset",
  "bearing_deg",
  "bearing_region",
  "fair_regime_by_region",
  "fair_regime_summary",
]
