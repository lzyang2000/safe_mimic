"""Approach bearing in the robot's heading frame and per-region fair-regime split."""

import math

import torch

from safe_mimic.evaluation import (
  BEARING_REGIONS,
  bearing_deg,
  bearing_region,
  fair_regime_by_region,
)


def _yaw_quat(yaw: float) -> torch.Tensor:
  return torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])


def test_bearing_is_zero_straight_ahead_and_positive_to_the_left() -> None:
  robot = torch.tensor([[0.0, 0.0]])
  ahead = bearing_deg(robot, _yaw_quat(0.0), torch.tensor([[2.0, 0.0]]))
  left = bearing_deg(robot, _yaw_quat(0.0), torch.tensor([[0.0, 2.0]]))
  right = bearing_deg(robot, _yaw_quat(0.0), torch.tensor([[0.0, -2.0]]))
  back = bearing_deg(robot, _yaw_quat(0.0), torch.tensor([[-2.0, 0.0]]))
  torch.testing.assert_close(ahead, torch.tensor([0.0]))
  torch.testing.assert_close(left, torch.tensor([90.0]))
  torch.testing.assert_close(right, torch.tensor([-90.0]))
  torch.testing.assert_close(back.abs(), torch.tensor([180.0]))


def test_bearing_follows_the_robot_heading_not_the_world_axes() -> None:
  # Robot faces +y (yaw 90 deg); a human at +y world is straight ahead.
  robot = torch.tensor([[1.0, 1.0]])
  ahead = bearing_deg(robot, _yaw_quat(math.pi / 2), torch.tensor([[1.0, 3.0]]))
  torch.testing.assert_close(ahead, torch.tensor([0.0]), atol=1e-4, rtol=0)
  # A human at +x world is now on the robot's RIGHT.
  right = bearing_deg(robot, _yaw_quat(math.pi / 2), torch.tensor([[3.0, 1.0]]))
  torch.testing.assert_close(right, torch.tensor([-90.0]), atol=1e-4, rtol=0)


def test_regions_partition_the_circle() -> None:
  assert BEARING_REGIONS == ("front", "left", "back", "right")
  assert bearing_region(0.0) == "front"
  assert bearing_region(44.9) == "front" and bearing_region(-44.9) == "front"
  assert bearing_region(45.0) == "left" and bearing_region(134.9) == "left"
  assert bearing_region(135.0) == "back" and bearing_region(-135.0) == "back"
  assert bearing_region(180.0) == "back" and bearing_region(-180.0) == "back"
  assert bearing_region(-45.0) == "right" and bearing_region(-134.9) == "right"


def _case(speed, clearance, cause, bearing):
  return {
    "nominal_approach_speed_mps": speed,
    "initial_clearance_m": clearance,
    "termination_cause": cause,
    "initial_bearing_deg": bearing,
  }


def test_fair_regime_split_by_region_counts_only_fair_episodes() -> None:
  cases = [
    _case(0.5, 1.0, "time_out", 10.0),
    _case(0.5, 1.0, "primary_human_collision", -20.0),
    _case(0.5, 1.0, "ee_body_pos", 170.0),
    _case(0.5, 1.0, "time_out", 90.0),
    _case(2.0, 1.0, "primary_human_collision", 0.0),  # too fast: excluded
    _case(0.5, 0.2, "primary_human_collision", 0.0),  # inside reach: excluded
  ]
  by = fair_regime_by_region(
    cases, max_speed_mps=0.75, min_clearance_m=0.8, bearing_key="initial_bearing_deg"
  )
  assert set(by) == set(BEARING_REGIONS)
  assert by["front"]["episodes"] == 2 and by["front"]["failure_rate"] == 0.5
  assert by["back"]["episodes"] == 1 and by["back"]["ee_body_pos"] == 1
  assert by["left"]["episodes"] == 1 and by["left"]["survival_rate"] == 1.0
  assert by["right"]["episodes"] == 0 and math.isnan(by["right"]["failure_rate"])


def test_missing_bearing_is_skipped_not_misfiled() -> None:
  cases = [_case(0.5, 1.0, "time_out", None), _case(0.5, 1.0, "time_out", 0.0)]
  by = fair_regime_by_region(
    cases, max_speed_mps=0.75, min_clearance_m=0.8, bearing_key="initial_bearing_deg"
  )
  assert sum(r["episodes"] for r in by.values()) == 1
