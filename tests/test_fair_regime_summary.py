"""Fair-regime summary.

Slow approach, human starting outside reach, and every non-timeout counted as
a failure (wrist-height trips included).
"""

from safe_mimic.evaluation import fair_regime_summary


def _case(speed, clearance, cause):
  return {
    "nominal_approach_speed_mps": speed,
    "initial_clearance_m": clearance,
    "termination_cause": cause,
    "collision": cause in ("primary_human_collision", "crowd_collision"),
    "timeout": cause == "time_out",
  }


CASES = [
  _case(0.5, 1.0, "time_out"),
  _case(0.5, 1.0, "primary_human_collision"),
  _case(0.7, 0.9, "ee_body_pos"),
  _case(0.7, 0.9, "time_out"),
  _case(0.5, 0.3, "primary_human_collision"),  # inside reach: excluded
  _case(1.2, 1.5, "primary_human_collision"),  # too fast: excluded
  _case(0.6, 1.2, "anchor_ori"),
]


def test_selects_only_slow_encounters_that_start_outside_reach() -> None:
  s = fair_regime_summary(CASES, max_speed_mps=0.75, min_clearance_m=0.8)
  assert s["episodes"] == 5


def test_counts_every_non_timeout_as_a_failure() -> None:
  s = fair_regime_summary(CASES, max_speed_mps=0.75, min_clearance_m=0.8)
  assert s["collisions"] == 1
  assert s["ee_body_pos"] == 1
  assert s["other_terminations"] == 1
  assert s["timeouts"] == 2
  assert s["survival_rate"] == 2 / 5
  assert s["failure_rate"] == 3 / 5
  assert s["collision_rate"] == 1 / 5


def test_empty_selection_reports_nan_rates() -> None:
  import math

  s = fair_regime_summary(CASES, max_speed_mps=0.1, min_clearance_m=0.8)
  assert s["episodes"] == 0
  assert math.isnan(s["failure_rate"])


def test_records_its_own_definition() -> None:
  s = fair_regime_summary(CASES, max_speed_mps=0.75, min_clearance_m=0.8)
  assert s["definition"] == {"max_speed_mps": 0.75, "min_clearance_m": 0.8}


def test_splits_safety_failures_from_tracking_terminations() -> None:
  """Collisions and falls are safety failures; wrist/ankle height trips are
  tracking terminations. ``failure_rate`` keeps counting both for continuity."""
  s = fair_regime_summary(CASES, max_speed_mps=0.75, min_clearance_m=0.8)
  # Selected: time_out, collision, ee_body_pos, time_out, anchor_ori  (n=5)
  assert s["safety_failures"] == 2  # collision + anchor_ori fall
  assert s["tracking_terminations"] == 1
  assert s["safety_failure_rate"] == 2 / 5
  assert s["tracking_termination_rate"] == 1 / 5
  assert s["failure_rate"] == 3 / 5


def test_escape_resolved_rate_is_nan_without_escape_keys() -> None:
  s = fair_regime_summary(CASES, max_speed_mps=0.75, min_clearance_m=0.8)
  import math as _math

  assert _math.isnan(s["escape_resolved_rate"])
  assert _math.isnan(s["escape_moves_mean"])


def test_escape_resolved_rate_counts_selected_cases_only() -> None:
  cases = [
    dict(c, escape_moves=float(k % 2), escape_resolved=(k % 2 == 1))
    for k, c in enumerate(CASES)
  ]
  s = fair_regime_summary(cases, max_speed_mps=0.75, min_clearance_m=0.8)
  selected = [
    c
    for c in cases
    if c["nominal_approach_speed_mps"] <= 0.75 and c["initial_clearance_m"] >= 0.8
  ]
  expected = sum(1 for c in selected if c["escape_resolved"]) / len(selected)
  assert s["escape_resolved_rate"] == expected
  assert s["escape_moves_mean"] == sum(c["escape_moves"] for c in selected) / len(
    selected
  )
