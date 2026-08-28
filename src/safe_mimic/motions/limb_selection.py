"""Kinematic filters for broad human arm- and leg-extension datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from safe_mimic.motions.soma_bvh import BvhMotion

PLAIN_WALK_DESCRIPTIONS = {
  "walking facing forward",
  "walk backwards",
  "walking sideways to the right",
  "walking sideways to the left",
  "walking clockwise in an arc",
  "walking counterclockwise in an arc, leftside",
  "walk front right diagonal",
  "advancing with a leftward diagonal stride",
  "neutral casual random walk",
  "strolling forward",
}

_BANNED_MOTION_TEXT = re.compile(
  r"\b(?:handstands?|cartwheels?|somersaults?|backflips?|frontflips?|"
  r"flips?|crawls?|crawling|all fours|injured|injury|dances?|dancing)\b",
  re.IGNORECASE,
)


@dataclass(frozen=True)
class LimbMotionMetrics:
  """Sparse-frame arm and leg extension measurements for one motion."""

  arm_peak_horizontal_m: float
  arm_excursion_m: float
  has_sustained_overhead_arm: bool
  leg_peak_horizontal_m: float
  leg_excursion_m: float


def _extension_metric(
  positions: np.ndarray,
  indices: dict[str, int],
  sides: tuple[tuple[str, str, str], ...],
) -> tuple[float, float]:
  peaks: list[float] = []
  excursions: list[float] = []
  for start_name, middle_name, end_name in sides:
    start = positions[:, indices[start_name]]
    middle = positions[:, indices[middle_name]]
    end = positions[:, indices[end_name]]
    horizontal = np.linalg.norm((end - start)[:, (0, 2)], axis=1)
    reach = np.linalg.norm(end - start, axis=1)
    limb_length = np.linalg.norm(middle - start, axis=1) + np.linalg.norm(
      end - middle, axis=1
    )
    straightness = reach / np.maximum(limb_length, 1e-9)
    valid = horizontal[straightness > 0.85]
    peaks.append(float(valid.max()) if len(valid) else 0.0)
    excursions.append(
      float(np.percentile(horizontal, 95) - np.percentile(horizontal, 20))
    )
  return max(peaks), max(excursions)


def measure_limb_motion(motion: BvhMotion) -> LimbMotionMetrics:
  """Measure outward limb motion and sustained overhead hand placement."""

  indices = {name: index for index, name in enumerate(motion.joint_names)}
  arm_peak, arm_excursion = _extension_metric(
    motion.positions_m,
    indices,
    (
      ("LeftArm", "LeftForeArm", "LeftHand"),
      ("RightArm", "RightForeArm", "RightHand"),
    ),
  )
  leg_peak, leg_excursion = _extension_metric(
    motion.positions_m,
    indices,
    (
      ("LeftLeg", "LeftShin", "LeftFoot"),
      ("RightLeg", "RightShin", "RightFoot"),
    ),
  )

  head_height = motion.positions_m[:, indices["Head"], 1]
  hand_heights = np.stack(
    (
      motion.positions_m[:, indices["LeftHand"], 1],
      motion.positions_m[:, indices["RightHand"], 1],
    ),
    axis=1,
  )
  above_head = hand_heights - head_height[:, None]
  overhead_fraction = float((above_head > 0.02).any(axis=1).mean())
  sustained_overhead = bool(above_head.max() > 0.25 and overhead_fraction > 0.18)

  return LimbMotionMetrics(
    arm_peak_horizontal_m=arm_peak,
    arm_excursion_m=arm_excursion,
    has_sustained_overhead_arm=sustained_overhead,
    leg_peak_horizontal_m=leg_peak,
    leg_excursion_m=leg_excursion,
  )


def qualifies_arm_extension(metrics: LimbMotionMetrics) -> bool:
  """Select forward/lateral arm actions while rejecting overhead lifts."""

  return (
    metrics.arm_peak_horizontal_m >= 0.42
    and metrics.arm_excursion_m >= 0.15
    and not metrics.has_sustained_overhead_arm
  )


def qualifies_leg_extension(metrics: LimbMotionMetrics) -> bool:
  """Select broad kick-, step-, jump-, and leg-swing-like actions."""

  return metrics.leg_peak_horizontal_m >= 0.45 and metrics.leg_excursion_m >= 0.15


def is_plain_walking(row: Mapping[str, str]) -> bool:
  """Return whether metadata describes one of the accepted plain walks."""

  return (
    row["package"] == "Locomotion"
    and row["category"] == "Basic Locomotion Neutral"
    and row["content_type_of_movement"] in {"walking", "walking, turning"}
    and row["content_props"] == "0"
    and row["content_uniform_style"] == "neutral"
    and row["content_short_description"] in PLAIN_WALK_DESCRIPTIONS
  )


def is_clean_motion_metadata(row: Mapping[str, str]) -> bool:
  """Reject motion classes previously identified as unsuitable."""

  if row["package"] == "Dances" or row["category"] in {
    "Stunts",
    "Basic Locomotion Styles",
  }:
    return False
  if row["content_uniform_style"] != "neutral":
    return False
  text = " ".join(
    row.get(field, "")
    for field in (
      "content_short_description",
      "content_natural_desc_1",
      "content_technical_description",
      "content_body_position",
    )
  )
  return _BANNED_MOTION_TEXT.search(text) is None
