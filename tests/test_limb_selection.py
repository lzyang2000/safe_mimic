from types import SimpleNamespace

import numpy as np

from safe_mimic.motions.limb_selection import (
  LimbMotionMetrics,
  is_clean_motion_metadata,
  is_plain_walking,
  measure_limb_motion,
  qualifies_arm_extension,
  qualifies_leg_extension,
)


def test_limb_extension_metrics_detect_horizontal_motion() -> None:
  names = (
    "Head",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "RightLeg",
    "RightShin",
    "RightFoot",
  )
  positions = np.zeros((5, len(names), 3), dtype=np.float64)
  indices = {name: index for index, name in enumerate(names)}
  positions[:, indices["Head"], 1] = 1.7
  horizontal = np.linspace(0.0, 0.8, 5)
  for start, middle, end, height in (
    ("LeftArm", "LeftForeArm", "LeftHand", 1.4),
    ("RightArm", "RightForeArm", "RightHand", 1.4),
    ("LeftLeg", "LeftShin", "LeftFoot", 0.9),
    ("RightLeg", "RightShin", "RightFoot", 0.9),
  ):
    positions[:, indices[start], 1] = height
    positions[:, indices[middle], 1] = height
    positions[:, indices[end], 1] = height
    positions[:, indices[middle], 2] = 0.5 * horizontal
    positions[:, indices[end], 2] = horizontal
  motion = SimpleNamespace(joint_names=names, positions_m=positions)

  metrics = measure_limb_motion(motion)

  assert qualifies_arm_extension(metrics)
  assert qualifies_leg_extension(metrics)
  assert not metrics.has_sustained_overhead_arm


def test_arm_extension_rejects_sustained_overhead_motion() -> None:
  metrics = LimbMotionMetrics(0.8, 0.4, True, 0.8, 0.4)

  assert not qualifies_arm_extension(metrics)
  assert qualifies_leg_extension(metrics)


def test_metadata_filters_reject_stylized_motion() -> None:
  row = {
    "package": "Locomotion",
    "category": "Basic Locomotion Neutral",
    "content_type_of_movement": "walking",
    "content_props": "0",
    "content_uniform_style": "neutral",
    "content_short_description": "walking facing forward",
    "content_natural_desc_1": "ordinary walking",
    "content_technical_description": "walk forward",
    "content_body_position": "standing",
  }

  assert is_plain_walking(row)
  assert is_clean_motion_metadata(row)

  row["content_short_description"] = "cartwheel forward"
  assert not is_plain_walking(row)
  assert not is_clean_motion_metadata(row)
