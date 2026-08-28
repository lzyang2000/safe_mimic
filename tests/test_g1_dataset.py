import csv
from pathlib import Path

import numpy as np

from safe_mimic.motions.g1_dataset import (
  G1MotionFilterCfg,
  assign_split,
  balanced_package_sample,
  filter_g1_motion,
  is_dance_motion,
  is_jump_like_motion,
  is_stationary_jump_motion,
  metadata_rejection_reason,
)


def _write_motion(path: Path, *, root_z_cm: float = 80.0) -> None:
  frame_count = 121
  raw = np.zeros((frame_count, 36), dtype=np.float64)
  raw[:, 0] = np.arange(frame_count)
  raw[:, 3] = root_z_cm
  header = ["Frame", "root_tX", "root_tY", "root_tZ"]
  header += ["root_rX", "root_rY", "root_rZ"]
  header += [f"joint_{index}" for index in range(29)]
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(raw)


def test_dance_is_retained_and_uses_expressive_height_envelope(tmp_path: Path) -> None:
  path = tmp_path / "dance.csv"
  _write_motion(path, root_z_cm=110.0)
  metadata = {
    "move_name": "dance",
    "package": "Dances",
    "category": "Dancing",
    "is_mirror": "False",
    "content_body_position": "standing, dance",
    "content_type_of_movement": "dancing",
  }
  limits = np.full(29, 2.0)
  result = filter_g1_motion(
    path,
    metadata,
    -limits,
    limits,
    G1MotionFilterCfg(),
  )

  assert is_dance_motion(metadata)
  assert metadata_rejection_reason(metadata) is None
  assert result.accepted
  assert result.quality is not None
  assert result.quality.root_height_max_m == 1.1


def test_non_dance_height_and_non_upright_metadata_are_rejected(tmp_path: Path) -> None:
  path = tmp_path / "walk.csv"
  _write_motion(path, root_z_cm=110.0)
  metadata = {
    "move_name": "walk",
    "package": "Locomotion",
    "category": "Basic Locomotion Neutral",
    "is_mirror": "False",
    "content_body_position": "standing",
    "content_type_of_movement": "walking",
  }
  limits = np.full(29, 2.0)
  result = filter_g1_motion(
    path,
    metadata,
    -limits,
    limits,
    G1MotionFilterCfg(),
  )

  assert not result.accepted
  assert result.reason is not None and "root height" in result.reason
  assert (
    metadata_rejection_reason(
      {**metadata, "content_body_position": "standing, handstanding"}
    )
    == "non-upright metadata: handstand"
  )


def test_mirrors_are_deferred_to_online_augmentation() -> None:
  assert metadata_rejection_reason({"is_mirror": "True"}) == (
    "mirrored duplicate (mirror online)"
  )


def test_transition_text_is_not_mistaken_for_sitting() -> None:
  assert metadata_rejection_reason(
    {
      "is_mirror": "False",
      "package": "Locomotion",
      "category": "Basic Locomotion Neutral",
      "content_body_position": "standing",
      "content_type_of_movement": "transition",
    }
  ) is None


def test_stationary_jump_filter_combines_semantics_and_planar_excursion() -> None:
  stationary = np.array([[0.0, 0.0, 0.8], [0.2, 0.0, 1.0]])
  traveling = np.array([[0.0, 0.0, 0.8], [0.8, 0.0, 1.0]])

  assert is_jump_like_motion({"description": "a small jump"})
  assert is_jump_like_motion({"description": "hopping on one leg"})
  assert not is_jump_like_motion({"description": "a hip-hop power step"})
  assert is_stationary_jump_motion(
    {"description": "a small jump"}, stationary
  )
  assert not is_stationary_jump_motion(
    {"description": "a forward jump"}, traveling
  )
  assert is_stationary_jump_motion(
    {"description": "leaping in place"}, traveling
  )
  assert not is_stationary_jump_motion(
    {"description": "standing still"}, stationary
  )


def test_balanced_sample_caps_each_package_and_is_deterministic() -> None:
  records = [
    {
      "move_name": f"loc_{index}",
      "package": "Locomotion",
      "category": "Walk" if index % 2 else "Jump",
    }
    for index in range(10)
  ] + [
    {
      "move_name": f"dance_{index}",
      "package": "Dances",
      "category": "Dancing",
    }
    for index in range(3)
  ]

  first = balanced_package_sample(records, max_per_package=4)
  second = balanced_package_sample(records, max_per_package=4)

  assert first == second
  assert sum(row["package"] == "Locomotion" for row in first) == 4
  assert sum(row["package"] == "Dances" for row in first) == 3
  assert {row["category"] for row in first if row["package"] == "Locomotion"} == {
    "Jump",
    "Walk",
  }


def test_split_is_grouped_by_take() -> None:
  first = assign_split({"take_name": "take_001"}, 0.1)
  second = assign_split(
    {"take_name": "take_001", "move_name": "different_segment"}, 0.1
  )
  assert first == second
