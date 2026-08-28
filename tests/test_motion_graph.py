import json
from pathlib import Path

import numpy as np

from safe_mimic.motions.capsule_path_bank import CapsulePathBank
from safe_mimic.motions.human_capsules import SOMA_CAPSULE_SPECS
from safe_mimic.motions.motion_graph import (
  extract_transition_features,
  nearest_transition_neighbors,
)


def _write_bank(path: Path) -> CapsulePathBank:
  path.mkdir()
  path_count = 2
  frame_count = 4
  capsule_count = len(SOMA_CAPSULE_SPECS)
  names = [spec.name for spec in SOMA_CAPSULE_SPECS]
  centers = np.zeros((path_count, frame_count, capsule_count, 3), dtype=np.float32)
  roots = np.zeros((path_count, frame_count, 3), dtype=np.float32)
  roots[..., 2] = 1.0
  for path_id in range(path_count):
    for frame in range(frame_count):
      root_x = frame * 0.1 * (path_id + 1)
      roots[path_id, frame, 0] = root_x
      centers[path_id, frame, :, 0] = root_x
      centers[path_id, frame, :, 2] = 1.0
  for foot_name in ("left_foot", "right_foot"):
    centers[:, :, names.index(foot_name), 2] = 0.03
  centers[:, :, names.index("left_hand"), 0] += np.linspace(0.2, 0.7, frame_count)
  centers[:, :, names.index("right_hand"), 0] += np.linspace(0.2, 0.7, frame_count)
  quaternions = np.zeros((path_count, frame_count, capsule_count, 4), dtype=np.float32)
  quaternions[..., 0] = 1.0
  np.save(path / "centers.npy", centers)
  np.save(path / "quaternions.npy", quaternions)
  np.save(path / "root_positions.npy", roots)
  np.save(
    path / "facing_yaw.npy",
    np.zeros((path_count, frame_count), dtype=np.float32),
  )
  np.save(
    path / "radii.npy",
    np.full((path_count, capsule_count), 0.1, dtype=np.float32),
  )
  np.save(
    path / "half_lengths.npy",
    np.full((path_count, capsule_count), 0.2, dtype=np.float32),
  )
  np.save(path / "frame_counts.npy", np.full(path_count, frame_count, dtype=np.int32))
  np.save(path / "ground_z.npy", np.zeros(path_count, dtype=np.float32))
  (path / "bank.json").write_text(
    json.dumps(
      {
        "format_version": 1,
        "complete": True,
        "fps": 10.0,
        "max_frames": frame_count,
        "path_count": path_count,
        "capsule_names": names,
      }
    )
  )
  return CapsulePathBank(path)


def test_extract_transition_features_and_activity_metrics(tmp_path: Path) -> None:
  features = extract_transition_features(_write_bank(tmp_path / "bank"))

  assert features.entry.shape == (2, 32)
  assert features.exit.shape == (2, 32)
  assert np.all(features.upright_boundary)
  assert np.all(features.supported_boundary)
  assert np.allclose(features.root_path_length_m, (0.3, 0.6))
  assert np.all(features.hand_excursion_m >= 0.5)


def test_nearest_transition_neighbors_orders_costs() -> None:
  query = np.asarray([[0.9, 0.0], [2.1, 0.0]], dtype=np.float32)
  reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

  ids, costs = nearest_transition_neighbors(query, reference, 2)

  assert ids.tolist() == [[1, 0], [2, 1]]
  assert np.all(costs[:, 0] <= costs[:, 1])
