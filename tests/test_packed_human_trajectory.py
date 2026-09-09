import json
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mimic.motions.packed_human_trajectory import (
  PackedHumanTrajectoryBank,
  PackedHumanTrajectorySampler,
)


def _write_test_bank(path: Path) -> PackedHumanTrajectoryBank:
  keypoint_names = (
    "Head",
    "Hips",
    "LeftFoot",
    "LeftToeBase",
    "RightFoot",
    "RightToeBase",
  )
  keypoints = np.zeros((2, 4, len(keypoint_names), 3), dtype=np.float16)
  hips = keypoint_names.index("Hips")
  head = keypoint_names.index("Head")
  for path_id, stride in enumerate((1.0, 2.0)):
    root_x = np.arange(4, dtype=np.float32) * stride
    keypoints[path_id, :, :, 0] = root_x[:, None]
    keypoints[path_id, :, head, 2] = 1.0
    keypoints[path_id, :, hips, 2] = 0.0
  path.mkdir()
  np.save(path / "keypoints.npy", keypoints)
  np.save(path / "facing_yaw.npy", np.zeros((2, 4), dtype=np.float16))
  np.save(path / "frame_counts.npy", np.asarray((4, 4), dtype=np.int32))
  np.save(path / "action_frames.npy", np.asarray((2, 3), dtype=np.int32))
  np.save(path / "entry_travel_m.npy", np.asarray((2.0, 6.0), dtype=np.float32))
  np.save(
    path / "sequence_path_ids.npy",
    np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
  )
  config = {
    "format_version": 1,
    "complete": True,
    "kind": "primary",
    "fps": 1.0,
    "path_count": 2,
    "max_frames": 4,
    "keypoint_names": list(keypoint_names),
    "root_keypoint": "Hips",
    "foot_keypoints": [
      "LeftFoot",
      "LeftToeBase",
      "RightFoot",
      "RightToeBase",
    ],
    "capsules": [
      {
        "name": "body",
        "start_keypoint": "Hips",
        "end_keypoint": "Head",
        "radius_m": 0.1,
      }
    ],
  }
  (path / "bank.json").write_text(json.dumps(config))
  return PackedHumanTrajectoryBank(path)


def test_packed_bank_validates_and_loads_entry_travel(tmp_path: Path) -> None:
  bank = _write_test_bank(tmp_path / "bank")

  assert len(bank) == 2
  np.testing.assert_allclose(bank.entry_travel_m, (2.0, 6.0))


def test_entry_time_selects_closest_pre_action_distance(tmp_path: Path) -> None:
  sampler = PackedHumanTrajectorySampler(
    _write_test_bank(tmp_path / "bank"),
    2,
    "cpu",
    update_hz=10.0,
  )

  entry_times = sampler.entry_times_for_distance(
    torch.tensor((0, 1)),
    torch.tensor((1.0, 4.0)),
  )

  torch.testing.assert_close(entry_times, torch.tensor((1.0, 1.0)))


def test_intersection_placement_uses_natural_entry_displacement(
  tmp_path: Path,
) -> None:
  sampler = PackedHumanTrajectorySampler(
    _write_test_bank(tmp_path / "bank"),
    1,
    "cpu",
    update_hz=10.0,
  )
  sampler.schedule_intersections(
    torch.tensor((0,)),
    path_ids=torch.tensor((0,)),
    global_intersection_times_s=torch.tensor((2.0,)),
    local_intersection_times_s=torch.tensor((2.0,)),
    target_positions_w=torch.tensor(((10.0, 5.0, 0.0),)),
    target_heading=torch.tensor((torch.pi / 2,)),
    crossing_angle_rad=0.0,
    ground_height_m=torch.tensor((0.0,)),
    body_scale_xyz=torch.ones((1, 3)),
    radius_scale=torch.ones(1),
    radius_margin_m=torch.zeros(1),
    playback_speed=torch.ones(1),
    heading_start_times_s=torch.zeros(1),
  )

  sampler.sample_held(0.0, force_ids=torch.tensor((0,)))
  torch.testing.assert_close(
    sampler._poses.root_positions_w[0],
    torch.tensor((10.0, 3.0, 0.0)),
    atol=1.0e-5,
    rtol=0.0,
  )
  sampler.sample_held(2.0, force_ids=torch.tensor((0,)))
  torch.testing.assert_close(
    sampler._poses.root_positions_w[0],
    torch.tensor((10.0, 5.0, 0.0)),
    atol=1.0e-5,
    rtol=0.0,
  )


def test_held_updates_and_deactivation(tmp_path: Path) -> None:
  sampler = PackedHumanTrajectorySampler(
    _write_test_bank(tmp_path / "bank"),
    1,
    "cpu",
    update_hz=2.0,
  )
  sampler.schedule_intersections(
    torch.tensor((0,)),
    path_ids=torch.tensor((0,)),
    global_intersection_times_s=torch.zeros(1),
    local_intersection_times_s=torch.zeros(1),
    target_positions_w=torch.zeros((1, 3)),
    target_heading=torch.zeros(1),
    crossing_angle_rad=0.0,
    ground_height_m=torch.zeros(1),
    body_scale_xyz=torch.ones((1, 3)),
    radius_scale=torch.ones(1),
    radius_margin_m=torch.zeros(1),
    playback_speed=torch.ones(1),
  )
  sampler.sample_held(0.0)
  first_root = sampler._poses.root_positions_w.clone()

  sampler.sample_held(0.2)
  assert sampler.last_updated_ids.numel() == 0
  torch.testing.assert_close(sampler._poses.root_positions_w, first_root)

  sampler.sample_held(0.5)
  assert sampler.last_updated_ids.tolist() == [0]
  assert sampler._poses.root_positions_w[0, 0] == pytest.approx(0.5)

  sampler.deactivate(torch.tensor((0,)))
  assert not sampler._poses.active[0]
  assert sampler._poses.centers_w[0, 0, 2] == sampler.inactive_height_m


def test_looping_ping_pongs_without_endpoint_teleport(tmp_path: Path) -> None:
  sampler = PackedHumanTrajectorySampler(
    _write_test_bank(tmp_path / "bank"),
    1,
    "cpu",
    update_hz=10.0,
    loop=True,
  )
  path_ids = torch.zeros(5, dtype=torch.long)
  times = torch.tensor((2.9, 3.0, 3.1, 5.9, 6.0))

  points = sampler._sample_keypoints(path_ids, times)[0]
  roots_x = points[:, sampler.root_index, 0]

  torch.testing.assert_close(
    roots_x,
    torch.tensor((2.9, 3.0, 2.9, 0.1, 0.0)),
    atol=2.0e-3,
    rtol=0.0,
  )
