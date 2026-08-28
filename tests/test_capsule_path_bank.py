import json
from pathlib import Path

import numpy as np
import torch

from safe_mimic.motions.capsule_path_bank import (
  CapsulePathBank,
  OnlineCapsulePathSampler,
)


def _write_tiny_bank(path: Path) -> None:
  path.mkdir()
  centers = np.zeros((1, 3, 2, 3), dtype=np.float32)
  root_positions = np.zeros((1, 3, 3), dtype=np.float32)
  for frame in range(3):
    centers[0, frame, :, 0] = frame
    centers[0, frame, 1, 2] = 1.0
    root_positions[0, frame, 0] = frame
  quaternions = np.zeros((1, 3, 2, 4), dtype=np.float32)
  quaternions[..., 0] = 1.0
  np.save(path / "centers.npy", centers)
  np.save(path / "quaternions.npy", quaternions)
  np.save(path / "root_positions.npy", root_positions)
  np.save(path / "facing_yaw.npy", np.zeros((1, 3), dtype=np.float32))
  np.save(path / "radii.npy", np.full((1, 2), 0.1, dtype=np.float32))
  np.save(path / "half_lengths.npy", np.full((1, 2), 0.2, dtype=np.float32))
  np.save(path / "frame_counts.npy", np.asarray([3], dtype=np.int32))
  np.save(path / "ground_z.npy", np.asarray([0.0], dtype=np.float32))
  config = {
    "format_version": 1,
    "complete": True,
    "fps": 1.0,
    "max_frames": 3,
    "path_count": 1,
    "capsule_names": ["first", "second"],
  }
  (path / "bank.json").write_text(json.dumps(config))


def test_memory_mapped_bank_interpolates_current_frames(tmp_path: Path) -> None:
  path = tmp_path / "bank"
  _write_tiny_bank(path)
  bank = CapsulePathBank(path)

  frames = bank.sample(np.asarray([0]), np.asarray([0.5]))

  assert isinstance(bank.centers, np.memmap)
  assert np.allclose(frames.root_positions_m[0], (0.5, 0.0, 0.0))
  assert np.allclose(frames.centers_m[0, :, 0], (0.5, 0.5))
  assert frames.active.tolist() == [True]


def test_online_sampler_places_exact_intersection_and_hides_inactive(
  tmp_path: Path,
) -> None:
  path = tmp_path / "bank"
  _write_tiny_bank(path)
  sampler = OnlineCapsulePathSampler(CapsulePathBank(path), 1, "cpu")
  sampler.schedule_intersections(
    np.asarray([0]),
    path_ids=np.asarray([0]),
    global_intersection_times_s=np.asarray([10.0]),
    robot_positions_at_intersection_w=np.asarray([[10.0, 5.0, 0.8]]),
    robot_yaw_at_intersection=np.asarray([0.0]),
    robot_path_heading_at_intersection=np.asarray([0.0]),
    intersection_phase=0.5,
    crossing_angle_rad=np.pi / 2.0,
  )

  at_intersection = sampler.sample(10.0)
  assert at_intersection.active.tolist() == [True]
  assert torch.allclose(
    at_intersection.root_positions_w[0, :2], torch.tensor([10.0, 5.0])
  )
  assert at_intersection.centers_w.shape == (1, 2, 3)

  before_clip = sampler.sample(8.0)
  assert before_clip.active.tolist() == [False]
  assert torch.all(before_clip.centers_w[..., 2] == -100.0)


def test_online_sampler_holds_between_updates(tmp_path: Path) -> None:
  path = tmp_path / "bank"
  _write_tiny_bank(path)
  sampler = OnlineCapsulePathSampler(
    CapsulePathBank(path), 1, "cpu", update_hz=10.0
  )
  sampler.schedule_intersections(
    np.asarray([0]),
    path_ids=np.asarray([0]),
    global_intersection_times_s=np.asarray([1.0]),
    robot_positions_at_intersection_w=np.zeros((1, 3)),
    robot_yaw_at_intersection=np.zeros(1),
    robot_path_heading_at_intersection=np.zeros(1),
  )

  first = sampler.sample_held(1.0).centers_w.clone()
  held = sampler.sample_held(1.05).centers_w.clone()
  updated = sampler.sample_held(1.1).centers_w.clone()

  assert torch.equal(first, held)
  assert not torch.equal(held, updated)


def test_compact_resident_bank_schedules_and_interpolates_on_device(
  tmp_path: Path,
) -> None:
  path = tmp_path / "bank"
  _write_tiny_bank(path)
  sampler = OnlineCapsulePathSampler(
    CapsulePathBank(path),
    1,
    "cpu",
    resident_path_ids=np.asarray([0]),
    preload_to_device=True,
  )
  sampler.schedule_intersections(
    torch.tensor([0]),
    path_ids=torch.tensor([0]),
    global_intersection_times_s=torch.tensor([4.0]),
    robot_positions_at_intersection_w=torch.tensor([[3.0, 2.0, 0.0]]),
    robot_yaw_at_intersection=torch.zeros(1),
    robot_path_heading_at_intersection=torch.zeros(1),
    intersection_phase=torch.tensor([0.5]),
  )

  poses = sampler.sample(4.0)

  assert sampler.device_storage_bytes > 0
  assert sampler.resident_path_ids.tolist() == [0]
  assert poses.active.tolist() == [True]
  assert torch.allclose(poses.root_positions_w[0, :2], torch.tensor([3.0, 2.0]))
