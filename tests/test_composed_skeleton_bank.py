import json
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_mimic.motions.composed_skeleton_bank import (
  OnlineComposedHumanSampler,
  SkeletonPathBank,
)
from safe_mimic.motions.human_capsules import SOMA_CROWD_PROXY_SPECS

JOINT_NAMES = (
  "Root",
  "Hips",
  "Spine1",
  "Spine2",
  "Chest",
  "Neck1",
  "Neck2",
  "Head",
  "LeftShoulder",
  "LeftArm",
  "LeftForeArm",
  "LeftHand",
  "RightShoulder",
  "RightArm",
  "RightForeArm",
  "RightHand",
  "LeftLeg",
  "LeftShin",
  "LeftFoot",
  "LeftToeBase",
  "RightLeg",
  "RightShin",
  "RightFoot",
  "RightToeBase",
)


def _write_tiny_skeleton_bank(path: Path) -> None:
  path.mkdir()
  path_count, frames, joints = 3, 3, len(JOINT_NAMES)
  positions = np.zeros((path_count, frames, joints, 3), dtype=np.float16)
  quaternions = np.zeros((path_count, frames, joints, 4), dtype=np.float16)
  quaternions[..., 0] = 1.0
  parents = np.asarray(
    (-1, 0, 1, 2, 3, 4, 5, 6, 4, 8, 9, 10, 4, 12, 13, 14, 1, 16, 17, 18, 1, 20, 21, 22),
    dtype=np.int64,
  )
  for path_id in range(path_count):
    positions[path_id, :, 0, 0] = path_id * 5.0 + np.arange(frames) * 0.1
  for joint in range(1, joints):
    positions[:, :, joint, 2] = 0.08
  positions[:, :, JOINT_NAMES.index("LeftLeg"), 1] = 0.10
  positions[:, :, JOINT_NAMES.index("RightLeg"), 1] = -0.10
  np.save(path / "local_positions.npy", positions)
  np.save(path / "local_quaternions.npy", quaternions)
  np.save(path / "frame_counts.npy", np.full(path_count, frames, dtype=np.int32))
  np.save(path / "ground_z.npy", np.zeros(path_count, dtype=np.float32))
  np.save(path / "source_path_ids.npy", np.asarray([10, 20, 30], dtype=np.int64))
  np.save(path / "parents.npy", parents)
  config = {
    "format_version": 1,
    "complete": True,
    "fps": 10.0,
    "max_frames": frames,
    "path_count": path_count,
    "source_path_count": 31,
    "joint_names": list(JOINT_NAMES),
  }
  (path / "bank.json").write_text(json.dumps(config))


def test_online_skeleton_composer_aligns_boundaries_and_intersection(
  tmp_path: Path,
) -> None:
  path = tmp_path / "skeleton_bank"
  _write_tiny_skeleton_bank(path)
  sampler = OnlineComposedHumanSampler(
    SkeletonPathBank(path), 1, "cpu", retain_joint_poses=True
  )
  env_ids = torch.tensor([0])
  sampler.schedule_intersections(
    env_ids,
    sequence_source_ids=torch.tensor([[10, 20, 30]]),
    global_intersection_times_s=torch.tensor([2.0]),
    robot_positions_at_intersection_w=torch.tensor([[4.0, 3.0, 0.0]]),
    robot_yaw_at_intersection=torch.zeros(1),
    robot_path_heading_at_intersection=torch.zeros(1),
    ground_height_m=torch.zeros(1),
  )

  expected_action_midpoint = (
    sampler.segment_durations_s[0, 0]
    + 0.5 * sampler.segment_durations_s[0, 1]
  )
  assert torch.allclose(
    sampler.local_intersection_times_s[0], expected_action_midpoint
  )

  poses = sampler.sample_held(2.0)
  first_boundary = sampler.segment_durations_s[:, 0]
  before = sampler._sample_chain(env_ids, first_boundary - 1e-4)[0]
  after = sampler._sample_chain(env_ids, first_boundary + 1e-4)[0]
  during_blend = sampler._sample_chain(env_ids, first_boundary + 0.1)[0]

  assert poses.active.tolist() == [True]
  assert sampler.joint_positions_local_m is not None
  assert sampler.joint_quaternions_local_wxyz is not None
  assert torch.allclose(
    poses.root_positions_w[0, :2], torch.tensor([4.0, 3.0]), atol=1e-5
  )
  assert torch.linalg.vector_norm(before - after) < 1e-3
  # The source clips use different absolute root coordinate frames. Alignment
  # must not reappear as a multi-meter local-position decay after the exactly
  # continuous boundary.
  assert torch.linalg.vector_norm(
    during_blend[:, sampler.anchor_index] - after[:, sampler.anchor_index]
  ) < 0.25
  assert sampler.device_storage_bytes > 0


def test_online_skeleton_composer_scales_time_and_locks_root_xy(
  tmp_path: Path,
) -> None:
  path = tmp_path / "skeleton_bank"
  _write_tiny_skeleton_bank(path)
  sampler = OnlineComposedHumanSampler(
    SkeletonPathBank(path), 1, "cpu", lock_root_xy=True
  )
  env_ids = torch.tensor([0])
  sampler.schedule_intersections(
    env_ids,
    sequence_source_ids=torch.tensor([[10, 20, 30]]),
    global_intersection_times_s=torch.tensor([0.0]),
    robot_positions_at_intersection_w=torch.tensor([[4.0, 3.0, 0.0]]),
    robot_yaw_at_intersection=torch.zeros(1),
    robot_path_heading_at_intersection=torch.zeros(1),
    action_phase=0.0,
    ground_height_m=torch.zeros(1),
    playback_speed=0.5,
    placement_yaw_override=torch.tensor([0.25]),
  )

  first = sampler.sample_held(0.0).root_positions_w.clone()
  sampler.dirty[:] = True
  second = sampler.sample_held(0.1).root_positions_w.clone()

  assert sampler.playback_speed.tolist() == [0.5]
  assert sampler.placement_yaw.tolist() == pytest.approx([0.25])
  assert first[0, :2].tolist() == pytest.approx([4.0, 3.0])
  assert second[0, :2].tolist() == pytest.approx([4.0, 3.0])
  assert sampler.current_translation_w[0, 0] != sampler.translation_w[0, 0]


def test_online_skeleton_composer_can_align_anatomical_facing(
  tmp_path: Path,
) -> None:
  path = tmp_path / "skeleton_bank"
  _write_tiny_skeleton_bank(path)
  sampler = OnlineComposedHumanSampler(SkeletonPathBank(path), 1, "cpu")
  sampler.schedule_intersections(
    torch.tensor([0]),
    sequence_source_ids=torch.tensor([[10, 20, 30]]),
    global_intersection_times_s=torch.zeros(1),
    robot_positions_at_intersection_w=torch.zeros((1, 3)),
    robot_yaw_at_intersection=torch.zeros(1),
    robot_path_heading_at_intersection=torch.zeros(1),
    action_phase=0.0,
    crossing_angle_rad=0.0,
    align_to_facing=True,
  )

  # The synthetic root travels along +X but its identity hip faces +Y.
  # Anatomical alignment must rotate +Y onto the desired +X heading.
  assert sampler.placement_yaw.tolist() == pytest.approx([-torch.pi / 2])


def test_online_skeleton_composer_supports_coarse_crowd_proxies(
  tmp_path: Path,
) -> None:
  path = tmp_path / "skeleton_bank"
  _write_tiny_skeleton_bank(path)
  sampler = OnlineComposedHumanSampler(
    SkeletonPathBank(path),
    2,
    "cpu",
    capsule_specs=SOMA_CROWD_PROXY_SPECS,
  )

  assert sampler._poses.centers_w.shape == (2, 5, 3)
  assert sampler._poses.radii_m.shape == (2, 5)


def test_stationary_sequence_continuation_preserves_boundary_pose(
  tmp_path: Path,
) -> None:
  path = tmp_path / "skeleton_bank"
  _write_tiny_skeleton_bank(path)
  sampler = OnlineComposedHumanSampler(
    SkeletonPathBank(path), 1, "cpu", lock_root_xy=True
  )
  env_ids = torch.tensor([0])
  slot = torch.tensor([[4.0, 3.0, 0.0]])
  sampler.schedule_intersections(
    env_ids,
    sequence_source_ids=torch.tensor([[10, 20, 30]]),
    global_intersection_times_s=torch.tensor([0.0]),
    robot_positions_at_intersection_w=slot,
    robot_yaw_at_intersection=torch.zeros(1),
    robot_path_heading_at_intersection=torch.zeros(1),
    action_phase=0.5,
    ground_height_m=torch.zeros(1),
    body_scale_xyz=torch.tensor([[1.05, 0.95, 1.1]]),
  )
  end_time = float(
    sampler.total_durations_s[0] - sampler.local_intersection_times_s[0]
  )
  sampler.dirty[:] = True
  before = sampler.sample_held(end_time)
  centers_before = before.centers_w.clone()

  old_yaw = sampler.placement_yaw.clone()
  next_yaw = old_yaw + sampler.alignment_yaw[:, 2]
  aligned_translation = (
    sampler.alignment_translation[:, 2] * sampler.body_scale_xyz
  )
  cosine, sine = torch.cos(old_yaw), torch.sin(old_yaw)
  rotated = aligned_translation.clone()
  rotated[:, 0] = cosine * aligned_translation[:, 0] - sine * aligned_translation[:, 1]
  rotated[:, 1] = sine * aligned_translation[:, 0] + cosine * aligned_translation[:, 1]
  next_translation = sampler.translation_w + rotated
  sampler.schedule_intersections(
    env_ids,
    sequence_source_ids=torch.tensor([[30, 10, 20]]),
    global_intersection_times_s=torch.tensor([end_time]),
    robot_positions_at_intersection_w=slot,
    robot_yaw_at_intersection=torch.zeros(1),
    robot_path_heading_at_intersection=torch.zeros(1),
    action_phase=0.0,
    ground_height_m=torch.zeros(1),
    body_scale_xyz=sampler.body_scale_xyz.clone(),
    placement_yaw_override=next_yaw,
    translation_override_w=next_translation,
  )
  after = sampler.sample_held(end_time)

  assert torch.allclose(centers_before, after.centers_w, atol=2e-5)


def test_expiry_ignores_unscheduled_and_deactivated_envs(tmp_path: Path) -> None:
  path = tmp_path / "skeleton_bank"
  _write_tiny_skeleton_bank(path)
  sampler = OnlineComposedHumanSampler(SkeletonPathBank(path), 2, "cpu")
  # Never-scheduled environments must not expire at any time.
  assert sampler.expired_env_ids(1.0e6).numel() == 0

  sampler.schedule_intersections(
    torch.tensor([0]),
    sequence_source_ids=torch.tensor([[10, 20, 30]]),
    global_intersection_times_s=torch.tensor([2.0]),
    robot_positions_at_intersection_w=torch.tensor([[4.0, 3.0, 0.0]]),
    robot_yaw_at_intersection=torch.zeros(1),
    robot_path_heading_at_intersection=torch.zeros(1),
    ground_height_m=torch.zeros(1),
  )
  far_future = 1.0e6
  assert sampler.expired_env_ids(far_future).tolist() == [0]
  # Deactivation (a parked human) silences the stale schedule's expiry so the
  # driving event never revives it mid-episode.
  sampler.scheduled[torch.tensor([0])] = False
  assert sampler.expired_env_ids(far_future).numel() == 0
