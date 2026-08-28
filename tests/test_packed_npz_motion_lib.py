import json
from pathlib import Path

import numpy as np
import torch
import yaml
from mjlab.tasks.tracking.mdp.commands import MotionLoader

from safe_mimic.motions.adaptive_motion_sampling import (
  AdaptiveMotionSampler,
  AdaptiveMotionSamplingCfg,
)
from safe_mimic.motions.packed_npz_motion_lib import (
  PackedNpzMotionLib,
  load_packed_npz_manifest,
)


def _write_tracker_npz(
  path: Path, *, base: float, fps: float = 2.0, frame_count: int = 3
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frames = np.arange(frame_count, dtype=np.float32)
  joint_pos = base + frames[:, None] + np.arange(2, dtype=np.float32)[None]
  joint_vel = np.full((frame_count, 2), base + 0.25, dtype=np.float32)
  body_pos = np.empty((frame_count, 4, 3), dtype=np.float32)
  body_pos[:] = base
  body_pos[..., 0] += frames[:, None]
  body_pos[..., 1] += np.arange(4, dtype=np.float32)[None]
  yaw = frames * (np.pi / 2.0)
  body_quat = np.zeros((frame_count, 4, 4), dtype=np.float32)
  body_quat[..., 0] = np.cos(yaw[:, None] / 2.0)
  body_quat[..., 3] = np.sin(yaw[:, None] / 2.0)
  body_lin_vel = body_pos + 10.0
  body_ang_vel = body_pos + 20.0
  np.savez_compressed(
    path,
    fps=np.asarray([fps], dtype=np.float64),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_lin_vel,
    body_ang_vel_w=body_ang_vel,
  )


def _write_jsonl_manifest(root: Path) -> tuple[Path, Path, Path]:
  first_path = root / "npz_50hz/take/original.npz"
  second_path = root / "npz_50hz/take/original_M.npz"
  _write_tracker_npz(first_path, base=0.0)
  _write_tracker_npz(second_path, base=100.0)
  records = (
    {
      "csv_path": "g1/csv/take/original.csv",
      "tracker_frame_count": 3,
      "weight": 1.0,
      "split": "train",
      "pair_role": "original",
      "sampling_pool": "wbc",
    },
    {
      "csv_path": "g1/csv/take/original_M.csv",
      "tracker_frame_count": 3,
      "weight": 2.0,
      "split": "validation",
      "pair_role": "mirror",
      "sampling_pool": "dance",
    },
  )
  manifest = root / "conversion_manifest.jsonl"
  with manifest.open("w", encoding="utf-8") as stream:
    for record in records:
      stream.write(json.dumps(record) + "\n")
  return manifest, first_path, second_path


def test_jsonl_manifest_keeps_pair_and_pool_metadata(tmp_path: Path) -> None:
  manifest, first_path, second_path = _write_jsonl_manifest(tmp_path)

  sources = load_packed_npz_manifest(manifest)

  assert [source.path for source in sources] == [first_path, second_path]
  assert [source.metadata["pair_role"] for source in sources] == [
    "original",
    "mirror",
  ]
  assert [source.metadata["sampling_pool"] for source in sources] == [
    "wbc",
    "dance",
  ]
  assert load_packed_npz_manifest(manifest, splits="train") == (sources[0],)


def test_packed_frames_match_individual_mjlab_loaders(tmp_path: Path) -> None:
  manifest, first_path, second_path = _write_jsonl_manifest(tmp_path)
  body_indexes = torch.tensor([3, 1])
  library = PackedNpzMotionLib(
    manifest, body_indexes, device="cpu", verbose=False
  )
  individual = (
    MotionLoader(str(first_path), body_indexes, device="cpu"),
    MotionLoader(str(second_path), body_indexes, device="cpu"),
  )

  frame = library.get_frame(torch.tensor([0, 1]), torch.tensor([0.0, 0.5]))

  assert library.num_motions() == 2
  assert library.total_frames == 6
  assert library.motion_start_idx.tolist() == [0, 3]
  assert library.motion_num_frames.tolist() == [3, 3]
  assert library.resident_bytes == 720
  assert torch.allclose(frame.joint_pos[0], individual[0].joint_pos[0])
  assert torch.allclose(frame.body_pos_w[0], individual[0].body_pos_w[0])
  assert torch.allclose(frame.joint_pos[1], individual[1].joint_pos[1])
  assert torch.allclose(frame.body_pos_w[1], individual[1].body_pos_w[1])


def test_verbose_loader_shows_motion_progress_bar(
  tmp_path: Path, capsys
) -> None:
  manifest, _, _ = _write_jsonl_manifest(tmp_path)

  PackedNpzMotionLib(manifest, [0], verbose=True)

  captured = capsys.readouterr()
  assert "[PackedNpzMotionLib] Loading motions" in captured.err
  assert "2/2" in captured.err
  assert "Loaded 2/2 motions" not in captured.out


def test_interpolation_and_last_interval_stay_inside_each_clip(tmp_path: Path) -> None:
  manifest, _, _ = _write_jsonl_manifest(tmp_path)
  library = PackedNpzMotionLib(manifest, [0], verbose=False)

  midpoint = library.get_frame(torch.tensor([0]), torch.tensor([0.25]))
  near_end = library.get_frame(torch.tensor([0]), torch.tensor([0.999]))
  wrapped = library.get_frame(torch.tensor([0]), torch.tensor([1.0]))

  assert torch.allclose(midpoint.joint_pos, torch.tensor([[0.5, 1.5]]))
  expected_quaternion = torch.tensor(
    [[[np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)]]],
    dtype=torch.float32,
  )
  assert torch.allclose(midpoint.body_quat_w, expected_quaternion, atol=1e-6)
  assert 1.99 < near_end.joint_pos[0, 0] <= 2.0
  assert near_end.joint_pos[0, 0] < 10.0
  assert torch.allclose(wrapped.joint_pos, torch.tensor([[0.0, 1.0]]))


def test_yaml_without_frame_counts_uses_npz_headers(tmp_path: Path) -> None:
  motion_root = tmp_path / "motions"
  motion_path = motion_root / "clip.npz"
  _write_tracker_npz(motion_path, base=4.0)
  manifest = tmp_path / "motions.yaml"
  with manifest.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(
      {
        "fps": 2.0,
        "root_path": "motions",
        "motions": [{"file": "clip.npz", "weight": 1.0}],
      },
      stream,
    )

  library = PackedNpzMotionLib(manifest, [2, 0], verbose=False)

  assert library.total_frames == 3
  assert library.get_motion_length(torch.tensor([0])).item() == 1.0
  sampled_ids = library.sample_motions(16)
  sampled_times = library.sample_time(sampled_ids)
  assert sampled_ids.tolist() == [0] * 16
  assert torch.all((sampled_times >= 0.0) & (sampled_times < 1.0))


def test_adaptive_sampler_increases_failed_phase_with_random_floor(
  tmp_path: Path,
) -> None:
  motion_root = tmp_path / "motions"
  _write_tracker_npz(motion_root / "long.npz", base=0.0, frame_count=5)
  _write_tracker_npz(motion_root / "short.npz", base=10.0, frame_count=3)
  manifest = tmp_path / "motions.yaml"
  with manifest.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(
      {
        "fps": 2.0,
        "root_path": "motions",
        "motions": [
          {"file": "long.npz", "weight": 1.0},
          {"file": "short.npz", "weight": 1.0},
        ],
      },
      stream,
    )
  library = PackedNpzMotionLib(manifest, [0], verbose=False)
  sampler = AdaptiveMotionSampler(
    library,
    AdaptiveMotionSamplingCfg(
      failure_ema_alpha=1.0,
      uniform_ratio=0.1,
      couple_pairs=False,
    ),
  )

  sampler.update(
    failures=torch.ones(8, dtype=torch.bool),
    motion_ids=torch.zeros(8, dtype=torch.long),
    motion_times=torch.full((8,), 1.5),
  )

  assert sampler.num_bins == 3
  assert sampler.probabilities[1] > sampler.base_probabilities[1]
  assert torch.all(sampler.probabilities > 0.0)
  assert torch.isclose(sampler.probabilities.sum(), torch.tensor(1.0))
  sample = sampler.sample(128)
  lengths = library.get_motion_length(sample.motion_ids)
  assert torch.all(sample.motion_times >= 0.0)
  assert torch.all(sample.motion_times < lengths)


def test_adaptive_sampler_preserves_pool_mass_and_couples_mirrors(
  tmp_path: Path,
) -> None:
  manifest, _, _ = _write_jsonl_manifest(tmp_path)
  library = PackedNpzMotionLib(manifest, [0], verbose=False)
  sampler = AdaptiveMotionSampler(
    library,
    AdaptiveMotionSamplingCfg(failure_ema_alpha=1.0, uniform_ratio=0.1),
  )
  base_pool_mass = torch.zeros(2)
  base_pool_mass.scatter_add_(
    0, sampler.bin_group_ids, sampler.base_probabilities
  )

  sampler.update(
    failures=torch.tensor([True]),
    motion_ids=torch.tensor([0]),
    motion_times=torch.tensor([0.5]),
  )
  adaptive_pool_mass = torch.zeros(2)
  adaptive_pool_mass.scatter_add_(
    0, sampler.bin_group_ids, sampler.probabilities
  )

  assert sampler.group_names == ("wbc", "dance")
  assert torch.allclose(adaptive_pool_mass, base_pool_mass)
  assert torch.allclose(sampler.failure_ema, torch.tensor([1.0, 0.0]))
  assert torch.isclose(
    sampler.probabilities[0] / sampler.base_probabilities[0],
    sampler.probabilities[1] / sampler.base_probabilities[1],
  )

  state = sampler.state_dict()
  restored = AdaptiveMotionSampler(library, sampler.cfg)
  restored.load_state_dict(state)
  assert torch.allclose(restored.probabilities, sampler.probabilities)
  assert torch.equal(restored.episode_counts, sampler.episode_counts)
