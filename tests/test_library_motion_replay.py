"""Filtered replay over a packed clip library: per-env clip bounds and chaining."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from safe_mimic.motions.packed_npz_motion_lib import PackedNpzMotionLib
from safe_mimic.tasks.kinematic_replay_command import (
  KinematicReplayMotionCommand,
  KinematicReplayMotionCommandCfg,
  LibraryMotionLoader,
  is_motion_manifest,
)

_BODIES = 30
_JOINTS = 29


def _write_clip(path: Path, frames: int, value: float) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    path,
    fps=np.array([50.0]),
    joint_pos=np.full((frames, _JOINTS), value, dtype=np.float32),
    joint_vel=np.zeros((frames, _JOINTS), dtype=np.float32),
    body_pos_w=np.full((frames, _BODIES, 3), value, dtype=np.float32),
    body_quat_w=np.tile(
      np.array([1.0, 0, 0, 0], dtype=np.float32), (frames, _BODIES, 1)
    ),
    body_lin_vel_w=np.zeros((frames, _BODIES, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((frames, _BODIES, 3), dtype=np.float32),
  )


def _manifest(tmp_path: Path, frames=(5, 8, 3)) -> Path:
  root = tmp_path / "npz"
  entries = []
  for i, n in enumerate(frames):
    _write_clip(root / f"clip{i}.npz", n, float(i))
    entries.append(
      {
        "file": f"clip{i}.npz",
        "weight": 1.0,
        "split": "train" if i < 2 else "validation",
      }
    )
  manifest = tmp_path / "lib.yaml"
  manifest.write_text(
    yaml.safe_dump({"fps": 50.0, "root_path": str(root), "motions": entries})
  )
  return manifest


def test_manifest_detection_by_suffix() -> None:
  assert is_motion_manifest("a/ballet.yaml")
  assert is_motion_manifest(Path("a/b.jsonl"))
  assert is_motion_manifest("x.yml")
  assert not is_motion_manifest("artifacts/motions/dance.npz")


def test_library_loader_exposes_flat_frames_and_clip_tables(tmp_path: Path) -> None:
  lib = PackedNpzMotionLib(
    _manifest(tmp_path), torch.arange(4), device="cpu", splits=None, verbose=False
  )
  loader = LibraryMotionLoader(lib)
  assert loader.time_step_total == 16
  assert loader.joint_pos.shape == (16, _JOINTS)
  assert loader.body_pos_w.shape == (16, 4, 3)
  assert loader.body_quat_w.shape == (16, 4, 4)
  assert loader.clip_start_idx.tolist() == [0, 5, 13]
  assert loader.clip_num_frames.tolist() == [5, 8, 3]
  assert loader.num_clips() == 3
  # Frames are copied verbatim: clip i is filled with the value i.
  assert (
    float(loader.joint_pos[6, 0]) == 1.0 and float(loader.body_pos_w[14, 0, 0]) == 2.0
  )


def test_manifest_splits_none_uses_every_clip(tmp_path: Path) -> None:
  every = PackedNpzMotionLib(
    _manifest(tmp_path), torch.arange(4), device="cpu", splits=None, verbose=False
  )
  train = PackedNpzMotionLib(
    _manifest(tmp_path), torch.arange(4), device="cpu", splits=("train",), verbose=False
  )
  assert LibraryMotionLoader(every).num_clips() == 3
  assert LibraryMotionLoader(train).num_clips() == 2


def test_cfg_defaults_to_every_split() -> None:
  fields = {
    f.name: f.default
    for f in KinematicReplayMotionCommandCfg.__dataclass_fields__.values()
  }
  assert fields["manifest_splits"] is None


def _stub(tmp_path: Path, *, mode: str, n: int = 4) -> KinematicReplayMotionCommand:
  command = KinematicReplayMotionCommand.__new__(KinematicReplayMotionCommand)
  lib = PackedNpzMotionLib(
    _manifest(tmp_path), torch.arange(4), device="cpu", splits=None, verbose=False
  )
  command.motion = LibraryMotionLoader(lib)
  command.cfg = SimpleNamespace(sampling_mode=mode)
  command._env = SimpleNamespace(device="cpu")
  command.time_steps = torch.zeros(n, dtype=torch.long)
  command._clip_start = torch.zeros(n, dtype=torch.long)
  command._clip_end = torch.full((n,), 16, dtype=torch.long)
  return command


def test_start_sampling_assigns_a_clip_and_its_first_frame(tmp_path: Path) -> None:
  command = _stub(tmp_path, mode="start")
  torch.manual_seed(0)
  ids = torch.arange(4)
  command._sample_start_time_steps(ids)
  assert torch.equal(command.time_steps, command._clip_start)
  starts = command.motion.clip_start_idx
  for s, e in zip(
    command._clip_start.tolist(), command._clip_end.tolist(), strict=True
  ):
    k = starts.tolist().index(s)
    assert e == s + int(command.motion.clip_num_frames[k])


def test_uniform_sampling_stays_inside_the_assigned_clip(tmp_path: Path) -> None:
  command = _stub(tmp_path, mode="uniform", n=256)
  torch.manual_seed(1)
  command._sample_start_time_steps(torch.arange(256))
  assert bool((command.time_steps >= command._clip_start).all())
  assert bool((command.time_steps < command._clip_end).all())
  # Every clip gets used and frames are not all at clip starts.
  assert len(set(command._clip_start.tolist())) == 3
  assert int((command.time_steps != command._clip_start).sum()) > 0


def test_advance_chains_to_a_new_clip_at_the_end_of_the_current_one(
  tmp_path: Path,
) -> None:
  command = _stub(tmp_path, mode="start")
  # Env 0 is mid-clip (clip 0: frames 0..4); env 1 sits on the last frame of
  # clip 1 (frames 5..12).
  command._clip_start[:] = torch.tensor([0, 5, 0, 0])
  command._clip_end[:] = torch.tensor([5, 13, 5, 5])
  command.time_steps[:] = torch.tensor([2, 12, 0, 0])
  torch.manual_seed(2)
  wrapped = command._advance_time_steps(torch.tensor([0, 1]))
  assert wrapped.tolist() == [False, True]
  assert int(command.time_steps[0]) == 3
  # Env 1 chained: it now sits at the FIRST frame of some clip with matching bounds.
  s, e, t = (
    int(command._clip_start[1]),
    int(command._clip_end[1]),
    int(command.time_steps[1]),
  )
  assert t == s
  assert s in command.motion.clip_start_idx.tolist()
  assert e == s + int(
    command.motion.clip_num_frames[command.motion.clip_start_idx.tolist().index(s)]
  )


def test_single_clip_replay_still_wraps_to_frame_zero() -> None:
  """Stubs without clip tables (the single-NPZ path) keep the modulo behaviour."""
  command = KinematicReplayMotionCommand.__new__(KinematicReplayMotionCommand)
  command.motion = SimpleNamespace(time_step_total=5)
  command.time_steps = torch.tensor([4, 1])
  wrapped = command._advance_time_steps(torch.tensor([0, 1]))
  assert wrapped.tolist() == [True, False]
  assert command.time_steps.tolist() == [0, 2]
