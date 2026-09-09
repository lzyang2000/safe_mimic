#!/usr/bin/env python3
"""Offline-compile stitched human motions into direct-playback keypoint banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from safe_mimic.motions.composed_skeleton_bank import (
  OnlineComposedHumanSampler,
  SkeletonPathBank,
  _quat_apply,
)
from safe_mimic.motions.human_capsules import (
  SOMA_CAPSULE_SPECS,
  SOMA_CROWD_PROXY_SPECS,
  CapsuleSpec,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--kind", choices=("primary", "crowd"), required=True)
  parser.add_argument("--skeleton-bank", type=Path)
  parser.add_argument("--transition-index", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--neighbor-count", type=int, default=8)
  parser.add_argument("--limit", type=int)
  parser.add_argument(
    "--storage-dtype", choices=("float16", "float32"), default="float16"
  )
  return parser.parse_args()


def _default_paths(kind: str) -> tuple[Path, Path]:
  root = Path("artifacts/bones-seed/datasets")
  if kind == "primary":
    return (
      root / "skeleton_path_bank_transition_v3",
      root / "capsule_path_bank_50hz_6s/transition_index_v3",
    )
  return (
    root / "skeleton_path_bank_standing_arm_actions_100",
    root / "standing_arm_actions_100",
  )


def _primary_sequences(path: Path, neighbors: int) -> np.ndarray:
  with np.load(path / "edges.npz", allow_pickle=False) as graph:
    action = np.asarray(graph["action_path_ids"], dtype=np.int64)
    entry = np.asarray(graph["action_entry_connector_ids"], dtype=np.int64)
    exit_ids = np.asarray(graph["action_exit_connector_ids"], dtype=np.int64)
  neighbors = min(neighbors, entry.shape[1])
  return np.stack(
    (
      entry[:, :neighbors],
      np.broadcast_to(action[:, None], (len(action), neighbors)),
      exit_ids[:, :neighbors],
    ),
    axis=-1,
  ).reshape(-1, 3)


def _crowd_sequences(path: Path, neighbors: int) -> np.ndarray:
  with np.load(path / "edges.npz", allow_pickle=False) as graph:
    action = np.asarray(graph["action_path_ids"], dtype=np.int64)
    next_ids = np.asarray(graph["stationary_next_ids"], dtype=np.int64)
  neighbors = min(neighbors, next_ids.shape[1])
  lookup = {int(path_id): row for row, path_id in enumerate(action)}
  sequences: list[tuple[int, int, int]] = []
  for row, first in enumerate(action):
    for rank in range(neighbors):
      second = int(next_ids[row, rank])
      second_row = lookup[second]
      third = int(next_ids[second_row, (rank + 1) % neighbors])
      sequences.append((int(first), second, third))
  return np.asarray(sequences, dtype=np.int64)


def _keypoint_names(specs: tuple[CapsuleSpec, ...]) -> tuple[str, ...]:
  required = {"Hips", "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"}
  for spec in specs:
    required.add(spec.start_joint)
    if spec.end_joint is not None:
      required.add(spec.end_joint)
  return tuple(sorted(required))


def _source_rows(bank: SkeletonPathBank, source_ids: np.ndarray) -> np.ndarray:
  lookup = {int(source): row for row, source in enumerate(bank.source_path_ids)}
  try:
    return np.asarray(
      [[lookup[int(source)] for source in sequence] for sequence in source_ids],
      dtype=np.int64,
    )
  except KeyError as error:
    raise ValueError(
      f"transition graph references missing source path {error}"
    ) from error


def _create_arrays(
  path: Path,
  path_count: int,
  max_frames: int,
  keypoint_count: int,
  storage_dtype: np.dtype,
) -> dict[str, np.memmap]:
  open_memmap = np.lib.format.open_memmap
  return {
    "keypoints": open_memmap(
      path / "keypoints.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, max_frames, keypoint_count, 3),
    ),
    "facing_yaw": open_memmap(
      path / "facing_yaw.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, max_frames),
    ),
    "frame_counts": open_memmap(
      path / "frame_counts.npy", mode="w+", dtype=np.int32, shape=(path_count,)
    ),
    "action_frames": open_memmap(
      path / "action_frames.npy", mode="w+", dtype=np.int32, shape=(path_count,)
    ),
    "entry_travel_m": open_memmap(
      path / "entry_travel_m.npy", mode="w+", dtype=np.float32, shape=(path_count,)
    ),
    "sequence_path_ids": open_memmap(
      path / "sequence_path_ids.npy",
      mode="w+",
      dtype=np.int64,
      shape=(path_count, 3),
    ),
  }


def main() -> None:
  args = parse_args()
  if args.batch_size < 1 or args.neighbor_count < 1:
    raise ValueError("batch size and neighbor count must be positive")
  default_bank, default_graph = _default_paths(args.kind)
  skeleton_path = args.skeleton_bank or default_bank
  graph_path = args.transition_index or default_graph
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite packed bank: {args.output}")

  bank = SkeletonPathBank(skeleton_path)
  sequences = (
    _primary_sequences(graph_path, args.neighbor_count)
    if args.kind == "primary"
    else _crowd_sequences(graph_path, args.neighbor_count)
  )
  if args.limit is not None:
    sequences = sequences[: args.limit]
  if len(sequences) == 0:
    raise ValueError("packed trajectory selection is empty")
  rows = _source_rows(bank, sequences)
  source_frame_counts = np.asarray(bank.frame_counts, dtype=np.int64)[rows]
  frame_counts = source_frame_counts.sum(axis=1) - 2
  max_frames = int(frame_counts.max())
  if args.kind == "primary":
    action_frames = source_frame_counts[:, 0] - 1 + (source_frame_counts[:, 1] - 1) // 2
    specs = SOMA_CAPSULE_SPECS
  else:
    action_frames = np.full(len(sequences), -1, dtype=np.int64)
    specs = SOMA_CROWD_PROXY_SPECS
  keypoint_names = _keypoint_names(specs)
  joint_lookup = {name: index for index, name in enumerate(bank.joint_names)}
  missing = set(keypoint_names) - set(joint_lookup)
  if missing:
    raise ValueError(f"skeleton bank lacks packed keypoints: {sorted(missing)}")
  keypoint_indices = torch.as_tensor(
    [joint_lookup[name] for name in keypoint_names],
    dtype=torch.long,
    device=args.device,
  )
  hips_index = joint_lookup["Hips"]

  temporary = args.output.with_name(args.output.name + ".building")
  if temporary.exists():
    raise FileExistsError(f"incomplete packed build already exists: {temporary}")
  temporary.parent.mkdir(parents=True, exist_ok=True)
  temporary.mkdir()
  storage_dtype = np.dtype(args.storage_dtype)
  arrays = _create_arrays(
    temporary, len(sequences), max_frames, len(keypoint_names), storage_dtype
  )
  arrays["frame_counts"][:] = frame_counts
  arrays["action_frames"][:] = action_frames
  arrays["sequence_path_ids"][:] = sequences

  sampler = OnlineComposedHumanSampler(
    bank,
    args.batch_size,
    args.device,
    capsule_specs=specs,
  )
  for start in range(0, len(sequences), args.batch_size):
    stop = min(start + args.batch_size, len(sequences))
    count = stop - start
    env_ids = torch.arange(count, device=args.device)
    batch_sequences = torch.as_tensor(
      sequences[start:stop], dtype=torch.long, device=args.device
    )
    sampler._prepare_sequences(env_ids, batch_sequences)
    batch_frames = int(frame_counts[start:stop].max())
    frame = torch.arange(batch_frames, device=args.device)
    expanded_ids = env_ids[:, None].expand(-1, batch_frames).reshape(-1)
    times = frame[None].expand(count, -1).reshape(-1).float() / bank.fps
    positions, quaternions, _ = sampler._sample_chain(expanded_ids, times)
    positions = positions.reshape(count, batch_frames, len(bank.joint_names), 3)
    quaternions = quaternions.reshape(count, batch_frames, len(bank.joint_names), 4)
    selected = positions.index_select(2, keypoint_indices)
    hips_quaternion = quaternions[:, :, hips_index]
    forward = _quat_apply(
      hips_quaternion,
      torch.tensor((0.0, 1.0, 0.0), device=args.device).expand(count, batch_frames, 3),
    )
    facing_yaw = torch.atan2(forward[..., 1], forward[..., 0])
    arrays["keypoints"][start:stop, :batch_frames] = (
      selected.detach().cpu().numpy().astype(storage_dtype, copy=False)
    )
    arrays["facing_yaw"][start:stop, :batch_frames] = (
      facing_yaw.detach().cpu().numpy().astype(storage_dtype, copy=False)
    )
    root_index = keypoint_names.index("Hips")
    action = torch.as_tensor(
      action_frames[start:stop].clip(min=0), dtype=torch.long, device=args.device
    )
    roots = selected[:, :, root_index, :2]
    action_roots = roots.gather(1, action[:, None, None].expand(-1, 1, 2)).squeeze(1)
    arrays["entry_travel_m"][start:stop] = (
      torch.linalg.vector_norm(action_roots - roots[:, 0], dim=-1)
      .detach()
      .cpu()
      .numpy()
    )
    completed = stop
    print(
      f"[packed-human-bank] {completed:,}/{len(sequences):,} trajectories",
      flush=True,
    )

  for array in arrays.values():
    array.flush()
  del arrays
  config = {
    "format_version": 1,
    "complete": True,
    "kind": args.kind,
    "fps": bank.fps,
    "path_count": len(sequences),
    "max_frames": max_frames,
    "storage_dtype": args.storage_dtype,
    "keypoint_names": list(keypoint_names),
    "root_keypoint": "Hips",
    "foot_keypoints": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"],
    "capsules": [
      {
        "name": spec.name,
        "start_keypoint": spec.start_joint,
        "end_keypoint": spec.end_joint,
        "radius_m": spec.radius_m,
      }
      for spec in specs
    ],
    "source_skeleton_bank": str(skeleton_path),
    "source_transition_index": str(graph_path),
  }
  (temporary / "bank.json").write_text(json.dumps(config, indent=2) + "\n")
  temporary.rename(args.output)
  size_bytes = sum(path.stat().st_size for path in args.output.iterdir())
  print(
    json.dumps(
      {
        "output": str(args.output.resolve()),
        "kind": args.kind,
        "trajectories": len(sequences),
        "max_frames": max_frames,
        "storage_gib": size_bytes / 2**30,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
