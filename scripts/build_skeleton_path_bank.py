#!/usr/bin/env python3
"""Build a compact transition-approved SOMA skeleton bank for CUDA composition."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from safe_mimic.motions.inertialization import (
  _convert_basis_positions,
  _convert_basis_rotations,
  _matrix_to_quaternion_wxyz,
  _to_local_transforms,
)
from safe_mimic.motions.soma_bvh import load_bvh_window

RUNTIME_JOINT_NAMES = (
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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument(
    "--capsule-bank",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/capsule_path_bank_50hz_6s"),
  )
  parser.add_argument("--transition-index", type=Path)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/skeleton_path_bank_transition_v3"),
  )
  parser.add_argument(
    "--storage-dtype", choices=("float16", "float32"), default="float16"
  )
  return parser.parse_args()


def _nearest_indices(
  sample_times_s: np.ndarray, target_times_s: np.ndarray
) -> np.ndarray:
  after = np.searchsorted(sample_times_s, target_times_s, side="left")
  after = np.clip(after, 0, len(sample_times_s) - 1)
  before = np.maximum(after - 1, 0)
  use_before = np.abs(target_times_s - sample_times_s[before]) < np.abs(
    sample_times_s[after] - target_times_s
  )
  return np.where(use_before, before, after)


def _runtime_hierarchy(motion) -> tuple[np.ndarray, np.ndarray]:
  source_indices = {joint.name: index for index, joint in enumerate(motion.joints)}
  missing = [name for name in RUNTIME_JOINT_NAMES if name not in source_indices]
  if missing:
    raise ValueError(f"SOMA hierarchy is missing runtime joints: {missing}")
  selected = np.asarray(
    [source_indices[name] for name in RUNTIME_JOINT_NAMES], dtype=np.int64
  )
  runtime_index = {int(source): index for index, source in enumerate(selected)}
  parents = np.empty(len(selected), dtype=np.int64)
  for runtime_joint, source_joint in enumerate(selected):
    source_parent = motion.joints[int(source_joint)].parent
    while source_parent >= 0 and source_parent not in runtime_index:
      source_parent = motion.joints[source_parent].parent
    parents[runtime_joint] = runtime_index.get(source_parent, -1)
  return selected, parents


def _create_arrays(
  path: Path,
  path_count: int,
  max_frames: int,
  joint_count: int,
  dtype: np.dtype,
) -> dict[str, np.memmap]:
  open_memmap = np.lib.format.open_memmap
  return {
    "local_positions": open_memmap(
      path / "local_positions.npy",
      mode="w+",
      dtype=dtype,
      shape=(path_count, max_frames, joint_count, 3),
    ),
    "local_quaternions": open_memmap(
      path / "local_quaternions.npy",
      mode="w+",
      dtype=dtype,
      shape=(path_count, max_frames, joint_count, 4),
    ),
    "frame_counts": open_memmap(
      path / "frame_counts.npy",
      mode="w+",
      dtype=np.int32,
      shape=(path_count,),
    ),
    "ground_z": open_memmap(
      path / "ground_z.npy",
      mode="w+",
      dtype=np.float32,
      shape=(path_count,),
    ),
    "source_path_ids": open_memmap(
      path / "source_path_ids.npy",
      mode="w+",
      dtype=np.int64,
      shape=(path_count,),
    ),
  }


def main() -> None:
  args = parse_args()
  transition_index = args.transition_index or (
    args.capsule_bank / "transition_index_v3"
  )
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite existing bank: {args.output}")
  with np.load(transition_index / "edges.npz", allow_pickle=False) as graph:
    accepted_ids = np.asarray(graph["accepted_path_ids"], dtype=np.int64)
  with (args.capsule_bank / "bank.json").open() as source:
    capsule_config = json.load(source)
  fps = float(capsule_config["fps"])
  max_frames = int(capsule_config["max_frames"])
  with (args.capsule_bank / "paths.jsonl").open() as source:
    all_records = [json.loads(line) for line in source]
  records = [all_records[int(path_id)] for path_id in accepted_ids]

  temporary = args.output.with_name(args.output.name + ".building")
  if temporary.exists():
    raise FileExistsError(f"remove or inspect incomplete build: {temporary}")
  temporary.parent.mkdir(parents=True, exist_ok=True)
  temporary.mkdir()
  storage_dtype = np.dtype(args.storage_dtype)
  arrays = _create_arrays(
    temporary,
    len(records),
    max_frames,
    len(RUNTIME_JOINT_NAMES),
    storage_dtype,
  )
  by_source: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
  for row, record in enumerate(records):
    by_source[str(record["source_path"])].append((row, record))

  runtime_parents: np.ndarray | None = None
  completed = 0
  for source_index, (source_path, source_records) in enumerate(by_source.items(), 1):
    motion = load_bvh_window(
      args.dataset_root / source_path,
      start_time_s=0.0,
      duration_s=1.0e9,
      output_fps=fps,
    )
    selected, parents = _runtime_hierarchy(motion)
    if runtime_parents is None:
      runtime_parents = parents
    elif not np.array_equal(runtime_parents, parents):
      raise ValueError(f"runtime hierarchy differs in {source_path}")
    positions = _convert_basis_positions(motion.positions_m[:, selected])
    rotations = _convert_basis_rotations(motion.rotations_world[:, selected])
    local_positions, local_rotations = _to_local_transforms(
      positions, rotations, parents
    )
    local_quaternions = _matrix_to_quaternion_wxyz(local_rotations)
    source_times_s = motion.frame_indices * motion.source_frame_time
    foot_indices = [
      RUNTIME_JOINT_NAMES.index(name)
      for name in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
    ]

    for row, record in source_records:
      frame_count = int(record["frame_count"])
      start_time_s = float(record["source_start_time_s"])
      target_times_s = start_time_s + np.arange(frame_count) / fps
      indices = _nearest_indices(source_times_s, target_times_s)
      arrays["local_positions"][row, :frame_count] = local_positions[indices]
      arrays["local_quaternions"][row, :frame_count] = local_quaternions[indices]
      if frame_count < max_frames:
        arrays["local_positions"][row, frame_count:] = local_positions[indices[-1]]
        arrays["local_quaternions"][row, frame_count:] = local_quaternions[indices[-1]]
      arrays["frame_counts"][row] = frame_count
      arrays["ground_z"][row] = float(
        np.percentile(positions[indices][:, foot_indices, 2], 2.0)
      )
      arrays["source_path_ids"][row] = int(record["path_id"])
      completed += 1
    if source_index % 50 == 0 or completed == len(records):
      print(
        f"compiled {completed}/{len(records)} paths from "
        f"{source_index}/{len(by_source)} source clips",
        flush=True,
      )

  assert runtime_parents is not None
  for array in arrays.values():
    array.flush()
  np.save(temporary / "parents.npy", runtime_parents)
  pose_storage_bytes = (
    len(records) * max_frames * len(RUNTIME_JOINT_NAMES) * 7 * storage_dtype.itemsize
  )
  config = {
    "format_version": 1,
    "complete": True,
    "fps": fps,
    "max_frames": max_frames,
    "path_count": len(records),
    "source_path_count": len(all_records),
    "source_clip_count": len(by_source),
    "storage_dtype": args.storage_dtype,
    "joint_names": list(RUNTIME_JOINT_NAMES),
    "pose_storage_bytes": pose_storage_bytes,
    "transition_index": str(transition_index.resolve()),
  }
  (temporary / "bank.json").write_text(json.dumps(config, indent=2) + "\n")
  temporary.rename(args.output)
  print(json.dumps({**config, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
  main()
