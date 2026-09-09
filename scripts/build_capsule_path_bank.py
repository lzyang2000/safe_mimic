#!/usr/bin/env python3
"""Prebuild retained BONES-SEED events as a memory-mapped capsule path bank."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safe_mimic.motions import SOMA_CAPSULE_SPECS, fit_soma_capsules, load_bvh_window


@dataclass(frozen=True)
class Segment:
  record: dict[str, object]
  start_time_s: float
  end_time_s: float
  description: str


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument(
    "--manifest",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/walk_punch_kick_1000/manifest.jsonl"),
  )
  parser.add_argument(
    "--temporal-labels",
    type=Path,
    default=Path(
      "artifacts/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
    ),
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/capsule_path_bank_50hz_6s"),
  )
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--max-duration", type=float, default=6.0)
  parser.add_argument(
    "--storage-dtype", choices=("float16", "float32"), default="float16"
  )
  parser.add_argument(
    "--families",
    nargs="+",
    choices=("walk", "punch", "kick"),
    default=("walk", "punch", "kick"),
  )
  parser.add_argument("--limit", type=int, help="build only the first N events")
  return parser.parse_args()


def _load_records(path: Path, families: set[str]) -> list[dict[str, object]]:
  with path.open() as source:
    return [
      record for line in source if (record := json.loads(line))["family"] in families
    ]


def _load_labels(
  path: Path, records: list[dict[str, object]]
) -> dict[str, list[dict[str, object]]]:
  wanted = {
    name
    for record in records
    for name in (Path(str(record["path"])).stem, str(record["move_name"]))
  }
  result: dict[str, list[dict[str, object]]] = {}
  with path.open() as source:
    for line in source:
      label = json.loads(line)
      filename = str(label["filename"])
      if filename in wanted:
        result[filename] = list(label["events"])
  return result


def _make_segments(
  records: list[dict[str, object]],
  labels: dict[str, list[dict[str, object]]],
  limit: int | None,
) -> list[Segment]:
  segments: list[Segment] = []
  for record in records:
    events = labels.get(Path(str(record["path"])).stem)
    if events is None:
      events = labels.get(str(record["move_name"]), [])
    if not events:
      events = [
        {
          "start_time": 0.0,
          "end_time": float(record["duration_frames"]) / 120.0,
          "description": record["description"],
        }
      ]
    for event in events:
      start_time_s = float(event["start_time"])
      end_time_s = float(event["end_time"])
      if end_time_s <= start_time_s:
        continue
      segments.append(
        Segment(
          record=record,
          start_time_s=start_time_s,
          end_time_s=end_time_s,
          description=str(event["description"]),
        )
      )
      if limit is not None and len(segments) >= limit:
        return segments
  return segments


def _create_arrays(
  directory: Path,
  path_count: int,
  max_frames: int,
  capsule_count: int,
  storage_dtype: np.dtype,
) -> dict[str, np.memmap]:
  open_memmap = np.lib.format.open_memmap
  return {
    "centers": open_memmap(
      directory / "centers.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, max_frames, capsule_count, 3),
    ),
    "quaternions": open_memmap(
      directory / "quaternions.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, max_frames, capsule_count, 4),
    ),
    "root_positions": open_memmap(
      directory / "root_positions.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, max_frames, 3),
    ),
    "facing_yaw": open_memmap(
      directory / "facing_yaw.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, max_frames),
    ),
    "radii": open_memmap(
      directory / "radii.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, capsule_count),
    ),
    "half_lengths": open_memmap(
      directory / "half_lengths.npy",
      mode="w+",
      dtype=storage_dtype,
      shape=(path_count, capsule_count),
    ),
    "frame_counts": open_memmap(
      directory / "frame_counts.npy",
      mode="w+",
      dtype=np.int32,
      shape=(path_count,),
    ),
    "ground_z": open_memmap(
      directory / "ground_z.npy",
      mode="w+",
      dtype=np.float32,
      shape=(path_count,),
    ),
  }


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


def _mujoco_facing_yaw(motion) -> np.ndarray:
  joint_index = {name: index for index, name in enumerate(motion.joint_names)}.get(
    "Hips", 0
  )
  forward_bvh = np.einsum(
    "nij,j->ni",
    motion.rotations_world[:, joint_index],
    np.asarray([0.0, 0.0, 1.0]),
  )
  return np.arctan2(forward_bvh[:, 2], forward_bvh[:, 0])


def _fill_segment(
  arrays: dict[str, np.memmap],
  path_index: int,
  segment: Segment,
  motion,
  fit,
  facing_yaw: np.ndarray,
  fps: float,
  max_duration_s: float,
  max_frames: int,
) -> dict[str, object]:
  source_start_s = float(np.clip(segment.start_time_s, 0.0, motion.duration_s))
  source_end_s = float(np.clip(segment.end_time_s, source_start_s, motion.duration_s))
  duration_s = min(max_duration_s, source_end_s - source_start_s)
  if duration_s <= 0.0:
    duration_s = min(1.0 / fps, max(motion.duration_s, 1.0 / fps))
  source_start_s += 0.5 * max(0.0, source_end_s - source_start_s - duration_s)
  frame_count = min(max_frames, max(1, int(math.ceil(duration_s * fps))))
  target_times_s = source_start_s + np.arange(frame_count) / fps
  source_times_s = motion.frame_indices * motion.source_frame_time
  indices = _nearest_indices(source_times_s, target_times_s)

  arrays["centers"][path_index, :frame_count] = fit.centers_m[indices]
  arrays["quaternions"][path_index, :frame_count] = fit.quaternions_wxyz[indices]
  arrays["root_positions"][path_index, :frame_count] = fit.root_path_m[indices]
  arrays["facing_yaw"][path_index, :frame_count] = facing_yaw[indices]
  if frame_count < max_frames:
    arrays["centers"][path_index, frame_count:] = fit.centers_m[indices[-1]]
    arrays["quaternions"][path_index, frame_count:] = fit.quaternions_wxyz[indices[-1]]
    arrays["root_positions"][path_index, frame_count:] = fit.root_path_m[indices[-1]]
    arrays["facing_yaw"][path_index, frame_count:] = facing_yaw[indices[-1]]
  arrays["radii"][path_index] = fit.radii_m
  arrays["half_lengths"][path_index] = fit.half_lengths_m
  arrays["frame_counts"][path_index] = frame_count

  joint_indices = {name: index for index, name in enumerate(motion.joint_names)}
  support = [
    joint_indices[name]
    for name in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
    if name in joint_indices
  ]
  if support:
    support_z = motion.positions_m[indices][:, support, 1]
    ground_z = float(np.percentile(support_z, 2.0))
  else:
    ground_z = float(np.min(motion.positions_m[indices, :, 1]))
  arrays["ground_z"][path_index] = ground_z

  return {
    "path_id": path_index,
    "family": segment.record["family"],
    "move_name": segment.record["move_name"],
    "source_path": segment.record["path"],
    "description": segment.description,
    "source_start_time_s": source_start_s,
    "source_end_time_s": source_start_s + duration_s,
    "frame_count": frame_count,
  }


def main() -> None:
  args = parse_args()
  if args.fps <= 0.0 or args.max_duration <= 0.0:
    raise ValueError("fps and max-duration must be positive")
  if args.limit is not None and args.limit < 1:
    raise ValueError("limit must be positive")
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite existing bank: {args.output}")

  records = _load_records(args.manifest, set(args.families))
  labels = _load_labels(args.temporal_labels, records)
  segments = _make_segments(records, labels, args.limit)
  if not segments:
    raise ValueError("no temporal segments selected")
  max_frames = int(math.ceil(args.max_duration * args.fps))
  capsule_count = len(SOMA_CAPSULE_SPECS)
  storage_dtype = np.dtype(args.storage_dtype)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_name(args.output.name + ".building")
  if temporary.exists():
    raise FileExistsError(f"remove or inspect incomplete build: {temporary}")
  temporary.mkdir()
  arrays = _create_arrays(
    temporary, len(segments), max_frames, capsule_count, storage_dtype
  )
  segments_by_source: dict[str, list[tuple[int, Segment]]] = defaultdict(list)
  for path_index, segment in enumerate(segments):
    segments_by_source[str(segment.record["path"])].append((path_index, segment))

  metadata: list[dict[str, object] | None] = [None] * len(segments)
  completed = 0
  for source_index, (source_path, source_segments) in enumerate(
    segments_by_source.items(), 1
  ):
    motion = load_bvh_window(
      args.dataset_root / source_path,
      start_time_s=0.0,
      duration_s=1.0e9,
      output_fps=args.fps,
    )
    fit = fit_soma_capsules(motion.joint_names, motion.positions_m)
    facing_yaw = _mujoco_facing_yaw(motion)
    for path_index, segment in source_segments:
      metadata[path_index] = _fill_segment(
        arrays,
        path_index,
        segment,
        motion,
        fit,
        facing_yaw,
        args.fps,
        args.max_duration,
        max_frames,
      )
      completed += 1
    if source_index % 50 == 0 or completed == len(segments):
      print(
        f"compiled {completed}/{len(segments)} paths from "
        f"{source_index}/{len(segments_by_source)} source clips",
        flush=True,
      )

  for array in arrays.values():
    array.flush()
  with (temporary / "paths.jsonl").open("w") as output:
    for record in metadata:
      assert record is not None
      output.write(json.dumps(record, separators=(",", ":")) + "\n")
  estimated_bytes = (
    len(segments) * max_frames * capsule_count * 7 * storage_dtype.itemsize
  )
  config = {
    "format_version": 1,
    "complete": True,
    "fps": args.fps,
    "max_duration_s": args.max_duration,
    "max_frames": max_frames,
    "storage_dtype": args.storage_dtype,
    "path_count": len(segments),
    "source_clip_count": len(segments_by_source),
    "families": list(args.families),
    "capsule_names": [spec.name for spec in SOMA_CAPSULE_SPECS],
    "pose_storage_bytes": estimated_bytes,
  }
  (temporary / "bank.json").write_text(json.dumps(config, indent=2) + "\n")
  temporary.rename(args.output)
  print(json.dumps({**config, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
  main()
