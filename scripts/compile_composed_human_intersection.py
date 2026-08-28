#!/usr/bin/env python3
"""Compose walk/action/walk, inertialize it, and intersect an mjlab robot path."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from safe_mimic.motions import (
  compile_human_robot_intersection,
  inertialize_bvh_sequence,
  load_bvh_window,
  load_mjlab_robot_trajectory,
  sample_capsule_domain_randomization,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument(
    "--bank",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/capsule_path_bank_50hz_6s"),
  )
  parser.add_argument("--transition-index", type=Path)
  parser.add_argument("--robot-motion", type=Path, required=True)
  parser.add_argument("--robot-root-body-index", type=int, default=0)
  parser.add_argument("--action-family", choices=("punch", "kick"), default="punch")
  parser.add_argument("--action-path-id", type=int)
  parser.add_argument(
    "--action-query",
    help="case-insensitive substring required in the action event description",
  )
  parser.add_argument("--seed", type=int, default=20260826)
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--transition-duration", type=float, default=0.2)
  parser.add_argument("--intersection-time", type=float)
  parser.add_argument("--crossing-angle-deg", type=float, default=90.0)
  parser.add_argument("--offset-forward", type=float, default=0.0)
  parser.add_argument("--offset-left", type=float, default=0.0)
  parser.add_argument("--domain-randomization", action="store_true")
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/bones-seed/compiled/composed_human_intersection.npz"),
  )
  return parser.parse_args()


def _load_path_metadata(path: Path) -> list[dict[str, object]]:
  with path.open() as source:
    records = [json.loads(line) for line in source]
  if any(record["path_id"] != index for index, record in enumerate(records)):
    raise ValueError("bank path metadata is not ordered by path_id")
  return records


def _load_segment(
  record: dict[str, object], dataset_root: Path, output_fps: float
):
  start_time_s = float(record["source_start_time_s"])
  end_time_s = float(record["source_end_time_s"])
  return load_bvh_window(
    dataset_root / str(record["source_path"]),
    start_time_s=start_time_s,
    duration_s=end_time_s - start_time_s,
    output_fps=output_fps,
  )


def main() -> None:
  args = parse_args()
  if args.fps <= 0.0 or args.transition_duration <= 0.0:
    raise ValueError("fps and transition-duration must be positive")
  transition_index = args.transition_index or (args.bank / "transition_index_v3")
  records = _load_path_metadata(args.bank / "paths.jsonl")
  with np.load(transition_index / "edges.npz", allow_pickle=False) as edges:
    action_ids = np.asarray(edges["action_path_ids"], dtype=np.int64)
    entry_ids = np.asarray(edges["action_entry_connector_ids"], dtype=np.int64)
    entry_costs = np.asarray(edges["action_entry_costs"], dtype=np.float64)
    exit_ids = np.asarray(edges["action_exit_connector_ids"], dtype=np.int64)
    exit_costs = np.asarray(edges["action_exit_costs"], dtype=np.float64)

  candidates = [
    index
    for index, path_id in enumerate(action_ids)
    if records[path_id]["family"] == args.action_family
    and (args.action_path_id is None or path_id == args.action_path_id)
    and (
      args.action_query is None
      or args.action_query.casefold()
      in str(records[path_id]["description"]).casefold()
    )
  ]
  if not candidates:
    raise ValueError("requested action is not in the transition-ready graph")
  rng = random.Random(args.seed)
  action_row = rng.choice(candidates)
  neighbor_rank = rng.randrange(entry_ids.shape[1])
  action_path_id = int(action_ids[action_row])
  entry_path_id = int(entry_ids[action_row, neighbor_rank])
  exit_path_id = int(exit_ids[action_row, neighbor_rank])

  source_records = [
    records[entry_path_id],
    records[action_path_id],
    records[exit_path_id],
  ]
  source_motions = [
    _load_segment(record, args.dataset_root, args.fps) for record in source_records
  ]
  composed = inertialize_bvh_sequence(
    source_motions,
    output_fps=args.fps,
    transition_duration_s=args.transition_duration,
  )
  action_start_frame = len(source_motions[0].positions_m) - 1
  action_target_frame = action_start_frame + (
    len(source_motions[1].positions_m) - 1
  ) // 2
  intersection_phase = action_target_frame / (len(composed.positions_m) - 1)

  robot = load_mjlab_robot_trajectory(
    args.robot_motion, root_body_index=args.robot_root_body_index
  )
  intersection_time_s = args.intersection_time
  if intersection_time_s is None:
    intersection_time_s = 0.5 * (float(robot.times_s[0]) + float(robot.times_s[-1]))
  randomization = None
  if args.domain_randomization:
    randomization = sample_capsule_domain_randomization(
      np.random.default_rng(args.seed)
    )
  description = " -> ".join(str(record["description"]) for record in source_records)
  compiled = compile_human_robot_intersection(
    composed,
    robot,
    intersection_time_s=intersection_time_s,
    intersection_phase=intersection_phase,
    crossing_angle_rad=np.deg2rad(args.crossing_angle_deg),
    intersection_offset_robot_m=(args.offset_forward, args.offset_left),
    randomization=randomization,
    source_description=description,
  )
  compiled.save(args.output)

  summary = {
    "output": str(args.output.resolve()),
    "frames": len(compiled.times_s),
    "duration_s": float(compiled.times_s[-1] - compiled.times_s[0]),
    "intersection_time_s": intersection_time_s,
    "intersection_frame": compiled.intersection_frame,
    "intersection_distance_m": compiled.intersection_distance_m,
    "transition_duration_s": args.transition_duration,
    "neighbor_rank": neighbor_rank,
    "paths": {
      "entry_walk": entry_path_id,
      "action": action_path_id,
      "exit_walk": exit_path_id,
    },
    "matching_costs": {
      "entry": float(entry_costs[action_row, neighbor_rank]),
      "exit": float(exit_costs[action_row, neighbor_rank]),
    },
    "descriptions": [record["description"] for record in source_records],
  }
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
