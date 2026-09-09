#!/usr/bin/env python3
"""Compile one retained BONES-SEED clip to cross an mjlab robot path."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from safe_mimic.motions import (
  compile_human_robot_intersection,
  load_bvh_samples,
  load_bvh_window,
  load_mjlab_robot_trajectory,
  sample_capsule_domain_randomization,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Place one retained SOMA human motion so its root path intersects an "
      "mjlab robot reference at a synchronized time."
    )
  )
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
  parser.add_argument("--robot-motion", type=Path, required=True)
  parser.add_argument("--robot-root-body-index", type=int, default=0)
  parser.add_argument("--family", choices=("walk", "punch", "kick"), default="walk")
  parser.add_argument("--move-name")
  parser.add_argument(
    "--event-query",
    help="case-insensitive substring required in the temporal event description",
  )
  parser.add_argument("--seed", type=int, default=20260826)
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--max-duration", type=float, default=6.0)
  parser.add_argument("--intersection-time", type=float)
  parser.add_argument("--intersection-phase", type=float, default=0.5)
  parser.add_argument("--crossing-angle-deg", type=float, default=90.0)
  parser.add_argument("--offset-forward", type=float, default=0.0)
  parser.add_argument("--offset-left", type=float, default=0.0)
  parser.add_argument("--ground-height", type=float, default=0.0)
  parser.add_argument("--domain-randomization", action="store_true")
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/bones-seed/compiled/human_intersection.npz"),
  )
  return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
  with path.open() as source:
    return [json.loads(line) for line in source]


def _load_relevant_labels(
  path: Path, records: list[dict[str, object]]
) -> dict[str, dict[str, object]]:
  wanted = {
    name
    for record in records
    for name in (Path(str(record["path"])).stem, str(record["move_name"]))
  }
  labels: dict[str, dict[str, object]] = {}
  with path.open() as source:
    for line in source:
      label = json.loads(line)
      filename = str(label["filename"])
      if filename in wanted:
        labels[filename] = label
  return labels


def _events_for_record(
  record: dict[str, object], labels: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
  label = labels.get(Path(str(record["path"])).stem)
  if label is None:
    label = labels.get(str(record["move_name"]))
  if label is None:
    return []
  return list(label["events"])


def _select_record_and_event(
  records: list[dict[str, object]],
  labels: dict[str, dict[str, object]],
  *,
  event_query: str | None,
  rng: random.Random,
) -> tuple[dict[str, object], dict[str, object] | None]:
  if event_query:
    query = event_query.casefold()
    candidates = [
      (record, event)
      for record in records
      for event in _events_for_record(record, labels)
      if query in str(event["description"]).casefold()
    ]
    if not candidates:
      raise ValueError(f"no retained temporal events contain {event_query!r}")
    return rng.choice(candidates)

  record = rng.choice(records)
  events = _events_for_record(record, labels)
  if not events:
    return record, None
  event = max(
    events,
    key=lambda item: float(item["end_time"]) - float(item["start_time"]),
  )
  return record, event


def main() -> None:
  args = parse_args()
  if args.fps <= 0.0 or args.max_duration <= 0.0:
    raise ValueError("fps and max-duration must be positive")
  if not 0.0 <= args.intersection_phase <= 1.0:
    raise ValueError("intersection-phase must be in [0, 1]")

  records = [
    record
    for record in _read_jsonl(args.manifest)
    if record["family"] == args.family
    and (args.move_name is None or record["move_name"] == args.move_name)
  ]
  if not records:
    detail = f" and move {args.move_name!r}" if args.move_name else ""
    raise ValueError(f"manifest has no {args.family!r} records{detail}")
  labels = _load_relevant_labels(args.temporal_labels, records)
  record, event = _select_record_and_event(
    records,
    labels,
    event_query=args.event_query,
    rng=random.Random(args.seed),
  )

  bvh_path = args.dataset_root / str(record["path"])
  probe = load_bvh_samples(bvh_path, sample_count=2)
  if event is None:
    event_start_s = 0.0
    event_end_s = probe.duration_s
    description = str(record["description"])
  else:
    event_start_s = float(event["start_time"])
    event_end_s = float(event["end_time"])
    description = str(event["description"])
  event_start_s = float(np.clip(event_start_s, 0.0, probe.duration_s))
  event_end_s = float(np.clip(event_end_s, event_start_s, probe.duration_s))
  event_duration_s = event_end_s - event_start_s
  duration_s = min(args.max_duration, event_duration_s)
  if duration_s <= 0.0:
    raise ValueError(f"selected event has no usable duration: {description!r}")
  source_start_s = event_start_s + 0.5 * (event_duration_s - duration_s)
  motion = load_bvh_window(
    bvh_path,
    start_time_s=source_start_s,
    duration_s=duration_s,
    output_fps=args.fps,
  )

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
  compiled = compile_human_robot_intersection(
    motion,
    robot,
    intersection_time_s=intersection_time_s,
    intersection_phase=args.intersection_phase,
    crossing_angle_rad=np.deg2rad(args.crossing_angle_deg),
    intersection_offset_robot_m=(args.offset_forward, args.offset_left),
    ground_height_m=args.ground_height,
    randomization=randomization,
    source_start_time_s=source_start_s,
    source_description=description,
  )
  compiled.save(args.output)

  summary = {
    "output": str(args.output.resolve()),
    "family": args.family,
    "move_name": record["move_name"],
    "source_path": str(bvh_path),
    "source_description": description,
    "source_window_s": [source_start_s, source_start_s + duration_s],
    "frames": len(compiled.times_s),
    "trajectory_time_s": [float(compiled.times_s[0]), float(compiled.times_s[-1])],
    "intersection_time_s": compiled.intersection_time_s,
    "intersection_frame": compiled.intersection_frame,
    "intersection_distance_m": compiled.intersection_distance_m,
    "crossing_angle_deg": args.crossing_angle_deg,
    "offset_robot_m": [args.offset_forward, args.offset_left],
    "domain_randomization": args.domain_randomization,
  }
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
