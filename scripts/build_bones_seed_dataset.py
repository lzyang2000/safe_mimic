#!/usr/bin/env python3
"""Build a balanced walk/punch/kick BONES-SEED trajectory manifest."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from safe_mimic.motions import (
  LimbMotionMetrics,
  is_clean_motion_metadata,
  is_plain_walking,
  load_bvh_samples,
  measure_limb_motion,
  qualifies_arm_extension,
  qualifies_leg_extension,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/walk_punch_kick_1000"),
  )
  parser.add_argument("--per-group", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=20260826)
  parser.add_argument("--probe-frames", type=int, default=32)
  parser.add_argument(
    "--workers", type=int, default=min(16, multiprocessing.cpu_count())
  )
  return parser.parse_args()


def _measure_task(
  task: tuple[int, Path, int],
) -> tuple[int, LimbMotionMetrics]:
  index, path, probe_frames = task
  motion = load_bvh_samples(path, probe_frames)
  return index, measure_limb_motion(motion)


def _sample_description_diverse(
  rows: list[dict[str, str]],
  count: int,
  rng: random.Random,
  excluded_names: set[str],
) -> list[dict[str, str]]:
  by_description: dict[str, list[dict[str, str]]] = defaultdict(list)
  for row in rows:
    if row["move_name"] not in excluded_names:
      by_description[row["content_short_description"]].append(row)
  for candidates in by_description.values():
    rng.shuffle(candidates)
  descriptions = list(by_description)
  rng.shuffle(descriptions)

  selected: list[dict[str, str]] = []
  while descriptions and len(selected) < count:
    next_descriptions: list[str] = []
    for description in descriptions:
      candidates = by_description[description]
      selected.append(candidates.pop())
      if candidates:
        next_descriptions.append(description)
      if len(selected) == count:
        break
    rng.shuffle(next_descriptions)
    descriptions = next_descriptions
  if len(selected) != count:
    raise ValueError(f"Requested {count} motions but only found {len(selected)}")
  return selected


def _record(
  family: str,
  row: dict[str, str],
  metrics: LimbMotionMetrics,
) -> dict[str, object]:
  return {
    "family": family,
    "move_name": row["move_name"],
    "path": row["move_soma_uniform_path"],
    "duration_frames": int(row["move_duration_frames"]),
    "actor": row["take_actor"],
    "take": row["take_name"],
    "package": row["package"],
    "category": row["category"],
    "description": row["content_short_description"],
    "movement_type": row["content_type_of_movement"],
    "metrics": asdict(metrics),
  }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
  with path.open("w") as output:
    for record in records:
      output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
  args = parse_args()
  if args.per_group < 1 or args.probe_frames < 2 or args.workers < 1:
    raise ValueError("per-group and workers must be positive; probe-frames >= 2")
  metadata_path = args.dataset_root / "metadata/seed_metadata_v004.csv"
  with metadata_path.open(newline="") as source:
    all_rows = list(csv.DictReader(source))
  originals = [row for row in all_rows if row["is_mirror"] == "False"]

  clean_indices = [
    index for index, row in enumerate(originals) if is_clean_motion_metadata(row)
  ]
  tasks = [
    (
      index,
      args.dataset_root / originals[index]["move_soma_uniform_path"],
      args.probe_frames,
    )
    for index in clean_indices
  ]
  metrics: dict[int, LimbMotionMetrics] = {}
  with multiprocessing.Pool(args.workers) as pool:
    for completed, (index, result) in enumerate(
      pool.imap_unordered(_measure_task, tasks, chunksize=24), 1
    ):
      metrics[index] = result
      if completed % 10000 == 0:
        print(f"measured {completed}/{len(tasks)}", flush=True)

  walking_pool = [
    originals[index] for index in clean_indices if is_plain_walking(originals[index])
  ]
  punch_pool = [
    originals[index]
    for index in clean_indices
    if qualifies_arm_extension(metrics[index])
  ]
  kick_pool = [
    originals[index]
    for index in clean_indices
    if qualifies_leg_extension(metrics[index])
  ]
  metrics_by_name = {
    originals[index]["move_name"]: result for index, result in metrics.items()
  }

  rng = random.Random(args.seed)
  selected_names: set[str] = set()
  selected: dict[str, list[dict[str, str]]] = {}
  for family, pool_rows in (
    ("walk", walking_pool),
    ("punch", punch_pool),
    ("kick", kick_pool),
  ):
    family_rows = _sample_description_diverse(
      pool_rows, args.per_group, rng, selected_names
    )
    selected[family] = family_rows
    selected_names.update(row["move_name"] for row in family_rows)

  records_by_family = {
    family: [_record(family, row, metrics_by_name[row["move_name"]]) for row in rows]
    for family, rows in selected.items()
  }
  all_records = [
    record
    for family in ("walk", "punch", "kick")
    for record in records_by_family[family]
  ]
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _write_jsonl(args.output_dir / "manifest.jsonl", all_records)
  for family, records in records_by_family.items():
    _write_jsonl(args.output_dir / f"{family}.jsonl", records)

  summary = {
    "dataset_root": str(args.dataset_root.resolve()),
    "seed": args.seed,
    "probe_frames": args.probe_frames,
    "per_group": args.per_group,
    "total": len(all_records),
    "selected": {family: len(rows) for family, rows in selected.items()},
    "eligible_originals": {
      "walk": len(walking_pool),
      "punch": len(punch_pool),
      "kick": len(kick_pool),
    },
    "unique_move_names": len(selected_names),
    "uses_mirrors": False,
    "groups_are_disjoint": True,
  }
  (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
