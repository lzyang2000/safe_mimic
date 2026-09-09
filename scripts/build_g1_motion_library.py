#!/usr/bin/env python3
"""Filter BONES-SEED G1 CSV motions into balanced training manifests."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import yaml
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from safe_mimic.motions.g1_dataset import (
  G1MotionFilterCfg,
  assign_split,
  balanced_package_sample,
  filter_g1_motion,
  is_dance_motion,
  load_metadata_rows,
  metadata_rejection_reason,
  quality_as_dict,
)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument("--metadata", type=Path)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
  parser.add_argument("--max-files", type=int, default=-1)
  parser.add_argument(
    "--max-per-package",
    type=int,
    default=0,
    help="Optional balanced cap per package; zero keeps the full accepted set",
  )
  parser.add_argument("--validation-fraction", type=float, default=0.1)
  parser.add_argument("--target-fps", type=float, default=50.0)
  parser.add_argument("--native-fps", type=float, default=120.0)
  parser.add_argument("--min-duration-s", type=float, default=0.5)
  return parser


def _joint_limits() -> tuple[np.ndarray, np.ndarray]:
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  if model.njnt != 30:
    raise RuntimeError(f"expected floating base plus 29 joints, found {model.njnt}")
  return model.jnt_range[1:, 0].copy(), model.jnt_range[1:, 1].copy()


def _worker(
  task: tuple[
    str,
    Mapping[str, str],
    np.ndarray,
    np.ndarray,
    G1MotionFilterCfg,
  ],
) -> dict[str, object]:
  csv_path, metadata, lower, upper, cfg = task
  result = filter_g1_motion(csv_path, metadata, lower, upper, cfg)
  record: dict[str, object] = {
    "move_name": metadata.get("move_name", Path(csv_path).stem),
    "csv_path": metadata.get("move_g1_path", csv_path),
    "package": metadata.get("package", ""),
    "category": metadata.get("category", ""),
    "movement": metadata.get("content_type_of_movement", ""),
    "body_position": metadata.get("content_body_position", ""),
    "description": metadata.get("content_natural_desc_1", ""),
    "take_name": metadata.get("take_name", ""),
    "actor_uid": metadata.get("actor_uid", ""),
    "is_dance": is_dance_motion(metadata),
    "accepted": result.accepted,
    "reason": result.reason,
    **quality_as_dict(result.quality),
  }
  return record


def _write_jsonl(path: Path, records: list[Mapping[str, object]]) -> None:
  with path.open("w", encoding="utf-8") as stream:
    for record in records:
      stream.write(json.dumps(record, sort_keys=True) + "\n")


def _rejection_bucket(reason: object) -> str:
  text = str(reason)
  if text.startswith("root height"):
    return "root_height_below" if " below " in text else "root_height_above"
  if text.startswith("root tilt"):
    return "root_tilt"
  if text.startswith("root speed"):
    return "root_speed"
  if text.startswith("joint speed"):
    return "joint_speed"
  if text.startswith("joint-limit"):
    return "joint_limit"
  if text.startswith("duration"):
    return "duration"
  if text.startswith("load/format"):
    return "load_or_format"
  return text


def main() -> None:
  args = _parser().parse_args()
  metadata_path = args.metadata or (
    args.dataset_root / "metadata/seed_metadata_v004.csv"
  )
  output_dir = args.output_dir or (args.dataset_root / "datasets/g1_general_mimic_v1")
  if not 0.0 < args.validation_fraction < 1.0:
    raise ValueError("validation-fraction must be between zero and one")
  if args.workers < 1:
    raise ValueError("workers must be positive")

  cfg = G1MotionFilterCfg(
    native_fps=args.native_fps,
    target_fps=args.target_fps,
    min_duration_s=args.min_duration_s,
  )
  lower, upper = _joint_limits()

  rows: list[dict[str, str]] = []
  metadata_rejections = Counter()
  missing = 0
  for row in load_metadata_rows(metadata_path):
    reason = metadata_rejection_reason(row)
    if reason is not None:
      metadata_rejections[reason] += 1
      continue
    relative_path = row.get("move_g1_path", "")
    csv_path = args.dataset_root / relative_path
    if not relative_path or not csv_path.is_file():
      missing += 1
      continue
    row["_resolved_csv_path"] = str(csv_path)
    rows.append(row)
    if 0 < args.max_files <= len(rows):
      break

  print(
    f"Kinematic filtering {len(rows):,} metadata-compatible originals with "
    f"{args.workers} workers"
  )
  tasks = [(row["_resolved_csv_path"], row, lower, upper, cfg) for row in rows]
  records: list[dict[str, object]] = []
  with ProcessPoolExecutor(max_workers=args.workers) as executor:
    for index, record in enumerate(executor.map(_worker, tasks, chunksize=16), 1):
      records.append(record)
      if index % 1000 == 0 or index == len(tasks):
        accepted = sum(bool(item["accepted"]) for item in records)
        print(f"  {index:,}/{len(tasks):,}: {accepted:,} accepted", flush=True)

  accepted_records = [record for record in records if bool(record["accepted"])]
  for record in accepted_records:
    record["split"] = assign_split(record, args.validation_fraction)

  selected = (
    list(balanced_package_sample(accepted_records, args.max_per_package))
    if args.max_per_package > 0
    else list(accepted_records)
  )
  selected.sort(key=lambda row: (str(row["split"]), str(row["move_name"])))
  train = [record for record in selected if record["split"] == "train"]
  validation = [record for record in selected if record["split"] == "validation"]
  rejected = [record for record in records if not bool(record["accepted"])]

  output_dir.mkdir(parents=True, exist_ok=True)
  _write_jsonl(output_dir / "filtered_all.jsonl", accepted_records)
  _write_jsonl(output_dir / "rejected_kinematic.jsonl", rejected)
  _write_jsonl(output_dir / "train.jsonl", train)
  _write_jsonl(output_dir / "validation.jsonl", validation)

  yaml_config = {
    "dataset_root": str(args.dataset_root.resolve()),
    "target_fps": cfg.target_fps,
    "motions": [
      {
        "file": record["csv_path"],
        "weight": 1.0,
        "description": record["description"],
        "split": record["split"],
      }
      for record in selected
    ],
  }
  with (output_dir / "csv_manifest.yaml").open("w", encoding="utf-8") as stream:
    yaml.safe_dump(yaml_config, stream, sort_keys=False)

  rejection_counts = Counter(_rejection_bucket(record["reason"]) for record in rejected)
  package_counts = Counter(str(record["package"]) for record in selected)
  category_counts = Counter(str(record["category"]) for record in selected)
  summary = {
    "configuration": asdict(cfg),
    "metadata_path": str(metadata_path),
    "metadata_compatible_originals": len(rows),
    "missing_csv_files": missing,
    "metadata_rejections": dict(metadata_rejections.most_common()),
    "kinematic_accepted": len(accepted_records),
    "kinematic_rejected": len(rejected),
    "kinematic_rejections": dict(rejection_counts.most_common()),
    "selected_count": len(selected),
    "selection_mode": (
      f"balanced cap {args.max_per_package} per package"
      if args.max_per_package > 0
      else "full accepted original set"
    ),
    "train_count": len(train),
    "validation_count": len(validation),
    "dance_count": sum(bool(record["is_dance"]) for record in selected),
    "package_counts": dict(sorted(package_counts.items())),
    "category_counts": dict(sorted(category_counts.items())),
  }
  with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")

  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
