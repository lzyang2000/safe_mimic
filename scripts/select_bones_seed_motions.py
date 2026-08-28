#!/usr/bin/env python3
"""Build a compact BONES-SEED manifest for plain walking and neutral kicks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from safe_mimic.motions import is_plain_walking


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dataset-root",
    type=Path,
    default=Path("artifacts/bones-seed"),
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/bones-seed/metadata/soma_uniform_plain_walk_kick.jsonl"),
  )
  return parser.parse_args()


def selection_reasons(row: dict[str, str]) -> list[str]:
  reasons: list[str] = []
  if is_plain_walking(row):
    reasons.append("plain_walking")
  if (
    row["content_short_description"] == "kicking trash"
    and row["content_uniform_style"] == "neutral"
    and row["content_props"] == "0"
  ):
    reasons.append("neutral_kick")
  return reasons


def main() -> None:
  args = parse_args()
  metadata_path = args.dataset_root / "metadata/seed_metadata_v004.csv"
  args.output.parent.mkdir(parents=True, exist_ok=True)

  selected = 0
  with metadata_path.open(newline="") as source, args.output.open("w") as output:
    for row in csv.DictReader(source):
      reasons = selection_reasons(row)
      if not reasons:
        continue
      path = args.dataset_root / row["move_soma_uniform_path"]
      if not path.is_file():
        raise FileNotFoundError(path)
      record = {
        "move_name": row["move_name"],
        "path": row["move_soma_uniform_path"],
        "duration_frames": int(row["move_duration_frames"]),
        "is_mirror": row["is_mirror"] == "True",
        "package": row["package"],
        "category": row["category"],
        "movement_type": row["content_type_of_movement"],
        "description": row["content_short_description"],
        "selection_reasons": reasons,
      }
      output.write(json.dumps(record, separators=(",", ":")) + "\n")
      selected += 1

  print(f"wrote {selected} motions to {args.output}")


if __name__ == "__main__":
  main()
