#!/usr/bin/env python3
"""Build a curated ballet-technique manifest from BONES-SEED.

BONES-SEED teaches classical technique in one ``dance_basic_*`` package plus a
small standalone pirouette take. This groups those clips by step, applies the
same kinematic filter the general mimic library uses, and writes manifests that
``convert_g1_tracker_npz.py`` can turn into 50 Hz tracker NPZs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from safe_mimic.motions.g1_dataset import (
  G1MotionFilterCfg,
  assign_split,
  filter_g1_motion,
  load_metadata_rows,
  quality_as_dict,
)

# Take-name fragment -> ballet step group. Order matters: the first match wins,
# so "double_slide" resolves through "slide" and pirouette loops that live in
# the turn_v2 take are still labelled as turns.
_GROUP_RULES: tuple[tuple[str, str], ...] = (
  ("chaines", "chaines"),
  ("padeburee", "pas_de_bourree"),
  ("pirouette", "pirouette"),
  ("entry_position", "position"),
  ("cross_turn", "jump_turn"),
  ("turn_v2", "ballet_turn"),
  ("turn_v1", "travelling_turn"),
  ("head_accent", "head_accent"),
  ("slide", "glissade"),
)

CORE_GROUPS = (
  "chaines",
  "pas_de_bourree",
  "pirouette",
  "ballet_turn",
  "position",
  "glissade",
)
EXTENDED_GROUPS = ("travelling_turn", "head_accent")
JUMP_GROUPS = ("jump_turn",)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/g1_ballet_v1"),
  )
  parser.add_argument("--target-fps", type=float, default=50.0)
  parser.add_argument("--validation-fraction", type=float, default=0.1)
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument(
    "--include-extended",
    action="store_true",
    help="Also keep travelling turns and head accents from the same package.",
  )
  parser.add_argument(
    "--include-jump-turns",
    action="store_true",
    help="Also keep the cross-turn takes, which leave the floor.",
  )
  return parser.parse_args()


def ballet_group(take_name: str) -> str | None:
  """Return the ballet step group for a take, or None when it is not one."""
  name = take_name.lower()
  if not (name.startswith("dance_basic") or "pirouette" in name):
    return None
  for fragment, group in _GROUP_RULES:
    if fragment in name:
      return group
  return "other_basic"


def joint_limits() -> tuple[np.ndarray, np.ndarray]:
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  if model.njnt != 30:
    raise RuntimeError(f"expected floating base plus 29 joints, found {model.njnt}")
  return model.jnt_range[1:, 0].copy(), model.jnt_range[1:, 1].copy()


_LIMITS: tuple[np.ndarray, np.ndarray] | None = None
_CFG: G1MotionFilterCfg | None = None
_ROOT: Path | None = None


def _init(root: Path, cfg: G1MotionFilterCfg) -> None:
  global _LIMITS, _CFG, _ROOT
  _LIMITS, _CFG, _ROOT = joint_limits(), cfg, root


def _worker(task: tuple[Mapping[str, str], str, str]) -> dict[str, object]:
  assert _LIMITS is not None and _CFG is not None and _ROOT is not None
  row, group, split = task
  csv_relative = str(row["move_g1_path"])
  result = filter_g1_motion(_ROOT / csv_relative, row, *_LIMITS, _CFG)
  return {
    "move_name": row["move_name"],
    "csv_path": csv_relative,
    "take_name": row["take_name"],
    "ballet_group": group,
    "package": row["package"],
    "category": row["category"],
    "description": row["content_natural_desc_1"],
    "short_description": row["content_short_description"],
    "movement": row["content_type_of_movement"],
    "body_position": row["content_body_position"],
    "actor_uid": row["actor_uid"],
    "is_mirror": row["is_mirror"] == "True",
    "split": split,
    "accepted": result.accepted,
    "reason": result.reason,
    **quality_as_dict(result.quality),
  }


def main() -> None:
  args = parse_args()
  keep = set(CORE_GROUPS)
  if args.include_extended:
    keep.update(EXTENDED_GROUPS)
  if args.include_jump_turns:
    keep.update(JUMP_GROUPS)

  cfg = G1MotionFilterCfg(target_fps=args.target_fps)
  tasks = []
  # Mirrors are kept in the candidate set so the manifest records why they
  # dropped: the shared filter rejects them because training mirrors online.
  for row in load_metadata_rows(args.dataset_root / "metadata/seed_metadata_v004.csv"):
    group = ballet_group(row["take_name"])
    if group is None or group not in keep:
      continue
    tasks.append((row, group, assign_split(row, args.validation_fraction)))

  print(f"filtering {len(tasks)} ballet clips")
  with ProcessPoolExecutor(
    max_workers=args.workers, initializer=_init, initargs=(args.dataset_root, cfg)
  ) as pool:
    records = list(pool.map(_worker, tasks, chunksize=8))

  accepted = [record for record in records if record["accepted"]]
  rejected = [record for record in records if not record["accepted"]]
  args.output_dir.mkdir(parents=True, exist_ok=True)

  def dump(name: str, rows: list[dict[str, object]]) -> None:
    with (args.output_dir / name).open("w", encoding="utf-8") as stream:
      for row in rows:
        stream.write(json.dumps(row, sort_keys=True) + "\n")

  dump("filtered_all.jsonl", accepted)
  dump("train.jsonl", [r for r in accepted if r["split"] == "train"])
  dump("validation.jsonl", [r for r in accepted if r["split"] == "validation"])
  dump("rejected_kinematic.jsonl", rejected)

  durations = [float(r["duration_s"]) for r in accepted]
  summary = {
    "configuration": {**asdict(cfg), "validation_fraction": args.validation_fraction},
    "groups_kept": sorted(keep),
    "counts": {
      "considered": len(records),
      "accepted": len(accepted),
      "rejected": len(rejected),
      "train": sum(r["split"] == "train" for r in accepted),
      "validation": sum(r["split"] == "validation" for r in accepted),
      "mirrors": sum(bool(r["is_mirror"]) for r in accepted),
    },
    "group_counts": dict(sorted(Counter(r["ballet_group"] for r in accepted).items())),
    "rejection_reasons": dict(
      Counter(
        re.sub(r"[-\d.]+\s*\w*/?\w*", "", str(r["reason"])).strip() for r in rejected
      ).most_common()
    ),
    "duration_s": {
      "total": round(sum(durations), 1),
      "mean": round(float(np.mean(durations)), 2) if durations else 0.0,
      "min": round(min(durations), 2) if durations else 0.0,
      "max": round(max(durations), 2) if durations else 0.0,
    },
  }
  with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
