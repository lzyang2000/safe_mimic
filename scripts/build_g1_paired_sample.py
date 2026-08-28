#!/usr/bin/env python3
"""Build a size-budgeted SEED G1 original/mirror sampling manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from safe_mimic.motions.g1_dataset import (
  assign_split,
  is_dance_motion,
  load_metadata_rows,
  metadata_rejection_reason,
  parse_metadata_bool,
)
from safe_mimic.motions.g1_size_sample import (
  grouped_distribution,
  stratified_payload_sample,
  tracker_float32_payload_bytes,
  tracker_frame_count,
)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--dataset-root", type=Path, default=Path("artifacts/bones-seed")
  )
  parser.add_argument("--metadata", type=Path)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--target-gib", type=float, default=5.0)
  parser.add_argument("--native-fps", type=float, default=120.0)
  parser.add_argument("--target-fps", type=float, default=50.0)
  parser.add_argument("--seed", default="20260827")
  parser.add_argument("--validation-fraction", type=float, default=0.1)
  parser.add_argument(
    "--policy",
    choices=("unfiltered", "wbc-dance"),
    default="unfiltered",
  )
  parser.add_argument("--wbc-manifest", type=Path)
  parser.add_argument("--dance-manifest", type=Path)
  parser.add_argument("--wbc-share", type=float, default=0.7)
  return parser


def _mirror_csv_path(path: Path) -> Path:
  return path.with_name(f"{path.stem}_M.csv")


def _original_csv_path(path: Path) -> Path:
  stem = path.stem[:-2] if path.stem.endswith("_M") else path.stem
  return path.with_name(stem).with_suffix(".csv")


def _wbc_original_paths(path: Path) -> set[str]:
  config = yaml.safe_load(path.read_text(encoding="utf-8"))
  originals: set[str] = set()
  for motion in config["motions"]:
    relative = _original_csv_path(Path(str(motion["file"])))
    if relative.parts[:2] != ("g1", "csv"):
      relative = Path("g1/csv") / relative
    originals.add(str(relative))
  return originals


def _accepted_dance_paths(path: Path) -> set[str]:
  dances: set[str] = set()
  with path.open(encoding="utf-8") as stream:
    for line in stream:
      if not line.strip():
        continue
      record = json.loads(line)
      if bool(record.get("is_dance", False)):
        dances.add(str(_original_csv_path(Path(str(record["csv_path"])))))
  return dances


def _output_record(
  metadata: Mapping[str, Any],
  *,
  pair_id: str,
  pair_role: str,
  native_fps: float,
  target_fps: float,
  validation_fraction: float,
) -> dict[str, Any]:
  native_frames = int(metadata["move_duration_frames"])
  frames = tracker_frame_count(
    native_frames,
    native_fps=native_fps,
    target_fps=target_fps,
  )
  return {
    "move_name": metadata.get("move_name", Path(pair_id).stem),
    "csv_path": metadata["move_g1_path"],
    "package": metadata.get("package", ""),
    "category": metadata.get("category", ""),
    "movement": metadata.get("content_type_of_movement", ""),
    "body_position": metadata.get("content_body_position", ""),
    "description": metadata.get("content_natural_desc_1", ""),
    "take_name": metadata.get("take_name", ""),
    "actor_uid": metadata.get("actor_uid", ""),
    "is_dance": is_dance_motion(metadata),
    "split": assign_split(metadata, validation_fraction),
    "pair_id": pair_id,
    "pair_role": pair_role,
    "native_frame_count": native_frames,
    "tracker_frame_count": frames,
    "tracker_payload_bytes": tracker_float32_payload_bytes(frames),
  }


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as stream:
    for record in records:
      stream.write(json.dumps(record, sort_keys=True) + "\n")


def _max_share_error(
  source: Mapping[str, Mapping[str, float | int]],
  sample: Mapping[str, Mapping[str, float | int]],
  share: str,
) -> float:
  names = set(source) | set(sample)
  return max(
    abs(
      float(source.get(name, {}).get(share, 0.0))
      - float(sample.get(name, {}).get(share, 0.0))
    )
    for name in names
  )


def main() -> None:
  args = _parser().parse_args()
  if args.target_gib <= 0.0:
    raise ValueError("target-gib must be positive")
  if not 0.0 < args.validation_fraction < 1.0:
    raise ValueError("validation-fraction must be between zero and one")
  if not 0.0 < args.wbc_share < 1.0:
    raise ValueError("wbc-share must be between zero and one")

  metadata_path = args.metadata or (
    args.dataset_root / "metadata/seed_metadata_v004.csv"
  )
  size_label = f"{args.target_gib:g}".replace(".", "p")
  policy_label = "unfiltered" if args.policy == "unfiltered" else "wbc70_dance30"
  default_output_name = f"g1_{policy_label}_paired_{size_label}g_v1"
  output_dir = args.output_dir or (
    args.dataset_root / f"datasets/{default_output_name}"
  )
  rows = list(load_metadata_rows(metadata_path))
  metadata_by_path = {
    str(row.get("move_g1_path", "")): row
    for row in rows
    if row.get("move_g1_path")
  }

  candidates: list[dict[str, Any]] = []
  originals_without_mirror = 0
  for row in rows:
    if parse_metadata_bool(row.get("is_mirror", False)):
      continue
    relative = Path(str(row.get("move_g1_path", "")))
    if not relative.parts or not (args.dataset_root / relative).is_file():
      continue
    mirror = _mirror_csv_path(relative)
    if (
      not (args.dataset_root / mirror).is_file()
      or str(mirror) not in metadata_by_path
    ):
      originals_without_mirror += 1
      continue
    candidates.append(
      _output_record(
        row,
        pair_id=str(relative),
        pair_role="original",
        native_fps=args.native_fps,
        target_fps=args.target_fps,
        validation_fraction=args.validation_fraction,
      )
    )

  target_bytes = round(args.target_gib * 1024**3)
  if args.policy == "unfiltered":
    source_pools = {"unfiltered": candidates}
    target_by_pool = {"unfiltered": target_bytes}
  else:
    wbc_manifest = args.wbc_manifest or (
      Path.home()
      / "twist2/seed_g1_cbf_standing_payload/seed_dataset_filtered.yaml"
    )
    dance_manifest = args.dance_manifest or (
      args.dataset_root / "datasets/g1_general_mimic_v1/filtered_all.jsonl"
    )
    wbc_paths = _wbc_original_paths(wbc_manifest)
    dance_paths = _accepted_dance_paths(dance_manifest)
    source_pools = {
      "wbc": [
        record
        for record in candidates
        if record["csv_path"] in wbc_paths
        and record["csv_path"] not in dance_paths
      ],
      "dance": [
        record for record in candidates if record["csv_path"] in dance_paths
      ],
    }
    wbc_target = round(target_bytes * args.wbc_share)
    target_by_pool = {
      "wbc": wbc_target,
      "dance": target_bytes - wbc_target,
    }

  originals = []
  selected_by_pool: dict[str, list[Mapping[str, Any]]] = {}
  for pool_name, pool_candidates in source_pools.items():
    selected = list(
      stratified_payload_sample(
        pool_candidates,
        target_by_pool[pool_name],
        seed=f"{args.seed}:{pool_name}",
      )
    )
    for record in selected:
      record["sampling_pool"] = pool_name
    originals.extend(selected)
    selected_by_pool[pool_name] = selected
  originals.sort(key=lambda record: str(record["csv_path"]))

  mirrors: list[dict[str, Any]] = []
  for original in originals:
    mirror_path = _mirror_csv_path(Path(str(original["csv_path"])))
    mirror_metadata = metadata_by_path[str(mirror_path)]
    mirror_record = _output_record(
      mirror_metadata,
      pair_id=str(original["pair_id"]),
      pair_role="mirror",
      native_fps=args.native_fps,
      target_fps=args.target_fps,
      validation_fraction=args.validation_fraction,
    )
    mirror_record["sampling_pool"] = original["sampling_pool"]
    mirrors.append(mirror_record)

  combined = [
    record
    for pair in zip(originals, mirrors, strict=True)
    for record in pair
  ]
  source_package = grouped_distribution(candidates, "package")
  sample_package = grouped_distribution(originals, "package")
  source_category = grouped_distribution(candidates, "category")
  sample_category = grouped_distribution(originals, "category")
  rejection_counts = Counter(
    metadata_rejection_reason(metadata_by_path[str(record["csv_path"])]) or "accepted"
    for record in originals
  )

  original_bytes = sum(int(record["tracker_payload_bytes"]) for record in originals)
  mirror_bytes = sum(int(record["tracker_payload_bytes"]) for record in mirrors)
  original_frames = sum(int(record["tracker_frame_count"]) for record in originals)
  pool_summary = {
    pool_name: {
      "source_pairs": len(source_pools[pool_name]),
      "selected_pairs": len(selected_by_pool[pool_name]),
      "target_payload_bytes": target_by_pool[pool_name],
      "selected_payload_bytes": sum(
        int(record["tracker_payload_bytes"])
        for record in selected_by_pool[pool_name]
      ),
    }
    for pool_name in source_pools
  }
  distribution_fidelity: dict[str, float] = {}
  for pool_name in source_pools:
    source_pool_package = grouped_distribution(source_pools[pool_name], "package")
    sample_pool_package = grouped_distribution(
      selected_by_pool[pool_name], "package"
    )
    source_pool_category = grouped_distribution(source_pools[pool_name], "category")
    sample_pool_category = grouped_distribution(
      selected_by_pool[pool_name], "category"
    )
    prefix = "" if args.policy == "unfiltered" else f"{pool_name}_"
    distribution_fidelity[f"{prefix}max_package_payload_share_error"] = (
      _max_share_error(source_pool_package, sample_pool_package, "payload_share")
    )
    distribution_fidelity[f"{prefix}max_category_payload_share_error"] = (
      _max_share_error(source_pool_category, sample_pool_category, "payload_share")
    )
    distribution_fidelity[f"{prefix}max_package_clip_share_error"] = (
      _max_share_error(source_pool_package, sample_pool_package, "clip_share")
    )
    distribution_fidelity[f"{prefix}max_category_clip_share_error"] = (
      _max_share_error(source_pool_category, sample_pool_category, "clip_share")
    )

  summary = {
    "configuration": {
      "seed": args.seed,
      "policy": args.policy,
      "wbc_share": args.wbc_share if args.policy == "wbc-dance" else None,
      "dance_share": 1.0 - args.wbc_share if args.policy == "wbc-dance" else None,
      "target_original_gib": args.target_gib,
      "native_fps": args.native_fps,
      "target_fps": args.target_fps,
      "strata": ["package", "category"],
      "selection": "stable hashed prefix per proportional payload quota",
    },
    "source": {
      "mirror_paired_originals": len(candidates),
      "originals_without_mirror": originals_without_mirror,
      "tracker_payload_bytes": sum(
        int(record["tracker_payload_bytes"]) for record in candidates
      ),
      "pools": pool_summary,
    },
    "sample": {
      "pairs": len(originals),
      "total_clips": len(combined),
      "original_frames_50hz": original_frames,
      "original_hours": original_frames / args.target_fps / 3600.0,
      "original_payload_bytes": original_bytes,
      "original_payload_gib": original_bytes / 1024**3,
      "mirror_payload_bytes": mirror_bytes,
      "mirror_payload_gib": mirror_bytes / 1024**3,
      "combined_payload_bytes": original_bytes + mirror_bytes,
      "combined_payload_gib": (original_bytes + mirror_bytes) / 1024**3,
      "dance_pairs": sum(bool(record["is_dance"]) for record in originals),
      "train_pairs": sum(record["split"] == "train" for record in originals),
      "validation_pairs": sum(
        record["split"] == "validation" for record in originals
      ),
      "metadata_filter_outcomes": dict(rejection_counts.most_common()),
      "pool_payload_shares": {
        pool_name: int(pool["selected_payload_bytes"]) / original_bytes
        for pool_name, pool in pool_summary.items()
      },
    },
    "distribution_fidelity": distribution_fidelity,
    "source_distribution": {
      "package": source_package,
      "category": source_category,
    },
    "sample_distribution": {
      "package": sample_package,
      "category": sample_category,
    },
  }

  output_dir.mkdir(parents=True, exist_ok=True)
  _write_jsonl(output_dir / "originals.jsonl", originals)
  _write_jsonl(output_dir / "mirrors.jsonl", mirrors)
  _write_jsonl(output_dir / "conversion_manifest.jsonl", combined)
  with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
  print(json.dumps(summary["sample"], indent=2, sort_keys=True))
  print(json.dumps(summary["distribution_fidelity"], indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
