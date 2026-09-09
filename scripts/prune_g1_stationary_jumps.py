#!/usr/bin/env python3
"""Prune paired stationary jumps from a converted G1 training library."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from safe_mimic.motions.g1_dataset import is_stationary_jump_motion
from safe_mimic.motions.g1_size_sample import grouped_distribution

DEFAULT_LIBRARY = Path("artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
  parser.add_argument("--max-planar-excursion-m", type=float, default=0.35)
  parser.add_argument(
    "--apply",
    action="store_true",
    help="Rewrite the active manifests; otherwise perform a dry run.",
  )
  return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  with path.open(encoding="utf-8") as stream:
    return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8") as stream:
    for record in records:
      stream.write(json.dumps(record, sort_keys=True) + "\n")
  temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
  temporary.replace(path)


def _npz_path(library: Path, record: Mapping[str, Any]) -> Path:
  if record.get("npz_path"):
    return Path(str(record["npz_path"])).expanduser().resolve()
  csv_path = Path(str(record["csv_path"]))
  if csv_path.parts[:2] == ("g1", "csv"):
    csv_path = Path(*csv_path.parts[2:])
  return library / "npz_50hz" / csv_path.with_suffix(".npz")


def _planar_excursion_m(root_pos_w: np.ndarray) -> float:
  offset = root_pos_w[:, :2] - root_pos_w[0, :2]
  return float(np.linalg.norm(offset, axis=1).max())


def _update_summary(
  summary: dict[str, Any],
  retained_originals: Sequence[Mapping[str, Any]],
  retained_mirrors: Sequence[Mapping[str, Any]],
  *,
  excluded_pairs: int,
  max_planar_excursion_m: float,
) -> None:
  original_bytes = sum(int(row["tracker_payload_bytes"]) for row in retained_originals)
  mirror_bytes = sum(int(row["tracker_payload_bytes"]) for row in retained_mirrors)
  original_frames = sum(int(row["tracker_frame_count"]) for row in retained_originals)
  pool_bytes = Counter()
  for row in retained_originals:
    pool_bytes[str(row["sampling_pool"])] += int(row["tracker_payload_bytes"])

  sample = summary["sample"]
  sample.update(
    {
      "pairs": len(retained_originals),
      "total_clips": len(retained_originals) + len(retained_mirrors),
      "original_frames_50hz": original_frames,
      "original_hours": original_frames / 50.0 / 3600.0,
      "original_payload_bytes": original_bytes,
      "original_payload_gib": original_bytes / 1024**3,
      "mirror_payload_bytes": mirror_bytes,
      "mirror_payload_gib": mirror_bytes / 1024**3,
      "combined_payload_bytes": original_bytes + mirror_bytes,
      "combined_payload_gib": (original_bytes + mirror_bytes) / 1024**3,
      "dance_pairs": sum(bool(row["is_dance"]) for row in retained_originals),
      "train_pairs": sum(row["split"] == "train" for row in retained_originals),
      "validation_pairs": sum(
        row["split"] == "validation" for row in retained_originals
      ),
      "pool_payload_shares": {
        name: value / original_bytes for name, value in sorted(pool_bytes.items())
      },
    }
  )
  summary["sample_distribution"] = {
    "package": grouped_distribution(retained_originals, "package"),
    "category": grouped_distribution(retained_originals, "category"),
  }
  if "distribution_fidelity" in summary:
    summary["distribution_fidelity_before_stationary_jump_pruning"] = summary.pop(
      "distribution_fidelity"
    )
  summary["stationary_jump_pruning"] = {
    "excluded_pairs": excluded_pairs,
    "excluded_clips": excluded_pairs * 2,
    "jump_like_max_planar_excursion_m": max_planar_excursion_m,
    "explicit_in_place_metadata_always_excluded": True,
  }


def main() -> None:
  args = _parse_args()
  library = args.library.expanduser().resolve()
  manifest_path = library / "conversion_manifest.jsonl"
  records = _load_jsonl(manifest_path)
  originals = [row for row in records if row["pair_role"] == "original"]
  mirrors = [row for row in records if row["pair_role"] == "mirror"]
  if len(originals) != len(mirrors):
    raise ValueError("motion library must contain one original and one mirror per pair")

  mirror_pair_ids = {str(row["pair_id"]) for row in mirrors}
  excluded: dict[str, tuple[float, dict[str, Any]]] = {}
  for index, record in enumerate(originals, 1):
    pair_id = str(record["pair_id"])
    if pair_id not in mirror_pair_ids:
      raise ValueError(f"missing mirror for {pair_id}")
    path = _npz_path(library, record)
    with np.load(path) as data:
      root_pos_w = np.asarray(data["body_pos_w"])[:, 0]
    if is_stationary_jump_motion(
      record,
      root_pos_w,
      max_planar_excursion_m=args.max_planar_excursion_m,
    ):
      excluded[pair_id] = (_planar_excursion_m(root_pos_w), record)
    if index % 2000 == 0 or index == len(originals):
      print(
        f"Scanned {index:,}/{len(originals):,} pairs; marked {len(excluded):,}",
        flush=True,
      )

  excluded_ids = set(excluded)
  retained = [row for row in records if str(row["pair_id"]) not in excluded_ids]
  retained_originals = [
    row for row in originals if str(row["pair_id"]) not in excluded_ids
  ]
  retained_mirrors = [row for row in mirrors if str(row["pair_id"]) not in excluded_ids]
  excluded_clips = [row for row in records if str(row["pair_id"]) in excluded_ids]
  report = {
    "library": str(library),
    "max_planar_excursion_m": args.max_planar_excursion_m,
    "excluded_pairs": len(excluded),
    "excluded_clips": len(excluded_clips),
    "retained_pairs": len(retained_originals),
    "retained_clips": len(retained),
    "excluded_train_pairs": sum(
      record["split"] == "train" for _, record in excluded.values()
    ),
    "excluded_validation_pairs": sum(
      record["split"] == "validation" for _, record in excluded.values()
    ),
    "excluded_pool_pairs": dict(
      Counter(record["sampling_pool"] for _, record in excluded.values())
    ),
  }
  print(json.dumps(report, indent=2, sort_keys=True))
  if not args.apply or not excluded:
    return

  audit_records = []
  for record in excluded_clips:
    excursion, _ = excluded[str(record["pair_id"])]
    audit_records.append(
      {
        **record,
        "stationary_jump_planar_excursion_m": excursion,
        "stationary_jump_filter_threshold_m": args.max_planar_excursion_m,
      }
    )
  _write_jsonl_atomic(library / "excluded_stationary_jumps.jsonl", audit_records)
  _write_jsonl_atomic(manifest_path, retained)
  _write_jsonl_atomic(library / "originals.jsonl", retained_originals)
  _write_jsonl_atomic(library / "mirrors.jsonl", retained_mirrors)

  yaml_path = library / "npz_manifest_50hz.yaml"
  yaml_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
  excluded_files = {
    str(_npz_path(library, record).relative_to(library / "npz_50hz"))
    for record in excluded_clips
  }
  yaml_config["motions"] = [
    motion
    for motion in yaml_config["motions"]
    if str(motion["file"]) not in excluded_files
  ]
  yaml_temporary = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
  with yaml_temporary.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(yaml_config, stream, sort_keys=False)
  yaml_temporary.replace(yaml_path)

  summary_path = library / "selection_summary.json"
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  _update_summary(
    summary,
    retained_originals,
    retained_mirrors,
    excluded_pairs=len(excluded),
    max_planar_excursion_m=args.max_planar_excursion_m,
  )
  _write_json_atomic(summary_path, summary)
  _write_json_atomic(library / "stationary_jump_pruning_summary.json", report)


if __name__ == "__main__":
  main()
