#!/usr/bin/env python3
"""Convert a filtered BONES-SEED G1 JSONL manifest to 50 Hz tracker NPZs."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import yaml
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from safe_mimic.motions.g1_tracker_npz import (
  convert_g1_csv_to_tracker_npz,
  validate_tracker_npz,
)

_MODEL: mujoco.MjModel | None = None
_DATA: mujoco.MjData | None = None


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
  parser.add_argument("--target-fps", type=float, default=50.0)
  parser.add_argument("--native-fps", type=float, default=120.0)
  parser.add_argument("--max-files", type=int, default=-1)
  parser.add_argument("--uncompressed", action="store_true")
  parser.add_argument("--validate-existing", action="store_true")
  return parser


def _init_worker() -> None:
  global _MODEL, _DATA
  _MODEL = mujoco.MjModel.from_xml_path(str(G1_XML))
  _DATA = mujoco.MjData(_MODEL)


def _worker(
  task: tuple[str, str, float, float, bool, bool],
) -> dict[str, Any]:
  csv_path, output_path, native_fps, target_fps, compressed, validate_existing = task
  output = Path(output_path)
  try:
    if output.is_file():
      if validate_existing:
        validate_tracker_npz(output)
      return {"status": "existing", "output": output_path}
    assert _MODEL is not None and _DATA is not None
    arrays = convert_g1_csv_to_tracker_npz(
      _MODEL,
      _DATA,
      csv_path,
      output,
      native_fps=native_fps,
      target_fps=target_fps,
      compressed=compressed,
    )
    return {
      "status": "converted",
      "output": output_path,
      "frames": int(arrays["joint_pos"].shape[0]),
      "bytes": output.stat().st_size,
    }
  except Exception as exc:  # noqa: BLE001 - preserve failure and continue full bank
    return {"status": "error", "output": output_path, "error": str(exc)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  with path.open(encoding="utf-8") as stream:
    return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
  args = _parser().parse_args()
  manifest = args.manifest or (
    args.dataset_root / "datasets/g1_general_mimic_v1/filtered_all.jsonl"
  )
  output_dir = args.output_dir or (
    args.dataset_root / "datasets/g1_general_mimic_v1/npz_50hz"
  )
  records = _load_jsonl(manifest)
  if args.max_files > 0:
    records = records[: args.max_files]
  output_dir.mkdir(parents=True, exist_ok=True)

  tasks: list[tuple[str, str, float, float, bool, bool]] = []
  output_by_move: dict[str, str] = {}
  for record in records:
    csv_relative = Path(str(record["csv_path"]))
    if csv_relative.parts[:2] == ("g1", "csv"):
      relative = Path(*csv_relative.parts[2:]).with_suffix(".npz")
    else:
      relative = csv_relative.with_suffix(".npz")
    output = output_dir / relative
    output_by_move[str(record["move_name"])] = str(output.resolve())
    tasks.append(
      (
        str(args.dataset_root / csv_relative),
        str(output),
        args.native_fps,
        args.target_fps,
        not args.uncompressed,
        args.validate_existing,
      )
    )

  print(f"Converting {len(tasks):,} accepted clips to {args.target_fps:g} Hz NPZ")
  results: list[dict[str, Any]] = []
  with ProcessPoolExecutor(
    max_workers=args.workers,
    initializer=_init_worker,
  ) as executor:
    for index, result in enumerate(executor.map(_worker, tasks, chunksize=4), 1):
      results.append(result)
      if index % 500 == 0 or index == len(tasks):
        errors = sum(item["status"] == "error" for item in results)
        print(f"  {index:,}/{len(tasks):,}: {errors:,} errors", flush=True)

  errors = [result for result in results if result["status"] == "error"]
  converted_records = [
    {
      **record,
      "npz_path": output_by_move[str(record["move_name"])],
    }
    for record in records
    if not any(
      error["output"] == output_by_move[str(record["move_name"])] for error in errors
    )
  ]
  with (output_dir.parent / "npz_manifest_50hz.yaml").open(
    "w", encoding="utf-8"
  ) as stream:
    yaml.safe_dump(
      {
        "fps": args.target_fps,
        "root_path": str(output_dir.resolve()),
        "motions": [
          {
            "file": str(Path(record["npz_path"]).relative_to(output_dir.resolve())),
            "weight": 1.0,
            "split": record["split"],
            "description": record["description"],
            "package": record["package"],
            "category": record["category"],
          }
          for record in converted_records
        ],
      },
      stream,
      sort_keys=False,
    )
  with (output_dir.parent / "npz_conversion_errors.jsonl").open(
    "w", encoding="utf-8"
  ) as stream:
    for error in errors:
      stream.write(json.dumps(error, sort_keys=True) + "\n")

  summary = {
    "requested": len(records),
    "converted": sum(result["status"] == "converted" for result in results),
    "existing": sum(result["status"] == "existing" for result in results),
    "errors": len(errors),
    "new_frames": sum(int(result.get("frames", 0)) for result in results),
    "new_bytes": sum(int(result.get("bytes", 0)) for result in results),
    "target_fps": args.target_fps,
    "compressed": not args.uncompressed,
  }
  with (output_dir.parent / "npz_conversion_summary.json").open(
    "w", encoding="utf-8"
  ) as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
  print(json.dumps(summary, indent=2, sort_keys=True))
  if errors:
    raise SystemExit(f"{len(errors)} conversions failed; see error manifest")


if __name__ == "__main__":
  main()
