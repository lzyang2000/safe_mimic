#!/usr/bin/env python3
"""Trim leading/trailing standing stance from every clip of a motion manifest.

Reads a YAML manifest (``fps``, ``root_path``, ``motions[].file``), writes
trimmed NPZs with the same relative paths under ``--output-root``, a new
manifest pointing at them, and a JSON summary of what was cut. Clips with no
detectable motion are copied untouched. Defaults keep about one second of
stance on each side (user request, 2026-09-09).

Example::

  uv run python scripts/trim_motion_stance.py \\
    --manifest artifacts/bones-seed/datasets/g1_ballet_v1/ballet.yaml \\
    --output-root artifacts/bones-seed/datasets/g1_ballet_v1_trim1s/npz_50hz \\
    --output-manifest artifacts/bones-seed/datasets/g1_ballet_v1_trim1s/ballet.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from safe_mimic.motions.stance_trim import StanceTrimCfg, trim_motion_arrays


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--output-manifest", type=Path, required=True)
  parser.add_argument("--keep-s", type=float, default=1.0)
  parser.add_argument("--joint-speed-rps", type=float, default=1.0)
  parser.add_argument("--root-speed-mps", type=float, default=0.15)
  parser.add_argument("--smoothing-frames", type=int, default=5)
  parser.add_argument("--min-length-s", type=float, default=2.0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cfg = StanceTrimCfg(
    keep_s=args.keep_s,
    joint_speed_rps=args.joint_speed_rps,
    root_speed_mps=args.root_speed_mps,
    smoothing_frames=args.smoothing_frames,
    min_length_s=args.min_length_s,
  )
  manifest = yaml.safe_load(args.manifest.read_text())
  root = Path(manifest["root_path"])
  if not root.is_absolute():
    root = (args.manifest.parent / root).resolve()
  fps = float(manifest["fps"])
  args.output_root.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, object]] = []
  total_in = total_out = 0
  for entry in manifest["motions"]:
    rel = Path(entry["file"])
    with np.load(root / rel) as data:
      arrays = {k: data[k] for k in data.files}
    n_in = int(arrays["joint_vel"].shape[0])
    trimmed, (start, end) = trim_motion_arrays(arrays, cfg)
    out_path = args.output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **trimmed)
    total_in += n_in
    total_out += end - start
    rows.append(
      {
        "file": str(rel),
        "frames_in": n_in,
        "start": start,
        "end": end,
        "lead_cut_s": start / fps,
        "trail_cut_s": (n_in - end) / fps,
      }
    )
  new_manifest = dict(manifest)
  new_manifest["root_path"] = str(args.output_root.resolve())
  args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
  args.output_manifest.write_text(yaml.safe_dump(new_manifest, sort_keys=False))
  summary = {
    "source_manifest": str(args.manifest),
    "cfg": cfg.__dict__,
    "clips": len(rows),
    "frames_in": total_in,
    "frames_out": total_out,
    "seconds_cut": (total_in - total_out) / fps,
    "clips_touched": sum(
      1 for r in rows if r["start"] > 0 or r["end"] < r["frames_in"]
    ),
    "rows": rows,
  }
  summary_path = args.output_manifest.with_suffix(".trim_summary.json")
  summary_path.write_text(json.dumps(summary, indent=1))
  print(
    f"{summary['clips']} clips, {summary['clips_touched']} trimmed, "
    f"{total_in / fps:.0f} s -> {total_out / fps:.0f} s "
    f"({summary['seconds_cut']:.0f} s of stance removed); "
    f"manifest {args.output_manifest}"
  )


if __name__ == "__main__":
  main()
