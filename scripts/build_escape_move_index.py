#!/usr/bin/env python3
"""Build the travel-phase index (``escape_moves.json``) for a clip manifest.

Example::

  uv run python scripts/build_escape_move_index.py \\
    --manifest artifacts/bones-seed/datasets/g1_ballet_v1_trim1s_mirror/ballet.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from safe_mimic.motions.escape_move_index import (
  EscapeMoveIndexCfg,
  build_escape_move_index,
)
from safe_mimic.motions.mirror_g1 import G1MirrorSpec


def _sector(direction_b: tuple[float, float]) -> str:
  angle = math.degrees(math.atan2(direction_b[1], direction_b[0]))
  if abs(angle) < 45.0:
    return "front"
  if 45.0 <= angle < 135.0:
    return "left"
  if abs(angle) >= 135.0:
    return "back"
  return "right"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="default: <manifest dir>/escape_moves.json",
  )
  parser.add_argument("--anchor-body", default="torso_link")
  parser.add_argument("--window-s", type=float, default=0.6)
  parser.add_argument("--min-travel-m", type=float, default=0.3)
  parser.add_argument("--lead-s", type=float, default=0.1)
  parser.add_argument("--exit-speed-mps", type=float, default=0.3)
  args = parser.parse_args()
  spec = G1MirrorSpec.from_model(mujoco.MjModel.from_xml_path(str(G1_XML)))
  cfg = EscapeMoveIndexCfg(
    window_s=args.window_s,
    min_travel_m=args.min_travel_m,
    lead_s=args.lead_s,
    exit_speed_mps=args.exit_speed_mps,
  )
  index = build_escape_move_index(
    args.manifest, spec.body_names.index(args.anchor_body), cfg
  )
  output = args.output or args.manifest.with_name("escape_moves.json")
  output.write_text(json.dumps(index, indent=1))
  candidates = [c for c in index["clips"] if c["candidate"]]
  sectors = Counter(_sector(tuple(c["direction_b"])) for c in candidates)
  print(
    f"{len(index['clips'])} clips, {len(candidates)} escape candidates "
    f"{dict(sectors)} -> {output}"
  )


if __name__ == "__main__":
  main()
