#!/usr/bin/env python3
"""Copy a G1 clip manifest and add a left/right mirrored twin of every clip.

Originals are copied under the same relative paths; each mirror is written
beside its source as ``<stem>__mirror.npz`` and gets a manifest entry that
inherits weight/split/metadata plus ``mirrored: true``. Existing dataset
directories are never modified.

Example::

  D=artifacts/bones-seed/datasets
  uv run python scripts/mirror_g1_motion_library.py \\
    --manifest $D/g1_ballet_v1_trim1s/ballet.yaml \\
    --output-root $D/g1_ballet_v1_trim1s_mirror/npz_50hz \\
    --output-manifest $D/g1_ballet_v1_trim1s_mirror/ballet.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np
import yaml
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from safe_mimic.motions.mirror_g1 import G1MirrorSpec, mirror_motion_arrays


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--output-manifest", type=Path, required=True)
  args = parser.parse_args()
  spec = G1MirrorSpec.from_model(mujoco.MjModel.from_xml_path(str(G1_XML)))
  manifest = yaml.safe_load(args.manifest.read_text())
  root = Path(manifest["root_path"])
  if not root.is_absolute():
    root = (args.manifest.parent / root).resolve()
  entries: list[dict] = []
  for entry in manifest["motions"]:
    rel = Path(entry["file"])
    src = root / rel
    dst = args.output_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    entries.append(dict(entry))
    with np.load(src) as data:
      arrays = {k: data[k] for k in data.files}
    mirrored_rel = rel.with_name(rel.stem + "__mirror.npz")
    np.savez_compressed(
      args.output_root / mirrored_rel, **mirror_motion_arrays(arrays, spec)
    )
    twin = dict(entry)
    twin["file"] = str(mirrored_rel)
    twin["mirrored"] = True
    entries.append(twin)
  out = dict(manifest)
  out["root_path"] = str(args.output_root.resolve())
  out["motions"] = entries
  args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
  args.output_manifest.write_text(yaml.safe_dump(out, sort_keys=False))
  print(f"{len(entries)} entries -> {args.output_manifest}")


if __name__ == "__main__":
  main()
