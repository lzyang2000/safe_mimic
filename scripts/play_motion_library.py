#!/usr/bin/env python3
"""Play the generalized G1 motion-library policy from a local checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import load_rl_cfg

from safe_mimic.tasks import (
  MOTION_LIBRARY_STATE_EST_TASK_ID,
  MOTION_LIBRARY_TASK_ID,
)
from safe_mimic.tasks.env_cfg import DEFAULT_G1_MOTION_LIBRARY_MANIFEST


def _latest_checkpoint(log_root: Path, task_id: str) -> Path:
  experiment_name = load_rl_cfg(task_id).experiment_name
  candidates = tuple((log_root / experiment_name).glob("*/model_*.pt"))
  if not candidates:
    raise FileNotFoundError(
      "No generalized motion-library checkpoint found under "
      f"{log_root / experiment_name}. Pass one explicitly as the first argument."
    )
  return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "checkpoint",
    nargs="?",
    type=Path,
    help="Checkpoint to play; defaults to the newest local model_*.pt.",
  )
  parser.add_argument(
    "--motion-file",
    type=Path,
    default=DEFAULT_G1_MOTION_LIBRARY_MANIFEST,
    help="Packed manifest or a single tracker NPZ to play.",
  )
  parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="viser")
  parser.add_argument("--device", default=None)
  parser.add_argument(
    "--state-estimation",
    action="store_true",
    help="Play the 160-input state-estimation task instead of the ablation.",
  )
  parser.add_argument("--no-terminations", action="store_true")
  parser.add_argument("--video", action="store_true")
  parser.add_argument("--video-length", type=int, default=200)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  task_id = (
    MOTION_LIBRARY_STATE_EST_TASK_ID
    if args.state_estimation
    else MOTION_LIBRARY_TASK_ID
  )
  checkpoint = (
    args.checkpoint.expanduser().resolve()
    if args.checkpoint is not None
    else _latest_checkpoint(args.log_root.expanduser().resolve(), task_id)
  )
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  motion_file = args.motion_file.expanduser().resolve()
  if not motion_file.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion_file}")

  print(f"[INFO] Playing checkpoint: {checkpoint}")
  print(f"[INFO] Motion source: {motion_file}")
  run_play(
    task_id,
    PlayConfig(
      checkpoint_file=str(checkpoint),
      motion_file=str(motion_file),
      num_envs=args.num_envs,
      device=args.device,
      viewer=args.viewer,
      no_terminations=args.no_terminations,
      video=args.video,
      video_length=args.video_length,
      log_root=str(args.log_root),
    ),
  )


if __name__ == "__main__":
  main()
