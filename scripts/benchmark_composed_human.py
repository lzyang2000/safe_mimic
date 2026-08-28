#!/usr/bin/env python3
"""Smoke-test the CUDA-resident online human motion composer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from safe_mimic.motions import OnlineComposedHumanSampler, SkeletonPathBank


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--bank",
    type=Path,
    default=Path(
      "artifacts/bones-seed/datasets/skeleton_path_bank_transition_v3"
    ),
  )
  parser.add_argument(
    "--transition-index",
    type=Path,
    default=Path(
      "artifacts/bones-seed/datasets/"
      "capsule_path_bank_50hz_6s/transition_index_v3"
    ),
  )
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--device", default="cuda")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.num_envs < 1:
    raise ValueError("num-envs must be positive")
  sampler = OnlineComposedHumanSampler(
    SkeletonPathBank(args.bank), args.num_envs, args.device
  )
  with np.load(args.transition_index / "edges.npz", allow_pickle=False) as graph:
    action_ids = torch.as_tensor(
      np.asarray(graph["action_path_ids"]).copy(), device=args.device
    )
    entry_ids = torch.as_tensor(
      np.asarray(graph["action_entry_connector_ids"]).copy(), device=args.device
    )
    exit_ids = torch.as_tensor(
      np.asarray(graph["action_exit_connector_ids"]).copy(), device=args.device
    )
  count = args.num_envs
  action_rows = torch.randint(len(action_ids), (count,), device=args.device)
  ranks = torch.randint(entry_ids.shape[1], (count,), device=args.device)
  sequences = torch.stack(
    (
      entry_ids[action_rows, ranks],
      action_ids[action_rows],
      exit_ids[action_rows, ranks],
    ),
    dim=-1,
  )
  env_ids = torch.arange(count, device=args.device)
  if torch.device(args.device).type == "cuda":
    torch.cuda.reset_peak_memory_stats(args.device)
  start = time.perf_counter()
  sampler.schedule_intersections(
    env_ids,
    sequence_source_ids=sequences,
    global_intersection_times_s=torch.full((count,), 2.0, device=args.device),
    robot_positions_at_intersection_w=torch.zeros((count, 3), device=args.device),
    robot_yaw_at_intersection=torch.zeros(count, device=args.device),
    robot_path_heading_at_intersection=torch.zeros(count, device=args.device),
  )
  poses = sampler.sample_held(2.0)
  if torch.device(args.device).type == "cuda":
    torch.cuda.synchronize(args.device)
    peak_bytes = torch.cuda.max_memory_allocated(args.device)
  else:
    peak_bytes = 0
  elapsed_s = time.perf_counter() - start
  print(
    json.dumps(
      {
        "num_envs": count,
        "resident_bank_mib": sampler.device_storage_bytes / 2**20,
        "schedule_plus_first_sample_ms": elapsed_s * 1000.0,
        "peak_allocated_mib": peak_bytes / 2**20,
        "active_envs": int(poses.active.sum()),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
