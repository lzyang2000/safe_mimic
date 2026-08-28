#!/usr/bin/env python3
"""Benchmark current-frame streaming from a prebuilt human capsule path bank."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from safe_mimic.motions import CapsulePathBank, OnlineCapsulePathSampler


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bank", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=200)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--update-hz", type=float, default=10.0)
  parser.add_argument("--seed", type=int, default=20260826)
  return parser.parse_args()


def _synchronize(device: torch.device) -> None:
  if device.type == "cuda":
    torch.cuda.synchronize(device)


def main() -> None:
  args = parse_args()
  if args.num_envs < 1 or args.iterations < 1 or args.warmup < 0:
    raise ValueError("num-envs and iterations must be positive; warmup non-negative")
  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is not available")

  bank = CapsulePathBank(args.bank)
  sampler = OnlineCapsulePathSampler(
    bank, args.num_envs, device, update_hz=args.update_hz
  )
  rng = np.random.default_rng(args.seed)
  env_ids = np.arange(args.num_envs)
  path_ids = rng.integers(0, len(bank), size=args.num_envs)
  intersection_times = np.full(args.num_envs, 5.0)
  robot_positions = np.zeros((args.num_envs, 3))
  robot_positions[:, 2] = 0.8
  sampler.schedule_intersections(
    env_ids,
    path_ids=path_ids,
    global_intersection_times_s=intersection_times,
    robot_positions_at_intersection_w=robot_positions,
    robot_yaw_at_intersection=rng.uniform(-np.pi, np.pi, args.num_envs),
    robot_path_heading_at_intersection=rng.uniform(-np.pi, np.pi, args.num_envs),
    intersection_phase=rng.uniform(0.35, 0.65, args.num_envs),
    crossing_angle_rad=rng.uniform(np.deg2rad(60), np.deg2rad(120), args.num_envs),
    body_scale_xyz=rng.uniform((0.9, 0.9, 0.92), (1.1, 1.1, 1.08), (args.num_envs, 3)),
    radius_scale=rng.uniform(0.92, 1.08, args.num_envs),
    radius_margin_m=rng.uniform(0.0, 0.025, args.num_envs),
  )

  for index in range(args.warmup):
    sampler.sample_held(5.0 + index / 50.0)
  _synchronize(device)
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
  else:
    baseline_allocated = 0

  start = time.perf_counter()
  poses = None
  for index in range(args.iterations):
    poses = sampler.sample_held(6.0 + index / 50.0)
  _synchronize(device)
  elapsed_s = time.perf_counter() - start
  assert poses is not None

  output_bytes = sum(
    tensor.numel() * tensor.element_size()
    for tensor in (
      poses.centers_w,
      poses.quaternions_wxyz,
      poses.radii_m,
      poses.half_lengths_m,
      poses.root_positions_w,
      poses.active,
    )
  )
  peak_extra_bytes = 0
  if device.type == "cuda":
    peak_extra_bytes = torch.cuda.max_memory_allocated(device) - baseline_allocated
  mean_ms = elapsed_s * 1000.0 / args.iterations
  result = {
    "bank": str(args.bank.resolve()),
    "paths": len(bank),
    "num_envs": args.num_envs,
    "capsules_per_env": len(bank.capsule_names),
    "device": str(device),
    "policy_hz": 50.0,
    "human_update_hz": args.update_hz,
    "iterations": args.iterations,
    "mean_step_ms": mean_ms,
    "steps_per_second": args.iterations / elapsed_s,
    "fraction_of_50hz_budget": mean_ms / 20.0,
    "current_pose_output_mib": output_bytes / 2**20,
    "peak_extra_device_mib": peak_extra_bytes / 2**20,
    "full_bank_pose_storage_mib": int(bank.config["pose_storage_bytes"]) / 2**20,
  }
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
