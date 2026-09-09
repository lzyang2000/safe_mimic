#!/usr/bin/env python3
"""Load the packed G1 motion bank and benchmark batched frame sampling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from safe_mimic.motions.adaptive_motion_sampling import AdaptiveMotionSampler
from safe_mimic.motions.packed_npz_motion_lib import PackedNpzMotionLib

G1_TRACKED_BODY_INDEXES = (0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--manifest",
    type=Path,
    default=Path(
      "artifacts/bones-seed/datasets/"
      "g1_wbc70_dance30_paired_10g_v1/conversion_manifest.jsonl"
    ),
  )
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--batch-size", type=int, default=4096)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--iterations", type=int, default=200)
  parser.add_argument("--split", choices=("train", "validation"))
  return parser


def _cuda_memory() -> dict[str, int]:
  return {
    "allocated_bytes": torch.cuda.memory_allocated(),
    "reserved_bytes": torch.cuda.memory_reserved(),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
  }


def main() -> None:
  args = _parser().parse_args()
  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but no CUDA device is available")
  if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
    raise ValueError(
      "batch-size and iterations must be positive; warmup cannot be negative"
    )

  if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = _cuda_memory()
    torch.cuda.synchronize()
  else:
    baseline = {}
  start = time.perf_counter()
  library = PackedNpzMotionLib(
    args.manifest,
    G1_TRACKED_BODY_INDEXES,
    device=device,
    splits=args.split,
  )
  if device.type == "cuda":
    torch.cuda.synchronize()
  load_seconds = time.perf_counter() - start
  loaded_memory = _cuda_memory() if device.type == "cuda" else {}

  def uniform_sample_once() -> None:
    motion_ids = library.sample_motions(args.batch_size)
    motion_times = library.sample_time(motion_ids)
    library.get_frame(motion_ids, motion_times)

  for _ in range(args.warmup):
    uniform_sample_once()
  if device.type == "cuda":
    torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(args.iterations):
    uniform_sample_once()
  if device.type == "cuda":
    torch.cuda.synchronize()
  elapsed = time.perf_counter() - start

  adaptive_sampler = AdaptiveMotionSampler(library)

  def adaptive_sample_once() -> None:
    sample = adaptive_sampler.sample(args.batch_size)
    library.get_frame(sample.motion_ids, sample.motion_times)

  for _ in range(args.warmup):
    adaptive_sample_once()
  if device.type == "cuda":
    torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(args.iterations):
    adaptive_sample_once()
  if device.type == "cuda":
    torch.cuda.synchronize()
  adaptive_elapsed = time.perf_counter() - start

  adaptive_sample = adaptive_sampler.sample(args.batch_size)
  if device.type == "cuda":
    torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(args.iterations):
    failures = torch.rand(args.batch_size, device=device) < 0.25
    adaptive_sampler.update(
      failures,
      adaptive_sample.motion_ids,
      adaptive_sample.motion_times,
    )
    adaptive_sample = adaptive_sampler.sample(args.batch_size)
    library.get_frame(adaptive_sample.motion_ids, adaptive_sample.motion_times)
  if device.type == "cuda":
    torch.cuda.synchronize()
  adaptive_update_elapsed = time.perf_counter() - start

  result = {
    "manifest": str(args.manifest.resolve()),
    "device": str(device),
    "motions": library.num_motions(),
    "total_frames": library.total_frames,
    "tracked_bodies": len(G1_TRACKED_BODY_INDEXES),
    "joint_count": 29,
    "resident_tensor_bytes": library.resident_bytes,
    "load_seconds": load_seconds,
    "batch_size": args.batch_size,
    "iterations": args.iterations,
    "uniform_sample_batch_ms": elapsed * 1000.0 / args.iterations,
    "uniform_sampled_frames_per_second": (args.batch_size * args.iterations / elapsed),
    "adaptive_bins": adaptive_sampler.num_bins,
    "adaptive_resident_bytes": adaptive_sampler.resident_bytes,
    "adaptive_sample_batch_ms": adaptive_elapsed * 1000.0 / args.iterations,
    "adaptive_sampled_frames_per_second": (
      args.batch_size * args.iterations / adaptive_elapsed
    ),
    "adaptive_update_sample_batch_ms": (
      adaptive_update_elapsed * 1000.0 / args.iterations
    ),
  }
  if device.type == "cuda":
    result["cuda_baseline"] = baseline
    result["cuda_after_load"] = loaded_memory
    result["cuda_increment"] = {
      key: loaded_memory[key] - baseline[key]
      for key in ("allocated_bytes", "reserved_bytes")
    }
    result["cuda_final"] = _cuda_memory()
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
