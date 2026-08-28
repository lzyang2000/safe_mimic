"""Deterministic size-budgeted sampling for paired BONES-SEED G1 motions."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

TRACKER_FLOAT32_BYTES_PER_FRAME = 1792
TRACKER_FPS_BYTES_PER_CLIP = 8


def tracker_frame_count(
  native_frame_count: int,
  *,
  native_fps: float = 120.0,
  target_fps: float = 50.0,
) -> int:
  """Return the frame count produced by the tracker NPZ converter."""
  if native_frame_count < 2:
    raise ValueError("native_frame_count must be at least two")
  if native_fps <= 0.0 or target_fps <= 0.0:
    raise ValueError("frame rates must be positive")
  duration_s = (native_frame_count - 1) / native_fps
  return max(2, math.floor(duration_s * target_fps) + 1)


def tracker_float32_payload_bytes(frame_count: int) -> int:
  """Return bytes used by one fully materialized mjlab tracker motion."""
  if frame_count < 1:
    raise ValueError("frame_count must be positive")
  return (
    frame_count * TRACKER_FLOAT32_BYTES_PER_FRAME
    + TRACKER_FPS_BYTES_PER_CLIP
  )


def _stable_key(seed: str, identifier: str) -> bytes:
  return hashlib.sha256(f"{seed}:{identifier}".encode()).digest()


def stratified_payload_sample(
  records: Sequence[Mapping[str, Any]],
  target_bytes: int,
  *,
  seed: str,
  identifier_field: str = "csv_path",
  payload_field: str = "tracker_payload_bytes",
  stratum_fields: tuple[str, ...] = ("package", "category"),
) -> list[Mapping[str, Any]]:
  """Sample close to a byte budget while preserving each stratum's byte share.

  Each package/category stratum receives a payload quota proportional to its
  share of the complete source set. Motions within a stratum are ordered by a
  stable seeded hash. The closer of the two prefix endpoints around the quota
  is retained, making the result deterministic and insensitive to input order.
  """
  if target_bytes < 1:
    raise ValueError("target_bytes must be positive")
  if not records:
    raise ValueError("records must not be empty")

  by_stratum: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
  for record in records:
    payload = int(record[payload_field])
    if payload < 1:
      raise ValueError("every record must have a positive payload")
    identifier = str(record[identifier_field])
    if not identifier:
      raise ValueError("every record must have an identifier")
    stratum = tuple(str(record.get(field, "")) for field in stratum_fields)
    by_stratum[stratum].append(record)

  source_bytes = sum(int(record[payload_field]) for record in records)
  if target_bytes >= source_bytes:
    return list(records)

  selected: list[Mapping[str, Any]] = []
  for stratum in sorted(by_stratum):
    candidates = sorted(
      by_stratum[stratum],
      key=lambda record: _stable_key(seed, str(record[identifier_field])),
    )
    stratum_bytes = sum(int(record[payload_field]) for record in candidates)
    quota = target_bytes * stratum_bytes / source_bytes
    prefix: list[Mapping[str, Any]] = []
    prefix_bytes = 0
    for record in candidates:
      if prefix_bytes >= quota:
        break
      prefix.append(record)
      prefix_bytes += int(record[payload_field])
    if prefix and abs(quota - (prefix_bytes - int(prefix[-1][payload_field]))) < abs(
      quota - prefix_bytes
    ):
      prefix.pop()
    selected.extend(prefix)
  return selected


def grouped_distribution(
  records: Sequence[Mapping[str, Any]],
  field: str,
  *,
  payload_field: str = "tracker_payload_bytes",
  frame_field: str = "tracker_frame_count",
) -> dict[str, dict[str, float | int]]:
  """Summarize count, frames, payload, and shares by one metadata field."""
  grouped: dict[str, dict[str, float | int]] = defaultdict(
    lambda: {"clips": 0, "frames": 0, "payload_bytes": 0}
  )
  for record in records:
    bucket = grouped[str(record.get(field, ""))]
    bucket["clips"] = int(bucket["clips"]) + 1
    bucket["frames"] = int(bucket["frames"]) + int(record[frame_field])
    bucket["payload_bytes"] = int(bucket["payload_bytes"]) + int(
      record[payload_field]
    )

  total_clips = sum(int(bucket["clips"]) for bucket in grouped.values())
  total_payload = sum(int(bucket["payload_bytes"]) for bucket in grouped.values())
  result: dict[str, dict[str, float | int]] = {}
  for name in sorted(grouped):
    bucket = grouped[name]
    result[name] = {
      **bucket,
      "clip_share": int(bucket["clips"]) / total_clips,
      "payload_share": int(bucket["payload_bytes"]) / total_payload,
    }
  return result


__all__ = [
  "TRACKER_FLOAT32_BYTES_PER_FRAME",
  "TRACKER_FPS_BYTES_PER_CLIP",
  "grouped_distribution",
  "stratified_payload_sample",
  "tracker_float32_payload_bytes",
  "tracker_frame_count",
]
