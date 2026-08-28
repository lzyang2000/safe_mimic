"""Boundary features and neighbor search for a PHP-style human motion graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safe_mimic.motions.capsule_path_bank import CapsulePathBank


@dataclass(frozen=True)
class TransitionFeatures:
  """Normalized entry/exit descriptors and basic path-quality measurements."""

  entry: np.ndarray
  exit: np.ndarray
  root_path_length_m: np.ndarray
  hand_excursion_m: np.ndarray
  foot_excursion_m: np.ndarray
  upright_boundary: np.ndarray
  supported_boundary: np.ndarray


def _quaternion_z_axis(quaternion: np.ndarray) -> np.ndarray:
  w, x, y, z = np.moveaxis(quaternion, -1, 0)
  return np.stack(
    (
      2.0 * (x * z + w * y),
      2.0 * (y * z - w * x),
      1.0 - 2.0 * (x * x + y * y),
    ),
    axis=-1,
  )


def _to_heading_frame(vectors: np.ndarray, yaw: np.ndarray) -> np.ndarray:
  cosine = np.cos(yaw)
  sine = np.sin(yaw)
  while cosine.ndim < vectors.ndim - 1:
    cosine = cosine[..., None]
    sine = sine[..., None]
  output = vectors.copy()
  output[..., 0] = cosine * vectors[..., 0] + sine * vectors[..., 1]
  output[..., 1] = -sine * vectors[..., 0] + cosine * vectors[..., 1]
  return output


def _boundary_features(
  bank: CapsulePathBank,
  frame: np.ndarray,
  neighbor: np.ndarray,
  *,
  is_entry: bool,
) -> np.ndarray:
  path_ids = np.arange(len(bank))
  names = {name: index for index, name in enumerate(bank.capsule_names)}
  feet = [names["left_foot"], names["right_foot"]]
  hands = [names["left_hand"], names["right_hand"]]
  torso = names["torso_upper"]
  centers = np.asarray(bank.centers[path_ids, frame], dtype=np.float32)
  adjacent = np.asarray(bank.centers[path_ids, neighbor], dtype=np.float32)
  root = np.asarray(bank.root_positions[path_ids, frame], dtype=np.float32)
  adjacent_root = np.asarray(
    bank.root_positions[path_ids, neighbor], dtype=np.float32
  )
  elapsed_s = np.maximum(np.abs(frame - neighbor) / bank.fps, 1e-6)
  if is_entry:
    center_velocity = (adjacent - centers) / elapsed_s[:, None, None]
    root_velocity = (adjacent_root - root) / elapsed_s[:, None]
  else:
    center_velocity = (centers - adjacent) / elapsed_s[:, None, None]
    root_velocity = (root - adjacent_root) / elapsed_s[:, None]
  yaw = np.asarray(bank.facing_yaw[path_ids, frame], dtype=np.float32)
  relative = _to_heading_frame(centers - root[:, None], yaw)
  center_velocity = _to_heading_frame(center_velocity, yaw)
  root_velocity = _to_heading_frame(root_velocity, yaw)
  torso_axis = _to_heading_frame(
    _quaternion_z_axis(
      np.asarray(bank.quaternions[path_ids, frame, torso], dtype=np.float32)
    ),
    yaw,
  )
  contacts = (
    centers[:, feet, 2] - np.asarray(bank.ground_z)[:, None] < 0.12
  ).astype(np.float32)
  return np.concatenate(
    (
      relative[:, feet].reshape(len(bank), -1) / 0.5,
      center_velocity[:, feet].reshape(len(bank), -1) / 2.0,
      relative[:, hands].reshape(len(bank), -1) / 0.75,
      center_velocity[:, hands].reshape(len(bank), -1) / 3.0,
      root_velocity / 1.5,
      torso_axis,
      contacts,
    ),
    axis=-1,
  )


def extract_transition_features(bank: CapsulePathBank) -> TransitionFeatures:
  """Extract matching features and quality gates from every prebuilt path."""

  path_count = len(bank)
  path_ids = np.arange(path_count)
  last = np.asarray(bank.frame_counts, dtype=np.int64) - 1
  entry_neighbor = np.minimum(2, last)
  exit_neighbor = np.maximum(0, last - 2)
  entry = _boundary_features(
    bank, np.zeros(path_count, dtype=np.int64), entry_neighbor, is_entry=True
  )
  exit = _boundary_features(bank, last, exit_neighbor, is_entry=False)

  names = {name: index for index, name in enumerate(bank.capsule_names)}
  torso = names["torso_upper"]
  left_foot = names["left_foot"]
  right_foot = names["right_foot"]
  root_start_z = np.asarray(bank.root_positions[:, 0, 2])
  root_end_z = np.asarray(bank.root_positions[path_ids, last, 2])
  torso_start_z = _quaternion_z_axis(
    np.asarray(bank.quaternions[:, 0, torso])
  )[:, 2]
  torso_end_z = _quaternion_z_axis(
    np.asarray(bank.quaternions[path_ids, last, torso])
  )[:, 2]
  upright = (
    (root_start_z > 0.75)
    & (root_start_z < 1.25)
    & (root_end_z > 0.75)
    & (root_end_z < 1.25)
    & (torso_start_z > 0.7)
    & (torso_end_z > 0.7)
  )
  foot_start_z = np.minimum(
    np.asarray(bank.centers[:, 0, left_foot, 2]),
    np.asarray(bank.centers[:, 0, right_foot, 2]),
  )
  foot_end_z = np.minimum(
    np.asarray(bank.centers[path_ids, last, left_foot, 2]),
    np.asarray(bank.centers[path_ids, last, right_foot, 2]),
  )
  supported = (
    (foot_start_z - np.asarray(bank.ground_z) < 0.18)
    & (foot_end_z - np.asarray(bank.ground_z) < 0.18)
  )

  root_path_length = np.zeros(path_count, dtype=np.float32)
  hand_excursion = np.zeros(path_count, dtype=np.float32)
  foot_excursion = np.zeros(path_count, dtype=np.float32)
  for path_id, frame_count in enumerate(bank.frame_counts):
    frame_count = int(frame_count)
    root = np.asarray(bank.root_positions[path_id, :frame_count], dtype=np.float32)
    centers = np.asarray(bank.centers[path_id, :frame_count], dtype=np.float32)
    root_path_length[path_id] = np.linalg.norm(
      np.diff(root[:, :2], axis=0), axis=-1
    ).sum()
    hand_excursion[path_id] = max(
      np.linalg.norm(
        np.ptp(centers[:, names[hand], :2] - centers[:, torso, :2], axis=0)
      )
      for hand in ("left_hand", "right_hand")
    )
    foot_excursion[path_id] = max(
      np.linalg.norm(
        np.ptp(centers[:, names[foot], :2] - root[:, :2], axis=0)
      )
      for foot in ("left_foot", "right_foot")
    )
  return TransitionFeatures(
    entry=entry,
    exit=exit,
    root_path_length_m=root_path_length,
    hand_excursion_m=hand_excursion,
    foot_excursion_m=foot_excursion,
    upright_boundary=upright,
    supported_boundary=supported,
  )


def nearest_transition_neighbors(
  query: np.ndarray,
  reference: np.ndarray,
  neighbor_count: int,
  *,
  chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
  """Return K nearest normalized-RMS transition matches without SciPy."""

  if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
    raise ValueError("query and reference must be 2D with equal feature widths")
  if not 1 <= neighbor_count <= len(reference):
    raise ValueError("neighbor_count must be within the reference set")
  if chunk_size < 1:
    raise ValueError("chunk_size must be positive")
  neighbor_ids = np.empty((len(query), neighbor_count), dtype=np.int64)
  neighbor_costs = np.empty((len(query), neighbor_count), dtype=np.float32)
  for start in range(0, len(query), chunk_size):
    stop = min(start + chunk_size, len(query))
    costs = np.sqrt(
      np.mean((query[start:stop, None] - reference[None]) ** 2, axis=-1)
    )
    partial = np.argpartition(costs, neighbor_count - 1, axis=1)[:, :neighbor_count]
    partial_costs = np.take_along_axis(costs, partial, axis=1)
    order = np.argsort(partial_costs, axis=1)
    neighbor_ids[start:stop] = np.take_along_axis(partial, order, axis=1)
    neighbor_costs[start:stop] = np.take_along_axis(partial_costs, order, axis=1)
  return neighbor_ids, neighbor_costs
