#!/usr/bin/env python3
"""Prune a capsule bank and build PHP-style locomotion/action graph edges."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from safe_mimic.motions import (
  CapsulePathBank,
  extract_transition_features,
  nearest_transition_neighbors,
)

PROP_KEYWORDS = (
  " dog",
  "crutch",
  "phone",
  "mobile",
  " object",
  " item",
  " box",
  "broom",
  " cart",
  "suitcase",
  "umbrella",
  "weapon",
  "sword",
  " stick",
  " bat",
  "chair",
  "door",
  "table",
  " bag",
  "bottle",
  " cup",
  " book",
  "holding a",
  "holding the",
  "holds a",
  "carrying",
  "carries",
  "picks up",
  "put down",
  "watering",
  "crate",
  "cigarette",
  "smoking",
  "camera",
  "microphone",
  " tray",
  " plate",
  " food",
)

ACTION_KEYWORDS = {
  "punch": (
    "punch",
    "jab",
    "strike",
    " hit",
    "fight",
    "boxing",
    "boxer",
    "point",
    "reach",
    "push",
    "throw",
    "block",
    "wave",
    "swing",
    " arm",
    " hand",
    "elbow",
    "gesture",
    "extend",
  ),
  "kick": (
    "kick",
    " leg",
    " foot",
    "stomp",
    "lunge",
    "jump",
    "knee",
    "squat",
    "crouch",
    "combat",
    "martial",
    "attack",
  ),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--bank",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/capsule_path_bank_50hz_6s"),
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument("--neighbors", type=int, default=8)
  parser.add_argument("--max-neighbor-cost", type=float, default=0.4)
  parser.add_argument("--min-duration", type=float, default=0.5)
  parser.add_argument("--min-walk-path", type=float, default=0.25)
  parser.add_argument("--min-hand-excursion", type=float, default=0.15)
  parser.add_argument("--min-foot-excursion", type=float, default=0.15)
  parser.add_argument("--allow-props", action="store_true")
  parser.add_argument("--allow-unlabeled-actions", action="store_true")
  return parser.parse_args()


def _load_metadata(path: Path, expected_count: int) -> list[dict[str, object]]:
  with path.open() as source:
    records = [json.loads(line) for line in source]
  if len(records) != expected_count:
    raise ValueError("path metadata length differs from bank path count")
  if any(record["path_id"] != index for index, record in enumerate(records)):
    raise ValueError("path metadata is not ordered by path_id")
  return records


def _has_prop(record: dict[str, object]) -> bool:
  text = f" {record['move_name']} {record['description']}".casefold()
  return any(keyword in text for keyword in PROP_KEYWORDS)


def _has_action_label(record: dict[str, object], family: str) -> bool:
  text = f" {record['move_name']} {record['description']}".casefold()
  return any(keyword in text for keyword in ACTION_KEYWORDS[family])


def main() -> None:
  args = parse_args()
  if args.neighbors < 1:
    raise ValueError("neighbors must be positive")
  if args.max_neighbor_cost <= 0.0 or args.min_duration <= 0.0:
    raise ValueError("cost and duration thresholds must be positive")
  output = args.output or (args.bank / "transition_index_v3")
  if output.exists():
    raise FileExistsError(f"refusing to overwrite existing index: {output}")

  bank = CapsulePathBank(args.bank)
  records = _load_metadata(args.bank / "paths.jsonl", len(bank))
  features = extract_transition_features(bank)
  duration_s = (np.asarray(bank.frame_counts) - 1) / bank.fps
  reasons: list[list[str]] = [[] for _ in range(len(bank))]
  for path_id, record in enumerate(records):
    family = str(record["family"])
    if duration_s[path_id] < args.min_duration:
      reasons[path_id].append("too_short")
    if not features.upright_boundary[path_id]:
      reasons[path_id].append("non_upright_boundary")
    if not features.supported_boundary[path_id]:
      reasons[path_id].append("unsupported_boundary")
    if family == "walk" and features.root_path_length_m[path_id] < args.min_walk_path:
      reasons[path_id].append("insufficient_root_path")
    if (
      family == "punch" and features.hand_excursion_m[path_id] < args.min_hand_excursion
    ):
      reasons[path_id].append("insufficient_hand_excursion")
    if (
      family == "kick" and features.foot_excursion_m[path_id] < args.min_foot_excursion
    ):
      reasons[path_id].append("insufficient_foot_excursion")
    if not args.allow_props and _has_prop(record):
      reasons[path_id].append("prop_dependent")
    if (
      family in ACTION_KEYWORDS
      and not args.allow_unlabeled_actions
      and not _has_action_label(record, family)
    ):
      reasons[path_id].append("unlabeled_action")

  families = np.asarray([record["family"] for record in records])
  initially_valid = np.asarray([not path_reasons for path_reasons in reasons])
  connector_ids = np.flatnonzero(initially_valid & (families == "walk"))
  action_ids = np.flatnonzero(initially_valid & (families != "walk"))
  if len(connector_ids) < args.neighbors:
    raise ValueError("too few valid walking connectors for requested graph degree")

  entry_local_ids, entry_costs = nearest_transition_neighbors(
    features.entry[action_ids],
    features.exit[connector_ids],
    args.neighbors,
  )
  exit_local_ids, exit_costs = nearest_transition_neighbors(
    features.exit[action_ids],
    features.entry[connector_ids],
    args.neighbors,
  )
  entry_connector_ids = connector_ids[entry_local_ids]
  exit_connector_ids = connector_ids[exit_local_ids]
  action_graph_valid = (entry_costs[:, -1] <= args.max_neighbor_cost) & (
    exit_costs[:, -1] <= args.max_neighbor_cost
  )
  for local_id in np.flatnonzero(~action_graph_valid):
    path_id = int(action_ids[local_id])
    if entry_costs[local_id, -1] > args.max_neighbor_cost:
      reasons[path_id].append("insufficient_entry_edges")
    if exit_costs[local_id, -1] > args.max_neighbor_cost:
      reasons[path_id].append("insufficient_exit_edges")

  accepted_action_ids = action_ids[action_graph_valid]
  entry_connector_ids = entry_connector_ids[action_graph_valid]
  entry_costs = entry_costs[action_graph_valid]
  exit_connector_ids = exit_connector_ids[action_graph_valid]
  exit_costs = exit_costs[action_graph_valid]
  walk_local_ids, walk_costs = nearest_transition_neighbors(
    features.exit[connector_ids],
    features.entry[connector_ids],
    args.neighbors,
  )
  walk_next_ids = connector_ids[walk_local_ids]
  accepted_ids = np.sort(np.concatenate((connector_ids, accepted_action_ids)))

  output.mkdir(parents=True)
  np.savez_compressed(
    output / "edges.npz",
    connector_path_ids=connector_ids,
    action_path_ids=accepted_action_ids,
    action_entry_connector_ids=entry_connector_ids,
    action_entry_costs=entry_costs,
    action_exit_connector_ids=exit_connector_ids,
    action_exit_costs=exit_costs,
    walk_next_ids=walk_next_ids,
    walk_next_costs=walk_costs,
    accepted_path_ids=accepted_ids,
  )
  with (output / "decisions.jsonl").open("w") as destination:
    for path_id, record in enumerate(records):
      destination.write(
        json.dumps(
          {
            "path_id": path_id,
            "family": record["family"],
            "move_name": record["move_name"],
            "description": record["description"],
            "accepted": not reasons[path_id],
            "reasons": reasons[path_id],
            "duration_s": float(duration_s[path_id]),
            "root_path_length_m": float(features.root_path_length_m[path_id]),
            "hand_excursion_m": float(features.hand_excursion_m[path_id]),
            "foot_excursion_m": float(features.foot_excursion_m[path_id]),
          },
          separators=(",", ":"),
        )
        + "\n"
      )

  accepted_mask = np.asarray([not value for value in reasons])
  accepted_by_family = {
    family: int(np.sum((families == family) & accepted_mask))
    for family in ("walk", "punch", "kick")
  }
  rejected_reasons = Counter(reason for values in reasons for reason in values)
  summary = {
    "format_version": 1,
    "source_bank": str(args.bank.resolve()),
    "path_count": len(bank),
    "accepted_count": len(accepted_ids),
    "accepted_by_family": accepted_by_family,
    "connector_count": len(connector_ids),
    "action_count": len(accepted_action_ids),
    "neighbors_per_boundary": args.neighbors,
    "max_neighbor_cost": args.max_neighbor_cost,
    "allow_props": args.allow_props,
    "allow_unlabeled_actions": args.allow_unlabeled_actions,
    "thresholds": {
      "min_duration_s": args.min_duration,
      "min_walk_path_m": args.min_walk_path,
      "min_hand_excursion_m": args.min_hand_excursion,
      "min_foot_excursion_m": args.min_foot_excursion,
    },
    "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
  }
  (output / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps({**summary, "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
  main()
