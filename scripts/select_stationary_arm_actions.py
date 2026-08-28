#!/usr/bin/env python3
"""Select diverse, locally anchored arm actions for annular crowds.

The output is intentionally an index, not another copy of the motion arrays.
It can be passed to ``build_skeleton_path_bank.py`` to produce the small bank
that is copied to VRAM at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safe_mimic.motions import (
  CapsulePathBank,
  extract_transition_features,
  nearest_transition_neighbors,
)

ACTION_KEYWORDS = {
  "combat": (
    "block",
    "box",
    "combat",
    "defend",
    "fight",
    "guard",
    "hook",
    "jab",
    "martial",
    "punch",
    "spar",
    "strike",
    "uppercut",
  ),
  "directional": (
    "beckon",
    "direct",
    "finger gun",
    "point",
    "reach",
    "salute",
    "shoo",
    "signal",
    "stop gesture",
    "wave",
    "welcome",
  ),
  "social": (
    "applause",
    "clap",
    "finger snap",
    "gesture",
    "greet",
    "laugh",
    "no speak",
    "shh",
    "shrug",
    "snap",
  ),
  "self_touch": (
    "bicep",
    "chin",
    "face",
    "forearm",
    "head",
    "itch",
    "mouth",
    "rub",
    "scratch",
    "self",
    "stretch",
    "warm up",
  ),
  "expressive": (
    "angry",
    "celebrat",
    "confus",
    "enthusiastic",
    "exaggerated",
    "flex",
    "frustrat",
    "omg",
    "rage",
    "strength",
    "surpris",
  ),
}

# These clips either require unavailable context or cease to look like a
# standing person in a crowd. Word-boundary matching avoids false positives
# such as "fallback" matching "fall".
REJECT_TERMS = (
  "appliance",
  "bag",
  "ball",
  "bat",
  "book",
  "bottle",
  "burger",
  "button",
  "camera",
  "cartwheel",
  "chair",
  "choreograph",
  "cleaning",
  "crawl",
  "crank",
  "crouch",
  "cup",
  "dance",
  "door",
  "drink",
  "eating",
  "fall",
  "falling",
  "floor",
  "guitar",
  "handstand",
  "handle",
  "handshake",
  "hat",
  "high five",
  "hug",
  "instrument",
  "jump",
  "jumping",
  "kneel",
  "lever",
  "lie down",
  "lying",
  "mirror",
  "object",
  "obstacle",
  "phone",
  "pick up",
  "pistol",
  "prop",
  "pull shoulder",
  "push-up",
  "rifle",
  "shelf",
  "sit",
  "sitting",
  "someone",
  "squat",
  "table",
  "ticket",
  "t-pose",
  "trash",
  "valve",
  "wall",
  "weapon",
  "lasso",
  "other person",
)

OVERHEAD_TERMS = (
  "above the head",
  "arms overhead",
  "arms raised high",
  "hands overhead",
  "hands raised high",
  "overhead arm",
)


@dataclass(frozen=True)
class Candidate:
  path_id: int
  category: str
  move_name: str
  actor: str
  description: str
  max_root_radius_m: float
  hand_excursion_m: float
  duration_s: float
  metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--bank",
    type=Path,
    default=Path("artifacts/bones-seed/datasets/capsule_path_bank_50hz_6s"),
  )
  parser.add_argument(
    "--source-manifest",
    type=Path,
    default=Path(
      "artifacts/bones-seed/datasets/walk_punch_kick_1000/manifest.jsonl"
    ),
  )
  parser.add_argument("--transition-index", type=Path)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(
      "artifacts/bones-seed/datasets/standing_arm_actions_100"
    ),
  )
  parser.add_argument("--count", type=int, default=100)
  parser.add_argument("--neighbors", type=int, default=8)
  parser.add_argument("--max-root-radius", type=float, default=0.25)
  parser.add_argument("--min-hand-excursion", type=float, default=0.18)
  parser.add_argument("--max-segments-per-source", type=int, default=2)
  parser.add_argument("--max-segments-per-actor", type=int, default=5)
  parser.add_argument("--seed", type=int, default=17)
  return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
  with path.open() as source:
    return [json.loads(line) for line in source]


def _contains_term(text: str, terms: tuple[str, ...]) -> bool:
  return any(
    re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) is not None
    for term in terms
  )


def _category(text: str) -> str | None:
  scores = {
    category: sum(keyword in text for keyword in keywords)
    for category, keywords in ACTION_KEYWORDS.items()
  }
  best = max(scores, key=scores.get)
  return best if scores[best] else None


def _stable_noise(seed: int, path_id: int) -> float:
  digest = hashlib.blake2b(
    f"{seed}:{path_id}".encode(), digest_size=8
  ).digest()
  return int.from_bytes(digest, "little") / float(2**64)


def _select_diverse(
  candidates: list[Candidate],
  count: int,
  *,
  max_per_source: int,
  max_per_actor: int,
  seed: int,
) -> list[Candidate]:
  if len(candidates) < count:
    raise ValueError(f"only {len(candidates)} candidates remain for {count} slots")
  by_category: dict[str, list[Candidate]] = defaultdict(list)
  for candidate in candidates:
    by_category[candidate.category].append(candidate)
  for values in by_category.values():
    values.sort(
      key=lambda candidate: (
        -candidate.hand_excursion_m,
        candidate.max_root_radius_m,
        _stable_noise(seed, candidate.path_id),
      )
    )

  category_order = tuple(ACTION_KEYWORDS)
  source_counts: Counter[str] = Counter()
  actor_counts: Counter[str] = Counter()
  selected: list[Candidate] = []
  selected_ids: set[int] = set()
  cursors: Counter[str] = Counter()

  # Round-robin prevents combat-heavy source metadata from swallowing the
  # quieter pointing, clapping, self-touch, and expressive actions.
  while len(selected) < count:
    progress = False
    for category in category_order:
      values = by_category.get(category, [])
      while cursors[category] < len(values):
        candidate = values[cursors[category]]
        cursors[category] += 1
        if candidate.path_id in selected_ids:
          continue
        if source_counts[candidate.move_name] >= max_per_source:
          continue
        if actor_counts[candidate.actor] >= max_per_actor:
          continue
        selected.append(candidate)
        selected_ids.add(candidate.path_id)
        source_counts[candidate.move_name] += 1
        actor_counts[candidate.actor] += 1
        progress = True
        break
      if len(selected) == count:
        break
    if not progress:
      break

  if len(selected) != count:
    counts = Counter(candidate.category for candidate in selected)
    raise ValueError(
      f"diversity caps yielded {len(selected)}/{count} clips; categories={dict(counts)}"
    )
  return selected


def main() -> None:
  args = parse_args()
  if args.count < 1 or args.neighbors < 1:
    raise ValueError("count and neighbors must be positive")
  if args.count <= args.neighbors:
    raise ValueError("count must exceed transition neighbor count")
  if args.max_root_radius <= 0.0 or args.min_hand_excursion <= 0.0:
    raise ValueError("motion thresholds must be positive")
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite existing subset: {args.output}")

  transition_index = args.transition_index or (
    args.bank / "transition_index_v3"
  )
  bank = CapsulePathBank(args.bank)
  paths = _read_jsonl(args.bank / "paths.jsonl")
  if len(paths) != len(bank):
    raise ValueError("path metadata length differs from bank")
  source_records = _read_jsonl(args.source_manifest)
  source_by_name = {str(record["move_name"]): record for record in source_records}
  if len(source_by_name) != len(source_records):
    raise ValueError("source manifest move names must be unique")
  with np.load(transition_index / "edges.npz", allow_pickle=False) as graph:
    accepted_actions = np.asarray(graph["action_path_ids"], dtype=np.int64)
  decisions = _read_jsonl(transition_index / "decisions.jsonl")
  if len(decisions) != len(bank):
    raise ValueError("transition decisions length differs from bank")

  rejected: Counter[str] = Counter()
  candidates: list[Candidate] = []
  for path_id in accepted_actions:
    path = paths[int(path_id)]
    if path["family"] != "punch":
      continue
    source = source_by_name.get(str(path["move_name"]))
    if source is None:
      rejected["missing_source_metadata"] += 1
      continue
    metrics = source.get("metrics", {})
    if bool(metrics.get("has_sustained_overhead_arm", False)):
      rejected["sustained_overhead_arm"] += 1
      continue
    text = " ".join(
      (
        str(path["move_name"]),
        str(path["description"]),
        str(source.get("description", "")),
        str(source.get("category", "")),
        str(source.get("movement_type", "")),
      )
    ).casefold().replace("_", " ")
    if _contains_term(text, REJECT_TERMS):
      rejected["context_or_posture"] += 1
      continue
    if any(term in text for term in OVERHEAD_TERMS):
      rejected["overhead_description"] += 1
      continue
    category = _category(text)
    if category is None:
      rejected["no_explicit_arm_action"] += 1
      continue
    frame_count = int(bank.frame_counts[int(path_id)])
    root = np.asarray(
      bank.root_positions[int(path_id), :frame_count, :2], dtype=np.float32
    )
    max_root_radius = float(
      np.linalg.vector_norm(root - root[0], axis=-1).max()
    )
    if max_root_radius > args.max_root_radius:
      rejected["root_drift"] += 1
      continue
    hand_excursion = float(decisions[int(path_id)]["hand_excursion_m"])
    if hand_excursion < args.min_hand_excursion:
      rejected["weak_arm_action"] += 1
      continue
    candidates.append(
      Candidate(
        path_id=int(path_id),
        category=category,
        move_name=str(path["move_name"]),
        actor=str(source.get("actor", "unknown")),
        description=str(path["description"]),
        max_root_radius_m=max_root_radius,
        hand_excursion_m=hand_excursion,
        duration_s=(frame_count - 1) / bank.fps,
        metadata={"path": path, "source": source},
      )
    )

  selected = _select_diverse(
    candidates,
    args.count,
    max_per_source=args.max_segments_per_source,
    max_per_actor=args.max_segments_per_actor,
    seed=args.seed,
  )
  selected_ids = np.asarray(
    [candidate.path_id for candidate in selected], dtype=np.int64
  )

  # Match end poses directly to other standing actions. The runtime applies
  # skeleton-space inertialization across the selected edge, so there is no
  # walk connector and no teleport at a boundary.
  features = extract_transition_features(bank)
  next_local, next_costs = nearest_transition_neighbors(
    features.exit[selected_ids],
    features.entry[selected_ids],
    args.neighbors + 1,
  )
  next_ids = selected_ids[next_local]
  # Remove the trivial self edge. It is always present in the K+1 search but
  # is not necessarily the first entry when a clip ends far from its start.
  filtered_ids = np.empty((args.count, args.neighbors), dtype=np.int64)
  filtered_costs = np.empty((args.count, args.neighbors), dtype=np.float32)
  for row, path_id in enumerate(selected_ids):
    keep = next_ids[row] != path_id
    available_ids = next_ids[row, keep]
    available_costs = next_costs[row, keep]
    if len(available_ids) < args.neighbors:
      available_ids = next_ids[row, : args.neighbors]
      available_costs = next_costs[row, : args.neighbors]
    filtered_ids[row] = available_ids[: args.neighbors]
    filtered_costs[row] = available_costs[: args.neighbors]

  args.output.mkdir(parents=True)
  np.save(args.output / "action_path_ids.npy", selected_ids)
  np.savez_compressed(
    args.output / "edges.npz",
    accepted_path_ids=np.sort(selected_ids),
    action_path_ids=selected_ids,
    stationary_next_ids=filtered_ids,
    stationary_next_costs=filtered_costs,
  )
  with (args.output / "manifest.jsonl").open("w") as destination:
    for candidate in selected:
      record = {
        "path_id": candidate.path_id,
        "category": candidate.category,
        "move_name": candidate.move_name,
        "actor": candidate.actor,
        "description": candidate.description,
        "max_root_radius_m": candidate.max_root_radius_m,
        "hand_excursion_m": candidate.hand_excursion_m,
        "duration_s": candidate.duration_s,
        "source_path": candidate.metadata["path"]["source_path"],
        "source_start_time_s": candidate.metadata["path"]["source_start_time_s"],
        "source_end_time_s": candidate.metadata["path"]["source_end_time_s"],
      }
      destination.write(json.dumps(record, separators=(",", ":")) + "\n")

  category_counts = Counter(candidate.category for candidate in selected)
  summary = {
    "format_version": 1,
    "count": len(selected),
    "candidate_count": len(candidates),
    "source_motion_count": len({candidate.move_name for candidate in selected}),
    "actor_count": len({candidate.actor for candidate in selected}),
    "categories": dict(sorted(category_counts.items())),
    "thresholds": {
      "max_root_radius_m": args.max_root_radius,
      "min_hand_excursion_m": args.min_hand_excursion,
      "max_segments_per_source": args.max_segments_per_source,
      "max_segments_per_actor": args.max_segments_per_actor,
    },
    "neighbors_per_action": args.neighbors,
    "rejected": dict(sorted(rejected.items())),
    "source_bank": str(args.bank.resolve()),
    "source_transition_index": str(transition_index.resolve()),
    "seed": args.seed,
  }
  (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps({**summary, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
  main()
