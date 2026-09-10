# Escape Moves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the filtered replay command answer a predicted collision by switching the raw clip to a travelling ballet move aligned with the escape direction, then resume the interrupted clip, with the CBF filter still on top.

**Architecture:** Three new pure modules (G1 mirroring, travel index, escape-move planner logic) plus a thin integration into `PlanarFilteredReplayMotionCommand` (per-env state machine, blended raw getters, actor hint). A mirrored copy of the trimmed ballet library gives left/right directional coverage; a JSON sidecar indexes each clip's travel phase. New task `...-Leash-Ballet-Moves`; every existing task is byte-identical because the feature is `None` by default.

**Tech Stack:** Python 3.12, torch, numpy, mujoco (FK verification), mjlab, pytest, ruff (2-space indent, 88 cols).

**Spec:** `docs/superpowers/specs/2026-09-10-escape-moves-design.md`

## Global Constraints

- Branch `escape-moves` in the worktree; never touch `main`. Small commits with the attribution trailer.
- Run python as `/tmp/claude-1000/wt/wpy` (venv python with `PYTHONPATH` = worktree `src`); ruff as `/tmp/claude-1000/wt/ruff`. Tests: `/tmp/claude-1000/wt/wpy -m pytest tests -q -p no:cacheprovider`.
- Never whole-file `ruff format` a file tracked in main (`kinematic_replay_command.py`, `env_cfg.py`, `tasks/__init__.py`, `mdp.py`, `evaluation.py`, the seven scripts, existing tests). New files may be formatted.
- Datasets: write only under `artifacts/bones-seed/datasets/g1_ballet_v1_trim1s_mirror/`. Never modify existing dataset dirs.
- Defaults from the spec: trigger 0.25 m/s for 3 steps, min alignment 0.7, speed cap 1.0, pose weight 0.5, blend 0.3 s, cooldown 1.0 s, window 0.6 s, min travel 0.3 m, lead 0.1 s, exit speed 0.3 m/s.
- Do not launch training or long benchmarks.

---

### Task 1: G1 mirroring module

**Files:**
- Create: `src/safe_mimic/motions/mirror_g1.py`
- Test: `tests/test_mirror_g1.py`

**Interfaces:**
- Produces: `G1MirrorSpec.from_model(model: mujoco.MjModel) -> G1MirrorSpec` with fields `joint_names: tuple[str, ...]`, `body_names: tuple[str, ...]` (model bodies minus `world`), `joint_perm: np.ndarray[int]`, `joint_sign: np.ndarray[float]`, `body_perm: np.ndarray[int]`; `mirror_motion_arrays(arrays: Mapping[str, np.ndarray], spec: G1MirrorSpec) -> dict[str, np.ndarray]`; `mirror_name(name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
"""Left/right mirroring of G1 tracker clips."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from safe_mimic.motions.mirror_g1 import G1MirrorSpec, mirror_motion_arrays, mirror_name

CLIP = Path(
  "artifacts/bones-seed/datasets/g1_ballet_v1_trim1s/npz_50hz/231006/"
  "dance_pirouette_001__A464.npz"
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(G1_XML))


@pytest.fixture(scope="module")
def spec(model) -> G1MirrorSpec:
  return G1MirrorSpec.from_model(model)


def test_mirror_name_swaps_sides() -> None:
  assert mirror_name("left_hip_roll_joint") == "right_hip_roll_joint"
  assert mirror_name("right_wrist_yaw_link") == "left_wrist_yaw_link"
  assert mirror_name("waist_yaw_joint") == "waist_yaw_joint"


def test_spec_permutes_sides_and_flips_roll_yaw(spec) -> None:
  j = spec.joint_names
  assert spec.joint_perm[j.index("left_knee_joint")] == j.index("right_knee_joint")
  assert spec.joint_perm[j.index("waist_pitch_joint")] == j.index("waist_pitch_joint")
  assert spec.joint_sign[j.index("left_knee_joint")] == 1.0  # pitch keeps sign
  assert spec.joint_sign[j.index("left_hip_roll_joint")] == -1.0
  assert spec.joint_sign[j.index("waist_yaw_joint")] == -1.0
  b = spec.body_names
  assert b[0] == "pelvis" and "world" not in b
  assert spec.body_perm[b.index("left_elbow_link")] == b.index("right_elbow_link")
  assert spec.body_perm[b.index("torso_link")] == b.index("torso_link")


def test_mirror_is_an_involution(spec) -> None:
  with np.load(CLIP) as data:
    arrays = {k: data[k] for k in data.files}
  twice = mirror_motion_arrays(mirror_motion_arrays(arrays, spec), spec)
  for key, value in arrays.items():
    np.testing.assert_allclose(twice[key], value, atol=1e-6)


def _fk_body_positions(model, spec, root_pos, root_quat, joint_pos) -> np.ndarray:
  data = mujoco.MjData(model)
  data.qpos[:3] = root_pos
  data.qpos[3:7] = root_quat
  data.qpos[7:] = joint_pos
  mujoco.mj_kinematics(model, data)
  return data.xpos[1:].copy()  # bodies minus world, model order


def test_fk_reproduces_original_and_mirrored_clouds(model, spec) -> None:
  with np.load(CLIP) as data:
    arrays = {k: data[k] for k in data.files}
  mirrored = mirror_motion_arrays(arrays, spec)
  for clip in (arrays, mirrored):
    for frame in (0, 50, 150):
      fk = _fk_body_positions(
        model,
        spec,
        clip["body_pos_w"][frame, 0],
        clip["body_quat_w"][frame, 0],
        clip["joint_pos"][frame],
      )
      np.testing.assert_allclose(fk, clip["body_pos_w"][frame], atol=5e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `/tmp/claude-1000/wt/wpy -m pytest tests/test_mirror_g1.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: safe_mimic.motions.mirror_g1`.

- [ ] **Step 3: Implement**

```python
"""Left/right mirroring of G1 tracker NPZ clips (reflection across the x-z plane)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np

TIME_MAJOR_BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def mirror_name(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name


@dataclass(frozen=True)
class G1MirrorSpec:
  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]
  joint_perm: np.ndarray
  joint_sign: np.ndarray
  body_perm: np.ndarray

  @classmethod
  def from_model(cls, model: mujoco.MjModel) -> G1MirrorSpec:
    joints = [
      j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    joint_names = tuple(
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints
    )
    body_names = tuple(
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)
    )
    joint_perm = np.array(
      [joint_names.index(mirror_name(n)) for n in joint_names], dtype=np.int64
    )
    # A hinge about body y is a pitch: same sign after reflection. Roll (x) and
    # yaw (z) reverse.
    joint_sign = np.array(
      [1.0 if abs(model.jnt_axis[j][1]) > 0.5 else -1.0 for j in joints]
    )
    body_perm = np.array(
      [body_names.index(mirror_name(n)) for n in body_names], dtype=np.int64
    )
    return cls(joint_names, body_names, joint_perm, joint_sign, body_perm)


def mirror_motion_arrays(
  arrays: Mapping[str, np.ndarray], spec: G1MirrorSpec
) -> dict[str, np.ndarray]:
  out: dict[str, np.ndarray] = {}
  for key, value in arrays.items():
    value = np.asarray(value)
    if key in ("joint_pos", "joint_vel"):
      out[key] = (value[:, spec.joint_perm] * spec.joint_sign).astype(value.dtype)
    elif key in TIME_MAJOR_BODY_KEYS:
      swapped = value[:, spec.body_perm].copy()
      if key == "body_quat_w":
        swapped[..., 1] *= -1.0  # (w, x, y, z) -> (w, -x, y, -z)
        swapped[..., 3] *= -1.0
      elif key == "body_ang_vel_w":
        swapped[..., 0] *= -1.0
        swapped[..., 2] *= -1.0
      else:
        swapped[..., 1] *= -1.0
      out[key] = swapped
    else:
      out[key] = value.copy()
  return out


__all__ = ["G1MirrorSpec", "TIME_MAJOR_BODY_KEYS", "mirror_motion_arrays", "mirror_name"]
```

- [ ] **Step 4: Run tests, expect PASS** (if FK fails on the ORIGINAL clip, the NPZ joint/body order differs from model order: print `spec.joint_names` vs. the converter's order and fix the spec, not the test.)

- [ ] **Step 5: Commit** `feat(motions): G1 left/right mirroring with FK-verified spec`

---

### Task 2: Mirror the trimmed ballet library

**Files:**
- Create: `scripts/mirror_g1_motion_library.py`
- Data: `artifacts/bones-seed/datasets/g1_ballet_v1_trim1s_mirror/{npz_50hz/, ballet.yaml}`

**Interfaces:**
- Consumes: Task 1.
- Produces: manifest with 696 entries; mirrored files named `<stem>__mirror.npz` beside copied originals (same relative dirs); mirror entries carry `mirrored: true`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Copy a G1 clip manifest and add a left/right mirrored twin of every clip."""

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
  entries = []
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
    np.savez_compressed(args.output_root / mirrored_rel, **mirror_motion_arrays(arrays, spec))
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
```

- [ ] **Step 2: Run it**

```
/tmp/claude-1000/wt/wpy scripts/mirror_g1_motion_library.py \
  --manifest artifacts/bones-seed/datasets/g1_ballet_v1_trim1s/ballet.yaml \
  --output-root artifacts/bones-seed/datasets/g1_ballet_v1_trim1s_mirror/npz_50hz \
  --output-manifest artifacts/bones-seed/datasets/g1_ballet_v1_trim1s_mirror/ballet.yaml
```
Expected: `696 entries -> ...`. Verify `load_packed_npz_manifest(<new yaml>)` returns 696 sources.

- [ ] **Step 3: Commit** the script: `feat(scripts): mirror a G1 clip manifest`.

---

### Task 3: Travel index

**Files:**
- Create: `src/safe_mimic/motions/escape_move_index.py`, `scripts/build_escape_move_index.py`
- Test: `tests/test_escape_move_index.py`
- Data: `artifacts/bones-seed/datasets/g1_ballet_v1_trim1s_mirror/escape_moves.json`

**Interfaces:**
- Produces: `EscapeMoveIndexCfg(window_s=0.6, min_travel_m=0.3, lead_s=0.1, exit_speed_mps=0.3)`; `ClipTravel(entry_frame: int, exit_frame: int, direction_b: tuple[float, float], speed_mps: float, travel_m: float, candidate: bool)`; `clip_travel(root_xy: np.ndarray[T,2], anchor_quat_wxyz: np.ndarray[T,4], fps: float, cfg) -> ClipTravel`; `quat_yaw(q: np.ndarray[...,4]) -> np.ndarray`; `build_escape_move_index(manifest: Path, anchor_body_index: int, cfg) -> dict`; `load_escape_move_index(path) -> dict`. JSON shape: `{"fps", "anchor_body_index", "window_s", "clips": [{"file", "entry_frame", "exit_frame", "direction_b", "speed_mps", "travel_m", "candidate"}]}` in manifest order.

- [ ] **Step 1: Failing tests**

```python
import json

import numpy as np

from safe_mimic.motions.escape_move_index import (
  ClipTravel,
  EscapeMoveIndexCfg,
  clip_travel,
  load_escape_move_index,
  quat_yaw,
)

FPS = 50.0


def _yaw_quat(yaw: float) -> np.ndarray:
  return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def test_quat_yaw_roundtrip() -> None:
  assert np.isclose(quat_yaw(_yaw_quat(0.7)), 0.7)


def test_static_clip_is_not_a_candidate() -> None:
  n = 100
  travel = clip_travel(np.zeros((n, 2)), np.tile(_yaw_quat(0.0), (n, 1)), FPS, EscapeMoveIndexCfg())
  assert travel.candidate is False and travel.travel_m == 0.0


def test_travel_segment_is_indexed_in_the_anchor_frame() -> None:
  # 1 s still, 1 s travelling 1.0 m along world +y at heading +90 deg (=> body +x), 1 s still.
  fps = FPS
  still = int(fps)
  xy = np.zeros((3 * still, 2))
  xy[still : 2 * still, 1] = np.linspace(0.0, 1.0, still)
  xy[2 * still :, 1] = 1.0
  quat = np.tile(_yaw_quat(np.pi / 2), (3 * still, 1))
  t = clip_travel(xy, quat, fps, EscapeMoveIndexCfg())
  assert isinstance(t, ClipTravel) and t.candidate
  assert t.travel_m > 0.55 and abs(t.speed_mps - t.travel_m / 0.6) < 1e-6
  assert np.allclose(t.direction_b, (1.0, 0.0), atol=1e-6)
  # entry = fastest-window start minus 0.1 s lead; window starts inside the travel.
  assert still - 5 <= t.entry_frame <= still + 15
  # exit: first frame after the window where speed < 0.3 m/s -> the trailing stance.
  assert 2 * still - 2 <= t.exit_frame <= 2 * still + 3
  assert t.exit_frame <= 3 * still - 1


def test_exit_frame_stays_inside_the_clip() -> None:
  n = 40  # 0.8 s: window nearly the whole clip, travel to the end
  xy = np.stack([np.linspace(0.0, 1.0, n), np.zeros(n)], axis=1)
  t = clip_travel(xy, np.tile(_yaw_quat(0.0), (n, 1)), FPS, EscapeMoveIndexCfg())
  assert t.candidate and t.exit_frame == n - 1


def test_index_roundtrip(tmp_path) -> None:
  payload = {"fps": 50.0, "anchor_body_index": 15, "window_s": 0.6, "clips": []}
  path = tmp_path / "escape_moves.json"
  path.write_text(json.dumps(payload))
  assert load_escape_move_index(path) == payload
```

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Implement module**

```python
"""Index each clip's travel phase so a planner can pick an escape move."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class EscapeMoveIndexCfg:
  window_s: float = 0.6
  min_travel_m: float = 0.3
  lead_s: float = 0.1
  exit_speed_mps: float = 0.3


@dataclass(frozen=True)
class ClipTravel:
  entry_frame: int
  exit_frame: int
  direction_b: tuple[float, float]
  speed_mps: float
  travel_m: float
  candidate: bool


def quat_yaw(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clip_travel(
  root_xy: np.ndarray, anchor_quat_wxyz: np.ndarray, fps: float, cfg: EscapeMoveIndexCfg
) -> ClipTravel:
  n = int(root_xy.shape[0])
  window = max(1, int(round(cfg.window_s * fps)))
  if n <= window:
    return ClipTravel(0, max(0, n - 1), (1.0, 0.0), 0.0, 0.0, False)
  disp = root_xy[window:] - root_xy[:-window]
  travel = np.linalg.norm(disp, axis=1)
  i = int(np.argmax(travel))
  travel_m = float(travel[i])
  if travel_m < 1e-6:
    return ClipTravel(0, n - 1, (1.0, 0.0), 0.0, 0.0, False)
  yaw = float(quat_yaw(anchor_quat_wxyz[i]))
  c, s = np.cos(-yaw), np.sin(-yaw)
  d = disp[i] / travel_m
  direction_b = (float(c * d[0] - s * d[1]), float(s * d[0] + c * d[1]))
  entry = max(0, i - int(round(cfg.lead_s * fps)))
  speed = np.linalg.norm(np.diff(root_xy, axis=0), axis=1) * fps  # speed[k] over k->k+1
  exit_frame = n - 1
  for k in range(i + window, n - 1):
    if speed[k] < cfg.exit_speed_mps:
      exit_frame = k
      break
  exit_frame = min(exit_frame, n - 1)
  return ClipTravel(
    entry_frame=entry,
    exit_frame=int(exit_frame),
    direction_b=direction_b,
    speed_mps=travel_m / cfg.window_s,
    travel_m=travel_m,
    candidate=travel_m >= cfg.min_travel_m,
  )


def build_escape_move_index(
  manifest: Path, anchor_body_index: int, cfg: EscapeMoveIndexCfg
) -> dict:
  data = yaml.safe_load(Path(manifest).read_text())
  root = Path(data["root_path"])
  fps = float(data["fps"])
  clips = []
  for entry in data["motions"]:
    with np.load(root / entry["file"]) as arrays:
      travel = clip_travel(
        arrays["body_pos_w"][:, 0, :2], arrays["body_quat_w"][:, anchor_body_index], fps, cfg
      )
    clips.append({"file": entry["file"], **asdict(travel)})
  return {
    "fps": fps,
    "anchor_body_index": anchor_body_index,
    "window_s": cfg.window_s,
    "clips": clips,
  }


def load_escape_move_index(path: str | Path) -> dict:
  return json.loads(Path(path).read_text())


__all__ = [
  "ClipTravel",
  "EscapeMoveIndexCfg",
  "build_escape_move_index",
  "clip_travel",
  "load_escape_move_index",
  "quat_yaw",
]
```

Script `scripts/build_escape_move_index.py`: argparse `--manifest`, `--output`, `--anchor-body torso_link`, cfg flags; resolve `anchor_body_index` with `G1MirrorSpec.from_model(...).body_names.index(anchor_body)`; dump JSON with `indent=1`; print candidate count and per-direction sector counts (front/left/back/right).

- [ ] **Step 4: Tests pass; run the script on the mirrored manifest.** Expected: ~528 candidates (264 x 2) and near-symmetric left/right counts.

- [ ] **Step 5: Commit** `feat(motions): travel-phase index for escape moves`.

---

### Task 4: Planner logic module

**Files:**
- Create: `src/safe_mimic/tasks/escape_moves.py`
- Test: `tests/test_escape_moves.py`

**Interfaces:**
- Produces:
  - `EscapeMoveCfg(index_file: str, trigger_speed_mps=0.25, trigger_steps=3, min_alignment=0.7, speed_cap_mps=1.0, pose_distance_weight=0.5, blend_s=0.3, cooldown_s=1.0, trigger_source="teacher")`, `__post_init__` validates source in `("teacher", "actor")` and positive thresholds.
  - `EscapeMoveTable(entry_frames: Tensor[K] long global, exit_frames: Tensor[K] long global, clip_starts: Tensor[K], clip_ends: Tensor[K], direction_b: Tensor[K,2], speed_mps: Tensor[K], entry_joint_pos: Tensor[K,J])` with `@classmethod from_index(index: dict, source_paths: Sequence[Path], clip_start_idx: Tensor, clip_num_frames: Tensor, joint_pos: Tensor, device) -> EscapeMoveTable` (maps index rows to library clips by `Path(file).name` matched against `source_paths[i].name`; skips non-candidates; raises if a row's file is not in the library).
  - `body_frame_planar(vec_w: Tensor[N,2], anchor_quat_w: Tensor[N,4]) -> Tensor[N,2]` (rotate by −yaw).
  - `update_trigger_count(count: Tensor[N] long, speed: Tensor[N], threshold: float) -> Tensor[N]`.
  - `select_escape_moves(table, escape_dir_b: Tensor[N,2], joint_pos: Tensor[N,J], cfg) -> Tensor[N] long` (index into the table, −1 when none qualifies or the direction norm is 0).
  - `blend_alpha(steps_left: Tensor[N] long, total_steps: int) -> Tensor[N] float` = `1 - steps_left/total` clamped to [0,1].
  - `nlerp(q0: Tensor[...,4], q1: Tensor[...,4], alpha: Tensor[...,1]) -> Tensor` with sign alignment and renormalisation.

- [ ] **Step 1: Failing tests** covering: cfg defaults & validation; `body_frame_planar` (heading +90° maps world +y to body +x); trigger count increments and resets; `select_escape_moves` picks the aligned candidate, applies the 0.7 gate (returns −1 for a perpendicular direction), caps speed, prefers a closer pose when alignment ties; `blend_alpha` schedule 0→1; `nlerp` endpoints and antipodal sign fix; `from_index` maps by file name, skips non-candidates, gathers `entry_joint_pos` from the flat library at global frames.

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Implement** (torch only; no mjlab imports so the tests stay light):

```python
@dataclass
class EscapeMoveCfg:
  index_file: str
  trigger_speed_mps: float = 0.25
  trigger_steps: int = 3
  min_alignment: float = 0.7
  speed_cap_mps: float = 1.0
  pose_distance_weight: float = 0.5
  blend_s: float = 0.3
  cooldown_s: float = 1.0
  trigger_source: str = "teacher"

  def __post_init__(self) -> None:
    if self.trigger_source not in ("teacher", "actor"):
      raise ValueError("trigger_source must be 'teacher' or 'actor'")
    if self.trigger_speed_mps <= 0 or self.trigger_steps < 1 or self.blend_s < 0:
      raise ValueError("escape move thresholds must be positive")
```

`select_escape_moves`: `norm = ||d||`; `cos = (direction_b @ d^T)` (K,N)ᵀ; `score = cos * clamp(speed, max=cap) - w * sqrt(mean((entry_joint_pos[None] - joint_pos[:, None])**2, -1))`; mask `cos >= min_alignment` and `norm > 1e-6`; `score[~mask] = -inf`; `best = argmax`; return `where(any(mask, 1), best, -1)`.

- [ ] **Step 4: Tests pass. Step 5: Commit** `feat(tasks): escape move planner logic`.

---

### Task 5: Command integration (state machine, blend, hint, metrics)

**Files:**
- Modify: `src/safe_mimic/tasks/kinematic_replay_command.py` (cfg class ~L1866+, `__init__` of the filtered command ~L440-570 metrics block, `_update_command` ~L1536 after `_advance_time_steps`, planar filter section ~L1618 to store `_last_intervention_w`, `_resample_command` ~L1159, raw getters L571-635)
- Test: `tests/test_escape_moves_command.py`

**Interfaces:**
- Consumes: Task 4.
- Produces: `PlanarFilteredReplayMotionCommandCfg.escape_moves: EscapeMoveCfg | None = None`; `PlanarFilteredReplayMotionCommand.set_actor_escape_hint(hint_b: Tensor[N,2])`; `escape_state` attributes (`_escape_mode`, `_escape_trigger_count`, `_escape_cooldown_steps`, `_escape_saved_start/_end/_frame`, `_escape_exit_frame`, `_escape_blend_frame`, `_escape_blend_steps_left`, `_escape_count`); metrics `escape_move_active`, `escape_move_count`, `escape_intervention_speed_mps`; method `_update_escape_moves(replay_env_ids)`.

- [ ] **Step 1: Failing tests** on an `object.__new__` stub (copy the pattern of `tests/test_closed_loop_root_target.py::_update_stub` but drive `_update_escape_moves` and the raw getters directly):
  - cfg default `None` (dataclass field default).
  - A fake `motion` with `joint_pos`/`body_*` flat tensors for 3 clips (frames 0-9, 10-19, 20-29), table with clip 2 as the only candidate (entry 22, exit 27, direction (1,0), speed 1.0). Env 0 in clip 0 at frame 4.
  - Teacher trigger vector (0.5, 0) for 3 steps → mode ESCAPING, `time_steps == 22`, saved (0,10,frame at trigger), `blend_frame == saved frame`, `blend_steps_left == blend_steps`, count metric 1. Two steps only → no switch.
  - Stepping `time_steps` to 27 then `_update_escape_moves` → NOMINAL, `time_steps == saved frame`, `blend_frame == 26`, cooldown set; a fresh trigger during cooldown does not switch.
  - Blended raw getters: with `blend_frame` set and `blend_steps_left = total/2`, `_raw_joint_pos()` equals the midpoint of the two frames; `_raw_body_pos_w()` root xy equals the robot root xy (both frames glued) and z is the midpoint.
  - `trigger_source="actor"`: the teacher vector is ignored, `set_actor_escape_hint` drives the switch.
  - `_reset_escape_state(env_ids)` clears mode/blend/count.

- [ ] **Step 2: Run, expect AttributeError.**

- [ ] **Step 3: Implement** (hunks only, follow the existing 2-space style):
  1. Cfg: `escape_moves: EscapeMoveCfg | None = None` next to `disable_filters`.
  2. In the filtered command `__init__` after metrics: `self._last_intervention_w = zeros(N,2)`, `self._actor_escape_hint_b = zeros(N,2)`, and if `cfg.escape_moves`: `self._escape_table = EscapeMoveTable.from_index(load_escape_move_index(cfg.escape_moves.index_file), [s.path for s in self.motion._library.sources], self.motion.clip_start_idx, self.motion.clip_num_frames, self.motion.joint_pos, self.device)` (require `_has_clip_library()`, else `ValueError`), `_escape_blend_steps = max(1, round(blend_s / step_dt))`, `_escape_cooldown_total = round(cooldown_s / step_dt)`, state tensors (long/bool, `_escape_blend_frame` filled with −1), metrics zeros.
  3. `_update_escape_moves(replay_env_ids)` implementing the spec's state machine with the pure functions; called in `_update_command` right after `wrapped = self._advance_time_steps(...)` when `self.cfg.escape_moves is not None`.
  4. After `result = filter_planar_velocity(...)`: `self._last_intervention_w[replay_env_ids] = result.intervention_w[replay_env_ids]` (also in the passthrough branch set zeros).
  5. Blend in the getters: add `_blend_mask()` (envs with `_escape_blend_frame >= 0`), `_aligned_cloud_for_frames(frames) -> (pos, quat, lin, ang)` that mirrors `_raw_body_*` but for arbitrary frames (yaw delta from `motion.body_quat_w[frames, anchor]` vs `robot_anchor_quat_w`, root xy = `robot_body_pos_w[:, 0, :2]`), and in `_raw_body_pos_w/_raw_body_quat_w/_raw_body_lin_vel_w/_raw_body_ang_vel_w/_raw_joint_pos/_raw_joint_vel` lerp toward the frozen frame where the mask is set (`nlerp` for quats). When `cfg.escape_moves is None` the getters take the old path unchanged.
  6. `_resample_command`: call `_reset_escape_state(env_ids)` first (when enabled).
  7. `set_actor_escape_hint(hint_b)`: copy into `_actor_escape_hint_b`.

- [ ] **Step 4: Tests pass; full suite passes (existing stub tests must not need changes since the feature is off).**

- [ ] **Step 5: Commit** `feat(tasks): escape-move planner in the filtered replay command`.

---

### Task 6: Builder kwarg, constants, task registration

**Files:**
- Modify: `src/safe_mimic/tasks/env_cfg.py` (constants after `UNTRIMMED_G1_BALLET_MANIFEST`; builder signature + body), `src/safe_mimic/tasks/__init__.py` (task id, rl cfg, registration, `__all__`)
- Test: `tests/test_escape_moves_task.py`

**Interfaces:**
- Produces: `DEFAULT_G1_BALLET_MIRROR_MANIFEST`, `DEFAULT_G1_BALLET_ESCAPE_INDEX`; builder kwarg `escape_moves: bool = False` (`True` requires `motion_manifest`; sets `motion.escape_moves = EscapeMoveCfg(index_file=str(Path(motion_manifest).with_name("escape_moves.json")))`); `LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID = "...-CoAdjust-Unified-Joint-Leash-Ballet-Moves"`; rl cfg `lidar_auxiliary_coadjust_unified_joint_leash_ballet_moves_rl_cfg` (experiment name `..._coadjust_unified_joint_leash_ballet_moves`, same actor/algorithm tweaks as the other Leash-Ballet cfgs).

- [ ] **Step 1: Failing tests**: Leash-Ballet cfg `commands["motion"].escape_moves is None` and its `asdict`-free field comparison against a fresh build is unchanged (build twice, compare `motion.motion_file`, `disable_filters`, `escape_moves`); Moves task train+play cfgs have `escape_moves.index_file` ending in `escape_moves.json`, `motion_file == str(DEFAULT_G1_BALLET_MIRROR_MANIFEST)`, `trigger_source == "teacher"`; `load_rl_cfg` experiment name suffix; `escape_moves=True` without manifest raises.

- [ ] **Step 2-4:** implement, tests pass. **Step 5: Commit** `feat(tasks): Leash-Ballet-Moves task`.

---

### Task 7: Script routing, actor trigger in play, envelope metrics

**Files:**
- Modify: `scripts/benchmark_policy_reaction_envelope.py`, `scripts/diagnose_joint_compliance.py`, `scripts/analyze_policy_failures.py`, `scripts/play_adjuster_ghost.py`, `scripts/diagnose_root_tracking.py`, `scripts/record_viser_comparison.py`, `scripts/benchmark_policy_avoidance.py`, `src/safe_mimic/evaluation.py`
- Test: `tests/test_fair_regime_summary.py` (append), `tests/test_escape_moves_task.py` (append an argparse-choices check by importing the scripts' parsers is NOT feasible; instead grep-free: assert the id string appears in each script via `Path(...).read_text()`).

**Interfaces:**
- Consumes: Task 6 id.
- Produces: every script accepts the Moves id (choices + builder `elif` with `motion_manifest=str(DEFAULT_G1_BALLET_MIRROR_MANIFEST), escape_moves=True`); avoidance variant `coadjust-unified-joint-leash-ballet-moves`; play flag `--escape-trigger {teacher,actor}` (actor: set `cfg.commands["motion"].escape_moves.trigger_source = "actor"` and wrap `env.step` to call `command.set_actor_escape_hint(policy.predict_avoidance(obs)[:, :2])` before stepping, obs taken from `env.get_observations()`); envelope per-episode tensors `escape_moves` (from `command.metrics["escape_move_count"]` at episode end) and `escape_resolved` (count > 0 and max `escape_intervention_speed_mps` during the episode < 0.2); `fair_regime_summary` gains `escape_resolved_rate` (NaN when no case has the key).

- [ ] **Steps:** failing test for `fair_regime_summary` with/without the key; implement; route scripts by cloning the Blind-NoHumans handling (the id appears in imports/choices/elif blocks; `benchmark_policy_avoidance.py` string lists, lambda, and id dict); ruff; commit `feat: route Leash-Ballet-Moves through eval/play scripts; escape metrics`.

---

### Task 8: Smoke, docs, wrap-up

- [ ] Smoke script (scratchpad): load Moves play cfg, 16 envs, cuda:0, `--dense` crowd defaults, zero actions for 300 steps; log per step `escape_move_active.sum()`, `escape_move_count.max()`, max abs joint-target delta between consecutive `_raw_joint_pos()` calls (expect < 0.3 rad, except none), and confirm at least one env resumes (mode returns to 0 after a switch). Report numbers.
- [ ] `ruff check src tests scripts`; full test suite.
- [ ] Ledger entry in `.superpowers/sdd/2026-09-03-phase3-unified-filtered-reference/progress.md` and handoff item 23 (branch, task id, train command, smoke numbers).
- [ ] Commit `docs: escape moves smoke + handoff`.

## Self-review

- Spec coverage: mirroring (T1-2), index (T3), planner cfg/state/selection/blend/metrics (T4-5), actor source (T5, T7), task/cfg (T6), routing + envelope metric (T7), tests (each task), rollout smoke (T8). Training launch is the parent's.
- Naming: `EscapeMoveCfg`, `EscapeMoveTable`, `select_escape_moves`, `blend_alpha`, `nlerp`, `body_frame_planar`, `update_trigger_count`, `set_actor_escape_hint`, `_update_escape_moves`, `_reset_escape_state`, metrics `escape_move_active/escape_move_count/escape_intervention_speed_mps`, constants `DEFAULT_G1_BALLET_MIRROR_MANIFEST/DEFAULT_G1_BALLET_ESCAPE_INDEX`, task id `..._LEASH_BALLET_MOVES_TASK_ID` used consistently above.
