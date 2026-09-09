# Phase 3: Unified filtered reference with nominal rewards — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the co-adjust LiDAR policy with the stock mjlab tracking rewards
evaluated against ONE complete privileged-filtered reference (root pose and
velocity, every body pose and velocity, joints), and nothing else.

**Architecture:** The pipeline is: LiDAR of randomized humans -> live planar +
link CBF filters on the reference (privileged) -> FK of every joint correction
and closed-loop integration of the filtered root -> the nominal tracking task's
reward set on that filtered reference. Everything else is the nominal task.
Two default-off flags on `PlanarFilteredReplayMotionCommandCfg` implement the
reference side; one env-cfg builder strips the bespoke reward stack; one task
registration ties it together. Every existing task stays bit-identical.

**Tech Stack:** MjLab ManagerBasedRlEnv, torch, mujoco (tests), rsl_rl
`AvoidanceAuxiliaryPPO` (unchanged).

**Spec:** user direction 2026-09-03 (this conversation): "pass the filtered
position / velocity of the links and the root into the original rewards ...
lidar of randomized human configurations -> live motion filter FK from
privileged info -> the nominal reward from the nominal tracking task on the
filtered motion while everything else should be the same as the nominal task."
Evidence motivating it: FKC2@10k envelope 69.6% (FKC 53.4%), pelvis
escape_ratio 0.01 (FKC 0.47), arm compliance flat (0.61 vs 0.57): the bespoke
reward stack (planar velocity/progress/freeze, filtered-joint, urgent-escape,
active-correction) is not the lever.

## Global constraints

- No git state changes (no commits, no resets, no stashes). Dirty tree is the
  source of truth.
- `uv run ruff check` clean; NEVER whole-file `ruff format` on tracked files
  (hunks only). 2-space indent, mirror file conventions.
- Full `uv run pytest -q` green (baseline 239).
- GPU is occupied by the FKC2 run until Task 4 stops it: CPU tests only; at most
  one tiny env-introspection smoke (<= 8 envs); NEVER `uv run train` before
  Task 4.
- Backward compatibility: both new cfg flags default False; every existing
  task, checkpoint, and benchmark behaves identically (pin with tests).
- The mjlab package is vendored in `.venv` and read-only: all changes live in
  `src/safe_mimic`, `scripts`, `tests`.

## File structure

- Modify `src/safe_mimic/tasks/kinematic_replay_command.py` — cfg flags
  `propagate_joint_corrections_to_body_targets`, `closed_loop_root_target`;
  generalized chain-FK propagation (whole body, anchor included); closed-loop
  root integration via a pure helper `advance_filtered_root_xy`.
- Modify `src/safe_mimic/tasks/env_cfg.py` — new builder
  `unitree_g1_lidar_unified_reference_tracking_env_cfg(play=False)`.
- Modify `src/safe_mimic/tasks/__init__.py` — `LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID`.
- Modify `scripts/benchmark_policy_reaction_envelope.py`,
  `scripts/diagnose_joint_compliance.py`, `scripts/benchmark_policy_avoidance.py`,
  `scripts/analyze_policy_failures.py` — task selection for the new task.
- Create `tests/test_whole_body_target_propagation.py`,
  `tests/test_closed_loop_root_target.py`,
  `tests/test_coadjust_unified_task.py`.

---

## Task 1: Whole-body FK propagation of joint corrections (anchor included)

**Files:**
- Modify: `src/safe_mimic/tasks/kinematic_replay_command.py`
  (`_init_arm_target_propagation` ~L511-665, `_update_arm_body_target_offsets`
  ~L667, `_compute_arm_body_target_offsets` ~L699, properties `body_quat_w`,
  `body_pos_w`, `body_lin_vel_w`, `anchor_pos_w`, `anchor_lin_vel_w`,
  `anchor_quat_w` ~L821-903, `__init__` ~L378, cfg dataclass ~L1490)
- Test: `tests/test_whole_body_target_propagation.py`

**Interfaces:**
- Consumes: existing `hinge_chain_body_positions(...)`, existing arm-mode
  caches `_arm_body_target_offset_w`, `_arm_body_target_quat_delta_w`,
  `_arm_body_target_lin_vel_w`, `_arm_prop_cloud_ids`, `_arm_prop_velocity_hold`.
- Produces: cfg field `propagate_joint_corrections_to_body_targets: bool = False`
  on `PlanarFilteredReplayMotionCommandCfg`; instance flag
  `self._propagate_targets: bool` (True in either arm or whole-body mode);
  `self._anchor_target_corrected: bool` (True only in whole-body mode);
  `self._init_target_propagation(joint_local_ids: list[int], *, allow_anchor_descendant: bool) -> None`.

Design rules:
1. Whole-body mode = the existing arm machinery with (a) ALL 29 hinge joints
   as the chain set and (b) the pelvis (cloud index 0, the free-joint body)
   as the chain root. The existing chain-root discovery ("parent not in chain
   -> must be a tracked cloud body") already resolves hip_pitch and waist_yaw
   parents to cloud index 0. Every tracked non-root body in the G1 cloud
   (hip_roll, knee, ankle_roll, torso, shoulder_roll, elbow, wrist_yaw) owns
   exactly one hinge, so the existing "tracked bodies below the chain must own
   a hinge" check passes unchanged. Keep both errors.
2. The anchor (`torso_link`) IS a chain descendant in whole-body mode (waist
   yaw/roll/pitch). Replace the unconditional "anchor cannot be a descendant"
   error with: raise only when `allow_anchor_descendant` is False; otherwise
   set `self._anchor_target_corrected = True`. The "root body cannot be a
   descendant" error stays unconditional.
3. Both flags True -> `ValueError("choose arm-only or whole-body propagation, not both")`
   at construction.
4. Properties: `body_pos_w`/`body_quat_w`/`body_lin_vel_w` switch on
   `self._propagate_targets` instead of `self._propagate_arm_targets`
   (`getattr(self, "_propagate_targets", False)` — the `object.__new__` tests
   construct instances without `__init__`). `anchor_pos_w` adds
   `self._arm_body_target_offset_w[:, self.motion_anchor_body_index]` and
   `anchor_lin_vel_w` adds `self._arm_body_target_lin_vel_w[:, self.motion_anchor_body_index]`
   ONLY when `getattr(self, "_anchor_target_corrected", False)`;
   `anchor_quat_w` returns `quat_mul(self._arm_body_target_quat_delta_w[:, anchor], raw)`
   under the same gate. `anchor_ang_vel_w`/`body_ang_vel_w` stay raw.
5. Keep the `_arm_*` attribute names (validated, test-pinned path); add a
   one-line comment at the cache allocation: "historical name: in whole-body
   mode these caches cover every tracked body incl. the anchor". Renaming is
   out of scope.
6. `_update_command` calls `_update_arm_body_target_offsets` when
   `self._propagate_targets`; `_resample_command` resets the caches under the
   same flag (today both gate on `_propagate_arm_targets`).
7. `mj_model.qpos0 == 0` assertion stays for every chain hinge (G1 satisfies it).

- [ ] **Step 1: Write the failing tests** (`tests/test_whole_body_target_propagation.py`)

```python
"""Whole-body reference-frame FK propagation of joint corrections."""

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
  PlanarFilteredReplayMotionCommandCfg,
  hinge_chain_body_positions,
)

# Same three-hinge chain as tests/test_arm_target_propagation.py but with an
# extra branch off the base so one root serves two chains (legs + waist).
_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0.1 0.2 0.3" euler="0.3 -0.2 0.5">
      <body name="waist" pos="0.0 0.0 0.2" euler="0.05 0.0 0.0">
        <joint name="jw" type="hinge" axis="0 0 1" pos="0 0 -0.02"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
        <body name="torso" pos="0.0 0.0 0.15">
          <joint name="jt" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.02" mass="0.1"/>
        </body>
      </body>
      <body name="thigh" pos="0.0 -0.1 -0.05" euler="0.1 0.2 -0.3">
        <joint name="jh" type="hinge" axis="0 1 0" pos="0.02 0.01 -0.03"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
        <body name="shin" pos="0.0 0.0 -0.3" euler="-0.2 0.1 0.15">
          <joint name="jk" type="hinge" axis="0 1 0" pos="0.0 0.015 0.0"/>
          <geom type="sphere" size="0.02" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _assert_quat_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
  dot = (actual * expected).sum(dim=-1).abs()
  torch.testing.assert_close(dot, torch.ones_like(dot), atol=1e-5, rtol=0)


def test_branched_chain_from_one_root_matches_mujoco_kinematics() -> None:
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)
  chain_names = ("waist", "thigh", "torso", "shin")  # topological order
  body_ids = [model.body(name).id for name in chain_names]
  joint_ids = [model.joint(name).id for name in ("jw", "jh", "jt", "jk")]
  base_id = model.body("base").id
  for qpos in (np.array([0.4, -0.3, 0.7, 0.2]), np.array([-0.9, 0.5, -0.2, 1.1])):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    positions, quaternions = hinge_chain_body_positions(
      torch.tensor(data.xpos[base_id], dtype=torch.float32).reshape(1, 1, 3),
      torch.tensor(data.xquat[base_id], dtype=torch.float32).reshape(1, 1, 4),
      parent_chain_index=(-1, -1, 0, 1),
      parent_root_slot=(0, 0, 0, 0),
      body_pos_l=torch.tensor(model.body_pos[body_ids], dtype=torch.float32),
      body_quat_l=torch.tensor(model.body_quat[body_ids], dtype=torch.float32),
      joint_pos_l=torch.tensor(model.jnt_pos[joint_ids], dtype=torch.float32),
      joint_axis_l=torch.tensor(model.jnt_axis[joint_ids], dtype=torch.float32),
      joint_angles=torch.tensor(qpos, dtype=torch.float32).reshape(1, 4),
    )
    torch.testing.assert_close(
      positions[0], torch.tensor(data.xpos[body_ids], dtype=torch.float32), atol=1e-5, rtol=0
    )
    _assert_quat_close(quaternions[0], torch.tensor(data.xquat[body_ids], dtype=torch.float32))


def test_both_propagation_flags_rejected() -> None:
  # Only the two flags matter for the guard; exercise the guard helper directly.
  from safe_mimic.tasks.kinematic_replay_command import _validate_propagation_flags

  with pytest.raises(ValueError, match="not both"):
    _validate_propagation_flags(arm=True, whole_body=True)
  assert _validate_propagation_flags(arm=False, whole_body=True) == (True, True)
  assert _validate_propagation_flags(arm=True, whole_body=False) == (True, False)
  assert _validate_propagation_flags(arm=False, whole_body=False) == (False, False)


def _bare_command(num_envs: int, num_bodies: int, anchor: int) -> PlanarFilteredReplayMotionCommand:
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command.motion_anchor_body_index = anchor
  command._arm_body_target_offset_w = torch.zeros(num_envs, num_bodies, 3)
  command._arm_body_target_quat_delta_w = torch.zeros(num_envs, num_bodies, 4)
  command._arm_body_target_quat_delta_w[..., 0] = 1.0
  command._arm_body_target_lin_vel_w = torch.zeros(num_envs, num_bodies, 3)
  return command


def test_anchor_targets_follow_correction_only_in_whole_body_mode(monkeypatch) -> None:
  raw_pos = torch.randn(2, 4, 3)
  raw_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4] * 2)
  raw_vel = torch.randn(2, 4, 3)
  command = _bare_command(2, 4, anchor=2)
  command._filter_initialized = torch.tensor([False, False])
  monkeypatch.setattr(command, "_raw_body_pos_w", lambda: raw_pos)
  monkeypatch.setattr(command, "_raw_body_quat_w", lambda: raw_quat)
  monkeypatch.setattr(command, "_raw_body_lin_vel_w", lambda: raw_vel)
  command._arm_body_target_offset_w[:, 2] = torch.tensor([0.1, -0.2, 0.3])
  command._arm_body_target_lin_vel_w[:, 2] = torch.tensor([1.0, 2.0, 3.0])
  yaw90 = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])
  command._arm_body_target_quat_delta_w[:, 2] = yaw90

  command._propagate_targets = True
  command._anchor_target_corrected = False  # arm-only mode
  torch.testing.assert_close(command.anchor_pos_w, raw_pos[:, 2])
  torch.testing.assert_close(command.anchor_quat_w, raw_quat[:, 2])
  torch.testing.assert_close(command.anchor_lin_vel_w, raw_vel[:, 2])

  command._anchor_target_corrected = True  # whole-body mode
  torch.testing.assert_close(command.anchor_pos_w, raw_pos[:, 2] + torch.tensor([0.1, -0.2, 0.3]))
  torch.testing.assert_close(command.anchor_lin_vel_w, raw_vel[:, 2] + torch.tensor([1.0, 2.0, 3.0]))
  torch.testing.assert_close(command.anchor_quat_w, quat_mul(yaw90.expand(2, 4), raw_quat[:, 2]))
```

The `_filter_initialized = False` rows keep the planar offset at zero so the
test isolates the FK correction (mirror how `test_arm_target_propagation.py`
sets `_filter_initialized` / `_filtered_root_xy_w`; add
`command._filtered_root_xy_w = raw_pos[:, 0, :2]` if the property needs it).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_whole_body_target_propagation.py -q`
Expected: FAIL (`_validate_propagation_flags` missing; anchor properties ignore
the caches).

- [ ] **Step 3: Implement**

In `kinematic_replay_command.py`:

```python
def _validate_propagation_flags(*, arm: bool, whole_body: bool) -> tuple[bool, bool]:
  """Return (propagate_targets, anchor_target_corrected) for the cfg flags."""
  if arm and whole_body:
    raise ValueError(
      "choose arm-only or whole-body propagation, not both "
      "(propagate_arm_corrections_to_body_targets vs "
      "propagate_joint_corrections_to_body_targets)"
    )
  return (arm or whole_body, whole_body)
```

`__init__` (replacing the two `_propagate_arm_targets` lines):

```python
    self._propagate_targets, self._anchor_target_corrected = _validate_propagation_flags(
      arm=bool(cfg.propagate_arm_corrections_to_body_targets),
      whole_body=bool(cfg.propagate_joint_corrections_to_body_targets),
    )
    if self._propagate_targets:
      if self._anchor_target_corrected:
        joint_local_ids = list(range(self.robot.num_joints))
      else:
        tokens = self.cfg.link_filter.arm_joint_name_tokens
        joint_local_ids = [
          joint_id
          for joint_id, joint_name in enumerate(self.robot.joint_names)
          if any(token in joint_name for token in tokens)
        ]
      self._init_target_propagation(
        joint_local_ids, allow_anchor_descendant=self._anchor_target_corrected
      )
```

Rename `_init_arm_target_propagation(self)` to
`_init_target_propagation(self, joint_local_ids: list[int], *, allow_anchor_descendant: bool)`;
drop its internal token scan (the caller supplies the ids); change the
`"arm target propagation requires arm joints"` error to
`"target propagation requires at least one hinge joint"`; replace
`if self.motion_anchor_body_index in descendant_cloud_ids: raise ...` with
`if self.motion_anchor_body_index in descendant_cloud_ids and not allow_anchor_descendant: raise ...`.
Keep `_update_arm_body_target_offsets` / `_compute_arm_body_target_offsets`
bodies unchanged. Replace every `self._propagate_arm_targets` /
`getattr(self, "_propagate_arm_targets", False)` read with `_propagate_targets`
(grep: `__init__`, `_resample_command`, `_update_command`, three properties).
Anchor properties:

```python
  @property
  def anchor_pos_w(self) -> torch.Tensor:
    raw = self._raw_body_pos_w()[:, self.motion_anchor_body_index]
    filtered = raw.clone()
    if hasattr(self, "_filter_initialized"):
      raw_root_xy = self._raw_body_pos_w()[:, 0, :2]
      offset_xy = self._filtered_root_xy_w - raw_root_xy
      offset_xy = torch.where(self._filter_initialized[:, None], offset_xy, 0.0)
      filtered[:, :2] += offset_xy
    if getattr(self, "_anchor_target_corrected", False):
      filtered += self._arm_body_target_offset_w[:, self.motion_anchor_body_index]
    return filtered
```

(same shape for `anchor_lin_vel_w` with `_arm_body_target_lin_vel_w` and the
velocity delta; `anchor_quat_w` returns
`quat_mul(self._arm_body_target_quat_delta_w[:, idx], raw)` under the gate.)
Default-off outputs must be bit-identical to today: keep the original
early-return structure where the flag is off (write it so the off path
executes exactly today's operations).

Cfg dataclass: add after `propagate_arm_corrections_to_body_targets`:

```python
  # Propagate EVERY joint correction (legs, waist, arms) into the body
  # targets via FK rooted at the pelvis, anchor (torso) included, so the
  # nominal tracking rewards and the ee_body_pos termination evaluate the
  # fully filtered reference. Mutually exclusive with the arm-only flag.
  propagate_joint_corrections_to_body_targets: bool = False
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_whole_body_target_propagation.py tests/test_arm_target_propagation.py tests/test_coadjust_fkc_lidar_auxiliary_task.py -q`
Expected: PASS (arm-mode tests unchanged — if any arm test referenced
`_propagate_arm_targets` directly, update only that attribute name to
`_propagate_targets` and note it in the report).

- [ ] **Step 5: Real-model smoke (CPU, no env)**: in a throwaway script (scratchpad,
not committed), load the G1 `mj_model` via
`mujoco.MjModel.from_xml_path(".venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml")`
(if the XML needs mjlab's asset compile, use the pattern in
`tests/test_soma_capsule_asset.py`), set random `qpos` for the 29 hinges,
run `mj_kinematics`, and compare `hinge_chain_body_positions` for the full
29-body chain rooted at `pelvis` (build `parent_chain_index` /
`parent_root_slot` from `body_parentid` like `_init_target_propagation` does)
against `xpos`/`xquat` for the 13 tracked bodies incl. `torso_link`. Report the
max abs error (expect ~1e-6). If the asset cannot be loaded standalone,
report that and rely on the 8-env strict-load smoke in Task 3.

## Task 2: Closed-loop filtered root target

**Files:**
- Modify: `src/safe_mimic/tasks/kinematic_replay_command.py`
  (`_update_command` ~L1277-1360: the pre-filter glue and post-filter
  integration; cfg dataclass)
- Test: `tests/test_closed_loop_root_target.py`

**Interfaces:**
- Produces: cfg field `closed_loop_root_target: bool = False`; pure helper
  ```python
  def advance_filtered_root_xy(
    *,
    pre_filter_xy_w: torch.Tensor,        # (N, 2) position the filter was evaluated at
    raw_root_xy_w: torch.Tensor,          # (N, 2) live-aligned raw root this step
    raw_root_velocity_xy_w: torch.Tensor, # (N, 2)
    filtered_velocity_xy_w: torch.Tensor, # (N, 2) planar filter output
    translation_residual_xy_w: torch.Tensor,  # (N, 2) previous residual
    step_dt: float,
    closed_loop: bool,
  ) -> tuple[torch.Tensor, torch.Tensor]:  # (new filtered xy, new residual)
  ```

Semantics:
- Open loop (today, bit-identical): `residual += dt * (v_filt - v_raw)`;
  `filtered = raw + residual`.
- Closed loop: `filtered = pre_filter_xy + dt * v_filt`; `residual = filtered - raw`.
  The pre-filter position in closed-loop mode is the PERSISTENT
  `_filtered_root_xy_w` (world frame, not re-glued to the robot); the open-loop
  pre-filter position stays `raw + residual` as today. Reset, wrap and first
  initialization set `filtered = raw`, `residual = 0` in both modes (unchanged
  code). The recovery velocity `recovery_gain * (raw - filtered)` (max
  0.4 m/s) already pulls the target back toward the live-aligned raw root, so
  a robot that fails to follow accumulates a bounded root error — the nominal
  task's semantics — and a robot that does follow sees the target stop
  running ahead of it.

- [ ] **Step 1: Write the failing tests** (`tests/test_closed_loop_root_target.py`)

```python
"""Closed-loop integration of the privileged filtered root target."""

import torch

from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommandCfg,
  advance_filtered_root_xy,
)


def test_cfg_defaults_off() -> None:
  fields = {f.name: f.default for f in PlanarFilteredReplayMotionCommandCfg.__dataclass_fields__.values()}
  assert fields["closed_loop_root_target"] is False


def test_open_loop_matches_legacy_bookkeeping() -> None:
  raw = torch.tensor([[1.0, 2.0]])
  residual = torch.tensor([[0.10, -0.05]])
  v_raw = torch.tensor([[0.5, 0.0]])
  v_filt = torch.tensor([[0.5, 1.0]])
  filtered, new_residual = advance_filtered_root_xy(
    pre_filter_xy_w=raw + residual,
    raw_root_xy_w=raw,
    raw_root_velocity_xy_w=v_raw,
    filtered_velocity_xy_w=v_filt,
    translation_residual_xy_w=residual,
    step_dt=0.02,
    closed_loop=False,
  )
  torch.testing.assert_close(new_residual, residual + 0.02 * (v_filt - v_raw))
  torch.testing.assert_close(filtered, raw + new_residual)


def test_closed_loop_integrates_world_frame_regardless_of_robot() -> None:
  previous_filtered = torch.tensor([[3.0, 4.0]])
  v_filt = torch.tensor([[1.0, -2.0]])
  for raw in (torch.tensor([[3.0, 4.0]]), torch.tensor([[2.5, 4.4]])):  # robot moved or not
    filtered, residual = advance_filtered_root_xy(
      pre_filter_xy_w=previous_filtered,
      raw_root_xy_w=raw,
      raw_root_velocity_xy_w=torch.zeros(1, 2),
      filtered_velocity_xy_w=v_filt,
      translation_residual_xy_w=torch.zeros(1, 2),
      step_dt=0.02,
      closed_loop=True,
    )
    torch.testing.assert_close(filtered, previous_filtered + 0.02 * v_filt)
    torch.testing.assert_close(residual, filtered - raw)


def test_closed_loop_target_stops_running_ahead_of_a_compliant_robot() -> None:
  # Robot tracks the filtered velocity exactly: the residual must stay constant
  # in closed loop but grow without bound in open loop when v_raw != v_filt.
  dt = 0.02
  v_raw = torch.tensor([[0.0, 0.0]])
  v_filt = torch.tensor([[1.0, 0.0]])
  raw = torch.zeros(1, 2)
  filtered_closed = torch.zeros(1, 2)
  residual_closed = torch.zeros(1, 2)
  residual_open = torch.zeros(1, 2)
  for _ in range(50):
    raw = raw + dt * v_filt  # live alignment: raw root == robot root
    filtered_closed, residual_closed = advance_filtered_root_xy(
      pre_filter_xy_w=filtered_closed,
      raw_root_xy_w=raw,
      raw_root_velocity_xy_w=v_raw,
      filtered_velocity_xy_w=v_filt,
      translation_residual_xy_w=residual_closed,
      step_dt=dt,
      closed_loop=True,
    )
    _, residual_open = advance_filtered_root_xy(
      pre_filter_xy_w=raw + residual_open,
      raw_root_xy_w=raw,
      raw_root_velocity_xy_w=v_raw,
      filtered_velocity_xy_w=v_filt,
      translation_residual_xy_w=residual_open,
      step_dt=dt,
      closed_loop=False,
    )
  torch.testing.assert_close(residual_closed, torch.zeros(1, 2), atol=1e-5, rtol=0)
  torch.testing.assert_close(residual_open, torch.tensor([[1.0, 0.0]]), atol=1e-5, rtol=0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_closed_loop_root_target.py -q` — expected
ImportError on `advance_filtered_root_xy`.

- [ ] **Step 3: Implement** — add the helper next to `_limit_obstacle_velocity`;
in `_update_command` replace the pre-filter glue

```python
    self._filtered_root_xy_w[replay_env_ids] = (
      raw_root_pos[replay_env_ids, :2]
      + self._root_translation_residual_xy_w[replay_env_ids]
    )
```

with

```python
    if not self.cfg.closed_loop_root_target:
      self._filtered_root_xy_w[replay_env_ids] = (
        raw_root_pos[replay_env_ids, :2]
        + self._root_translation_residual_xy_w[replay_env_ids]
      )
    pre_filter_root_xy_w = self._filtered_root_xy_w.clone()
```

and the post-filter block (`_filtered_root_velocity_xy_w`, residual `+=`,
`_filtered_root_xy_w =`) with

```python
    self._filtered_root_velocity_xy_w[replay_env_ids] = result.velocity_w[replay_env_ids]
    filtered_xy_w, residual_xy_w = advance_filtered_root_xy(
      pre_filter_xy_w=pre_filter_root_xy_w,
      raw_root_xy_w=raw_root_pos[:, :2],
      raw_root_velocity_xy_w=raw_root_velocity[:, :2],
      filtered_velocity_xy_w=result.velocity_w,
      translation_residual_xy_w=self._root_translation_residual_xy_w,
      step_dt=self._env.step_dt,
      closed_loop=self.cfg.closed_loop_root_target,
    )
    self._filtered_root_xy_w[replay_env_ids] = filtered_xy_w[replay_env_ids]
    self._root_translation_residual_xy_w[replay_env_ids] = residual_xy_w[replay_env_ids]
```

Cfg field (after the propagation flags):

```python
  # Integrate the filtered root position in the world frame instead of
  # re-anchoring it to the live robot root each step, so the nominal root
  # position reward measures whether the robot actually executed the planar
  # escape. The recovery term still bounds the offset.
  closed_loop_root_target: bool = False
```

- [ ] **Step 4: Run** `uv run pytest tests/test_closed_loop_root_target.py tests/test_reference_filter.py tests/test_human_aware_mdp.py -q` — PASS. Any test
that pins open-loop residual arithmetic must still pass untouched.

## Task 3: Unified task — env cfg builder, registration, eval scripts, tests

**Files:**
- Modify: `src/safe_mimic/tasks/env_cfg.py` (after
  `unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg`)
- Modify: `src/safe_mimic/tasks/__init__.py` (constant near L66, registration
  after the FKC2 block ~L455, `__all__` ~L515)
- Modify: `scripts/benchmark_policy_reaction_envelope.py`,
  `scripts/diagnose_joint_compliance.py`, `scripts/benchmark_policy_avoidance.py`,
  `scripts/analyze_policy_failures.py`
- Test: `tests/test_coadjust_unified_task.py`

**Interfaces:**
- Consumes: Task 1/2 cfg fields `propagate_joint_corrections_to_body_targets`,
  `closed_loop_root_target`.
- Produces: `unitree_g1_lidar_unified_reference_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`;
  `LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID = "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified"`;
  experiment name `safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified`;
  `benchmark_policy_avoidance.py` task variant `"coadjust-unified"`.

Builder:

```python
_NOMINAL_TRACKING_REWARD_NAMES = (
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
  "action_rate_l2",
  "joint_limit",
  "self_collisions",
)


def unitree_g1_lidar_unified_reference_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Nominal tracking rewards on one fully filtered reference.

  Pipeline: LiDAR of randomized humans -> privileged planar + link CBF filters
  -> FK of every joint correction (anchor included) and closed-loop
  integration of the filtered root -> the stock mjlab tracking reward set on
  that reference. Every bespoke avoidance reward (planar velocity / progress /
  freeze, filtered-joint, urgent-escape, survival, proximity penalties) is
  removed; the two human collision terminations stay because the humans are
  ray-only geometry with no physics contact. The auxiliary teacher observation
  groups stay: they supervise the co-adjust head, they are not rewards.
  """
  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=play)
  motion = cfg.commands["motion"]
  assert isinstance(motion, PlanarFilteredReplayMotionCommandCfg)
  motion.propagate_joint_corrections_to_body_targets = True
  motion.closed_loop_root_target = True
  for name in tuple(cfg.rewards):
    if name not in _NOMINAL_TRACKING_REWARD_NAMES:
      cfg.rewards.pop(name)
  missing = set(_NOMINAL_TRACKING_REWARD_NAMES) - set(cfg.rewards)
  if missing:
    raise ValueError(f"nominal tracking rewards missing: {sorted(missing)}")
  return cfg
```

Registration mirrors the FKC2 block exactly (adjust on, offset 0, residual
gain 0.0, mix 1.0 -> 0.0 over 8000) with the new experiment name and
`env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg()`,
`play_env_cfg=unitree_g1_lidar_unified_reference_tracking_env_cfg(play=True)`.

Eval scripts: add `LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID` to every
`--task-id` choices tuple and to the `task_id`-dependent branches. For the
unified task the env cfg MUST be built by
`unitree_g1_lidar_unified_reference_tracking_env_cfg(play=...)` (not the
auxiliary builder plus patches) so the reward set and both flags match
training; the existing FKC/FKC2 patch branches must not fire for it. In
`benchmark_policy_avoidance.py` add `"coadjust-unified"` to the variant
Literal/choices, the `cfg_fn` map (value: the unified builder), the
`task_id` map, and EVERY membership tuple that lists `"coadjust-fkc2"`
(enumerate them in the report: expected 6 sites incl. L62, L141, L188, L235,
L256, L318, L512).

- [ ] **Step 1: Write the failing tests** (`tests/test_coadjust_unified_task.py`)

```python
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
)
from safe_mimic.tasks.env_cfg import (
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)

_EXPECTED_ACTOR_TERM_ORDER = (
  "command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions",
)


def test_reward_set_is_exactly_the_nominal_tracking_set() -> None:
  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  nominal = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)
  assert set(cfg.rewards) == set(nominal.rewards)
  for name, term in nominal.rewards.items():
    assert cfg.rewards[name].weight == term.weight, name
    assert cfg.rewards[name].params == term.params, name
    assert cfg.rewards[name].func is term.func, name


def test_terminations_are_nominal_plus_human_collisions() -> None:
  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  nominal = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)
  assert set(cfg.terminations) == set(nominal.terminations) | {
    "crowd_collision", "primary_human_collision",
  }


def test_filtered_reference_flags_and_actor_layout() -> None:
  for play in (False, True):
    cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(play=play)
    motion = cfg.commands["motion"]
    assert motion.propagate_joint_corrections_to_body_targets is True
    assert motion.propagate_arm_corrections_to_body_targets is False
    assert motion.closed_loop_root_target is True
    assert motion.expose_filtered_command is False
    assert motion.align_reference_to_robot_each_step is True
    assert motion.sampling_mode == ("start" if play else "uniform")
    assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERM_ORDER
    assert "avoidance_teacher" in cfg.observations
    assert "avoidance_robustness" in cfg.observations


def test_existing_tasks_untouched() -> None:
  for task_id in (
    LIDAR_AUXILIARY_COADJUST_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  ):
    motion = load_env_cfg(task_id).commands["motion"]
    assert motion.propagate_joint_corrections_to_body_targets is False
    assert motion.closed_loop_root_target is False
  aux = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg()
  assert "safe_planar_velocity" in aux.rewards
  assert "urgent_escape_progress" in aux.rewards


def test_motion_terminations_consume_the_filtered_reference() -> None:
  # User requirement (2026-09-03): motion-based terminations must evaluate the
  # filtered motion. mjlab's stock terminations read command.anchor_pos_w,
  # command.anchor_quat_w and command.body_pos_relative_w (recomputed from
  # body_pos_w + anchor pose in update_relative_body_poses); pin that the
  # unified task uses the stock functions AND that the filtered command class
  # overrides every property they consume.
  from mjlab.tasks.tracking import mdp as tracking_mdp

  from safe_mimic.tasks.kinematic_replay_command import PlanarFilteredReplayMotionCommand

  cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg()
  assert cfg.terminations["anchor_pos"].func is tracking_mdp.bad_anchor_pos_z_only
  assert cfg.terminations["anchor_ori"].func is tracking_mdp.bad_anchor_ori
  assert cfg.terminations["ee_body_pos"].func is tracking_mdp.bad_motion_body_pos_z_only
  overridden = PlanarFilteredReplayMotionCommand.__dict__
  for name in ("anchor_pos_w", "anchor_quat_w", "body_pos_w", "body_quat_w"):
    assert isinstance(overridden[name], property), name


def test_unified_task_registered_like_coadjust() -> None:
  env_cfg = load_env_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID)
  assert env_cfg.commands["motion"].closed_loop_root_target is True
  rl_cfg = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID)
  base = load_rl_cfg(LIDAR_AUXILIARY_COADJUST_TASK_ID)
  assert rl_cfg.actor.adjust_command_with_joint_prediction is True
  assert rl_cfg.actor.command_joint_pos_offset == 0
  assert rl_cfg.actor.avoidance_joint_action_residual_gain == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_start == 1.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_end == 0.0
  assert rl_cfg.algorithm.avoidance_teacher_mix_decay_updates == 8000
  assert rl_cfg.experiment_name == (
    "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_unified"
  )
  assert rl_cfg.actor.__class__ is base.actor.__class__
```

(Adjust `rl_cfg.experiment_name` to wherever `_perceptive_lidar_runner_cfg`
stores it — check `tests/test_coadjust_fkc2_lidar_auxiliary_task.py` for the
attribute path it asserts and use the same.)

- [ ] **Step 2: Run** `uv run pytest tests/test_coadjust_unified_task.py -q` — FAIL (import errors).
- [ ] **Step 3: Implement** builder, constant, registration, `__all__`, eval scripts.
- [ ] **Step 4: Run** the new test file, `tests/test_task_runner_cfg.py`,
`tests/test_coadjust_fkc2_lidar_auxiliary_task.py`, then the FULL suite
`uv run pytest -q` — all PASS; `uv run ruff check` clean.
- [ ] **Step 5: 8-env strict-load smoke (CPU or GPU-light, no training):**
throwaway scratchpad script that builds the play env for the unified task
with `num_envs=8`, steps it 5 times with zero actions, and asserts
`env.reward_manager` active term names == the nominal nine, that
`command.anchor_pos_w`, `command.body_pos_w` and
`command.filtered_root_velocity_xy_w` are finite, and that the actor obs dim
is 154. Report the numbers. This exercises whole-body chain construction on
the real G1 (topology errors raise here).

## Task 4 (controller): reviews, stop FKC2, launch, gates, docs

1. Per-task review (opus; fall back to fresh sonnet on 529s); final
   whole-branch review focusing on: bit-identity of every existing task,
   whole-body chain correctness on G1 (Task 1 step 5 / Task 3 step 5
   evidence), closed-loop semantics through reset/wrap, eval-script parity.
2. Stop the FKC2 run (`coadjust_fkc2_scratch_4096`, PID 503642) after the
   final review passes; record its last iteration and checkpoints in the ledger.
3. Launch from scratch (standing authorization), ALWAYS via `run_in_background`:

```bash
uv run train SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified \
  --env.scene.num-envs 4096 --agent.run-name coadjust_unified_scratch_4096
```

   Watchdog + milestone monitors (1k/5k/10k) with fatal signatures.
4. Health at 1k/5k: reward decomposition (only the nine nominal terms),
   `Episode_Termination/ee_body_pos` (should stay low: targets now consistent
   for legs too), episode length, `filter_reference_offset_m` metric (root
   residual; expect larger than open-loop runs while the pelvis learns).
5. Gates at 5k/10k (same protocol as FKC/FKC2): envelope 1024 / compliance 512
   (state compliance arms+legs, planar escape_ratio) / packed 1024 with
   `--task-id`/`--task-variant` set to the unified task. Baselines: FKC@10k
   envelope 53.4% / 40 falls, arm t 0.568, escape_ratio 0.47, packed 47.2%.
6. Ledger `.superpowers/sdd/2026-09-03-phase3-unified-filtered-reference/progress.md`
   updated at every ruling; `handoff.md` executive summary gets item 7
   (Phase 3) at launch and after the 10k gate.

## Task ordering

Task 1 -> Task 2 (same file, same fork) -> Task 3 (sonnet) -> Task 4
(controller). Reviews after Task 2 (covers 1+2) and after Task 3.
