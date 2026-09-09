# Phase 2c: FK-consistent task-space objective for limb corrections

User direction (2026-09-02): build this, then stop the current coadjust run;
the fixed objective becomes the next from-scratch run (launch authorized).

Root cause being fixed (verified in code): the filtered command corrects the
JOINT-space reference (`joint_pos`/`joint_vel`) but its task-space body cloud
(`body_pos_w`) gets only the rigid planar root offset — limb corrections are
never propagated (kinematic_replay_command.py:475-483). Consequently
`motion_body_pos`/`motion_body_ori` rewards and, critically, the
`ee_body_pos` termination (`bad_motion_body_pos_z_only`, threshold 0.25 m on
wrist/ankle HEIGHT vs the raw-shaped reference) punish obedience: a
shoulder-led arm drop moves the wrist down 0.4-0.6 m, so full compliance
terminates the episode. This caps execution cosine at ~0.10-0.19 across every
lineage and explains most of the exposure-gate "falls".

Fix: single source of truth. `body_pos_w` (and `anchor`-independent
consumers) return arm-body positions corrected by reference-frame FK of the
filtered joint residual. Every consumer — rewards, termination, critic obs,
metrics — becomes consistent automatically. Deployment is unaffected
(training-time privileged computation only).

## Global constraints

- No git state changes; `uv run ruff check` clean; no whole-file `ruff
  format` on tracked files. Full `uv run pytest -q` green (baseline 191).
- GPU: the coadjust training run occupies it. CPU tests only; at most one
  tiny env-introspection smoke (<= 8 envs) if truly needed; NEVER `uv run
  train`.
- Backward compatibility: default-off; every existing task, benchmark, and
  checkpoint behaves identically unless the new flag is set.
- 2-space indent, mirror file conventions.

## Task 1: FK propagation in the command (core)

In `src/safe_mimic/tasks/kinematic_replay_command.py`:

New cfg field on `PlanarFilteredReplayMotionCommandCfg`:
`propagate_arm_corrections_to_body_targets: bool = False`.

When enabled, `PlanarFilteredReplayMotionCommand` computes, once per
`_update_command` (cache the result; properties must not recompute):

1. Identify the arm chains once at init: joints whose names contain
   shoulder/elbow/wrist tokens (reuse `cfg.link_filter.arm_joint_name_tokens`)
   and, for the REFERENCE body cloud, the cloud indices of bodies that are
   kinematic descendants of any arm joint. Derive the mapping cloud-index ->
   global body id -> parent chain from the motion command's tracked body
   names and `mj_model` (`body_parentid`, `jnt_bodyid`; patterns exist in
   `_build_link_joint_ancestry`).
2. Compute reference-pose joint frames for the arm joints WITHOUT touching
   sim state: anchor_w = raw_ref_body_pos[parent_body] +
   quat_rotate(raw_ref_body_quat[parent_body], mj_model.jnt_pos[joint]);
   axis_w = quat_rotate(raw_ref_body_quat[jnt_body's parent frame — verify
   MuJoCo's convention: jnt_pos/jnt_axis are in the frame of the body the
   joint belongs to (`jnt_bodyid`'s body frame), so rotate by that body's
   PARENT-side pose; establish the exact convention from the MuJoCo docs in
   the installed source and verify with the analytic test below). Use the
   RAW (unaligned-corrected) reference cloud that `_raw_body_pos_w()` /
   `_raw_body_quat_w()` return so alignment is inherited.
3. Rodrigues-rollout the arm-descendant cloud positions by the joint deltas
   `filtered_joint_pos - _raw_joint_pos()` restricted to arm joints, in
   root-to-leaf order — generalize `_rollout_candidate_link_positions` (or
   factor a shared helper) to take (positions, axes, anchors, deltas) as
   arguments instead of reading sim data, with a single candidate.
   Positions only; body orientations stay raw (document the approximation).
4. Mask: env rows whose arm residual max-abs < 1e-4 keep raw targets
   (skip compute where convenient; correctness first, the arm subset is
   small either way).
5. `body_pos_w` returns raw + planar offset + this arm displacement for the
   affected cloud indices. `anchor_pos_w`/`anchor_quat_w` (torso/pelvis
   anchor) unaffected by construction — assert the anchor body is not an
   arm descendant at init. Body VELOCITY targets stay uncorrected
   (document: residual rates are bounded; positions dominate the conflict).

Tests (new file, CPU): (a) analytic single-joint case — build a tiny
synthetic chain (or use the real model constants on CPU via mujoco bindings
if the existing tests do that; otherwise pure-tensor synthetic frames) where
rotating an "elbow" by theta must move the "wrist" along the known arc;
verify the FK displacement matches trig to 1e-5; (b) zero residual => zero
displacement, non-arm bodies always untouched; (c) default-off mode:
`body_pos_w` bit-identical to today; (d) caching: two property reads, one
compute.

## Task 2: task registration + wiring + consistency tests

1. Register `LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID =
   "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-FKC"`
   in `src/safe_mimic/tasks/__init__.py`: identical to the CoAdjust
   registration (adjust on, offset 0, residual gain 0.0, mix 1.0/0.0/8000)
   plus an env-cfg parameter that sets
   `propagate_arm_corrections_to_body_targets = True` on the motion command
   (add a keyword arg to `unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg`
   mirroring how `expose_filtered_command` was added; default False).
   Experiment name
   `"safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_fkc"`.
   Add to `__all__`.
2. Extend the eval scripts' task selection with the new task id:
   `--task-id` choices in `scripts/benchmark_policy_reaction_envelope.py` and
   `scripts/diagnose_joint_compliance.py`; a `"coadjust-fkc"` variant in
   `scripts/benchmark_policy_avoidance.py` (same env builder; treat like
   "coadjust"/"auxiliary" at every task_variant use-site — enumerate them in
   the report). NOTE: eval env cfgs for this task should ALSO set the
   propagation flag True so reported task-space metrics are consistent with
   training (wire through the same env-cfg parameter; benchmarks' encounter
   pins are untouched).
3. Tests mirroring tests/test_coadjust_lidar_auxiliary_task.py: registration
   values, propagation flag default-off for all other tasks, base tasks
   unchanged.

## Task 3 (controller): gate, stop, relaunch

Full pytest + reviews. Then: stop the running coadjust training (user
directive: after the build), archive its final TB state in the ledger,
launch from scratch:

```bash
uv run train SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-FKC \
  --env.scene.num-envs 4096 --agent.run-name coadjust_fkc_scratch_4096
```

Monitors + the standard gates (compliance pair, envelope, packed) at 10k/15k.
Success criteria: execution cosine h25 well above 0.19 (the ceiling should be
gone), falls ~0, envelope < 59.5% (coadjust@10k) trending toward the 10.5%
teacher ceiling; watch `Episode_Termination/ee_body_pos` — it should no
longer punish corrections.

## Task ordering

Task 1 first (defines the cfg field Task 2 wires). Task 2 after. Reviews per
task; final whole-branch review before the stop-and-relaunch.
