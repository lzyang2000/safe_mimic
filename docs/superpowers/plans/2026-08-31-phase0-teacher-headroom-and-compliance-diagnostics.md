# Phase 0: teacher-headroom and compliance diagnostics

Approved in-chat 2026-08-31. Goal: before committing a training cycle, measure
(a) the collision-rate ceiling of the privileged reference filter itself,
(b) whether the model_21000 policy mechanically/behaviorally complies with the
arm residual path, and (c) the exact termination causes in the reaction
envelope. Results gate Phase 1 (consolidated retraining) vs Phase 2 (filter
retuning first).

Spec authority: this file plus `handoff.md` ("Required diagnostics before
changing the filter again"). Where they conflict, this file wins.

## Global constraints

- **No git commits, no `git add/restore/reset/checkout`.** The dirty working
  tree is the source of truth (see handoff.md) and contains unrelated user
  changes. New files are simply left untracked.
- **No changes under `src/`.** Scripts are self-contained, mirroring the
  existing `scripts/benchmark_*.py` conventions (module docstring, argparse,
  `main()`, 2-space indent, `gc.collect()` + `torch.cuda.empty_cache()`
  cleanup).
- Lint: `uv run ruff check <file>` and `uv run ruff format <file>` must pass.
- **GPU discipline:** implementers may run at most one smoke test at a time,
  with `--num-envs <= 16` and short horizons (`--episode-length-s <= 2` or
  `--total-steps <= 60`). Full-size runs are executed by the controller only.
  If CUDA OOM occurs, wait 60 s and retry once.
- Implementers never spawn subagents.
- Reference checkpoint (for smoke tests needing one):
  `logs/rsl_rl/safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_joint_robust/2026-08-31_00-16-59_auxiliary_joint_robust_4096/model_21000.pt`
- Reference motion: `artifacts/motions/lafan1_dance1_subject1_demo_motion.npz`
- Existing scripts to copy patterns from (read them first):
  `scripts/benchmark_policy_reaction_envelope.py` (env setup for the
  benchmark encounter distribution, sampler access, `_minimum_clearance`,
  `_bin_summary`/`_grid_summary`/`_print_table`, JSON payload shape) and
  `scripts/benchmark_policy_avoidance.py` (runner/policy loading, per-term
  termination causes via `raw_env.termination_manager.active_terms`,
  actor access for auxiliary predictions).

## Task 1: `scripts/benchmark_teacher_envelope.py` (new file)

Measure the collision rate of the privileged teacher itself: the robot is
kinematically slaved to the filtered reference (no policy, no dynamics), under
the same encounter distribution as the reaction-envelope benchmark. This is
the upper bound for any student policy.

CLI (argparse, all with these exact defaults):
`--motion-file` (Path, required), `--num-envs` 2048, `--seed` 31, `--device`
"cuda:0", `--min-radius-m` 0.75, `--max-radius-m` 4.0, `--min-intercept-s`
0.5, `--max-intercept-s` 4.0, `--episode-length-s` 6.0, `--output` (Path,
required).

Environment construction:
1. `cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=True)`
   then apply exactly the same modifications as
   `benchmark_policy_reaction_envelope.py` lines 159-184: seed, num_envs,
   `episode_length_s`, motion file, pop `push_robot`, crowd event min/max
   count 0 + `obstacle_free_probability` 0 + `randomize_density` False,
   primary event radius/delay from args, `show_mesh`/`print_velocity` False,
   sensor `debug_vis` False, all observation groups `enable_corruption =
   False`, lidar `noise_cfg` None.
2. Kinematic-teacher mode on the motion command (assert it is a
   `PlanarFilteredReplayMotionCommandCfg`):
   - `write_reference_to_sim = True`
   - `align_reference_to_robot_each_step = False`
   - copy the zero-G kinematic setup from
     `unitree_g1_kinematic_reference_lidar_demo_env_cfg` in
     `src/safe_mimic/tasks/env_cfg.py`: `cfg.sim.mujoco.gravity = (0,0,0)`;
     `cfg.sim.mujoco.disableflags = tuple(dict.fromkeys((*cfg.sim.mujoco.disableflags, "contact", "actuation")))`
   - `cfg.terminations.clear()` (collision is measured analytically below).
3. No checkpoint, no runner, no policy: build `ManagerBasedRlEnv(cfg=cfg,
   device=...)` directly, call its reset once, then step it with zero actions
   of shape `(num_envs, action_dim)` each control step (verify the env's step
   signature from the installed mjlab source; actuation is disabled and the
   command overwrites robot state, so actions are inert).

Measurement:
- After reset, record per env exactly as the reaction benchmark does:
  `scheduled_ttc_s` (sampler `global_intersection_times_s` minus current
  global time), `initial_root_distance_m`, `nominal_approach_speed_mps`,
  `initial_clearance_m` (reuse the `_minimum_clearance` logic: minimum
  `mdp.capsule_link_surface_clearances` over
  `command.cfg.link_filter.body_names` with
  `command.cfg.link_filter.link_radius_m`).
- Step loop: run `int(round(episode_length_s / raw_env.step_dt))` steps (do
  NOT rely on `max_episode_length` or dones; terminations are cleared). Track
  running per-env minimum clearance at the top of each iteration and once
  more after the loop.
- Teacher saturation diagnostics, accumulated per env from
  `command.metrics[...]` after each step: max of
  `filter_intervention_speed_mps`, max of `filter_cbf_violation_mps`, max of
  `link_filter_cbf_violation_mps`, and the fraction of steps where
  `filter_intervention_speed_mps >= 0.97 * cfg.planar_filter.max_intervention_speed_mps`
  (planar-cap saturation fraction).
- Collision per env: `minimum_clearance_m < 0.1` at any measured step (same
  0.1 m margin the training termination uses).

Output:
- JSON payload mirroring the reaction benchmark: a `configuration` block, a
  `summary` (episodes, collisions, collision_rate, mean scheduled TTC /
  initial distance / nominal speed, plus a `teacher_saturation` sub-dict with
  mean and p90 across envs of each saturation diagnostic), then
  `scheduled_ttc_bins`, `initial_distance_bins`, `nominal_speed_bins`,
  `initial_clearance_bins`, `distance_speed_grid`, and per-env `cases` —
  reuse (copy into this script) `_bin_summary`, `_grid_summary`,
  `_print_table` and the exact bin edges from the reaction benchmark.
- Print the same tables to stdout plus the saturation summary.

Acceptance:
- `uv run ruff check` / `format` clean.
- Smoke run completes and writes valid JSON:
  `uv run python scripts/benchmark_teacher_envelope.py --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz --num-envs 8 --episode-length-s 2 --output /tmp/claude-1000/-home-yangl-twist2-safe-mimic/35b8dced-b6f5-4f85-b4a5-e05b7bdafddc/scratchpad/teacher_smoke.json`
- In the smoke output, scheduled TTC values fall in [0.5, 4.0] (+/- one step),
  distances in [0.75, 4.0], and no NaNs in the summary.

## Task 2: `scripts/diagnose_joint_compliance.py` (new file)

The handoff's four compliance signals, per arm side, on active avoidance
frames, for an auxiliary checkpoint with the explicit joint-residual path.

CLI: `--checkpoint` (Path, required), `--motion-file` (Path, required),
`--num-envs` 512, `--seed` 31, `--device` "cuda:0", `--min-radius-m` 0.75,
`--max-radius-m` 4.0, `--min-intercept-s` 0.5, `--max-intercept-s` 4.0,
`--total-steps` 600, `--active-threshold-rad` 0.05, `--output` (required).

Environment + policy:
- Env cfg exactly as `benchmark_policy_reaction_envelope.py` lines 159-184
  (episode_length_s stays at its default 6.0; the run continues across
  auto-resets up to `--total-steps`).
- Load runner + checkpoint exactly as that script does;
  `policy = runner.get_inference_policy(device=...)`. Assert
  `isinstance(policy, PerceptiveLidarActor)` and
  `policy.avoidance_joint_action_residual_gain > 0`.
- `observations, _ = env.reset()` after loading (restored-clock rule).
- Step the wrapper with normal `policy(observations)` actions inside
  `torch.inference_mode()`.

Definitions:
- `joint_names = tuple(raw_env.scene["robot"].joint_names)` (this order
  matches both the action space and the teacher's joint block, per the
  comment in `src/safe_mimic/tasks/__init__.py`).
- Left-arm joint mask: name starts with `left_` and contains one of
  `shoulder|elbow|wrist` (7 joints); right-arm analogous.
- Per step: `teacher = observations["avoidance_teacher"]` (31 values: 2
  planar + 29 joint radians); `pred = policy.predict_avoidance(observations)`;
  joint blocks are `[..., 2:]`.
- Added normalized residual: `gain * mask * pred_joints / scales` computed
  from the actor's buffers (`avoidance_joint_action_mask`,
  `avoidance_joint_action_scales`, `avoidance_joint_action_residual_gain`).
  Once, at the first step, verify it matches
  `policy._joint_action_residual(pred)` to within 1e-6 and record the max
  deviation in the output (`# noqa: SLF001` on that line).
- Active frame (per side): max abs teacher joint residual over that side's
  arm joints >= `--active-threshold-rad`.
- Achieved joint motion: ring buffers (length 26) of `robot.data.joint_pos`
  and of done flags; for horizons h in {5, 25} steps (0.1 s / 0.5 s),
  `dq = q[t+h] - q[t]`, valid only when no done occurred in (t, t+h].
  Windows crossing an env reset are discarded via the done ring buffer.
- Side arm clearance: minimum of `mdp.capsule_link_surface_clearances`
  restricted to that side's {shoulder_roll, elbow, wrist_yaw} links (their
  indices within `command.cfg.link_filter.body_names`) against the primary
  human capsules, with the link filter's `link_radius_m`.

Metrics (accumulate sums/counts online on GPU; do not store full traces),
reported per side and pooled:
1. Prediction quality on active frames: MAE of `pred_side` vs `teacher_side`
   (arm joints of that side), mean cosine similarity between the two vectors.
2. Mechanical path: mean L2 of the added normalized residual on active
   frames; the recorded max deviation from `_joint_action_residual`.
3. Execution over each horizon h (active frames with valid windows): mean
   cosine(dq_side, teacher_side), mean cosine(dq_side, pred_side), mean
   projection of dq_side onto the unit teacher_side direction (radians), and
   mean L2 of dq_side.
4. Clearance response: on active frames where that side's arm clearance <
   0.8 m and the 25-step window is valid, the fraction where
   clearance(t+25) - clearance(t) > 0.
Also report: overall active-frame rate per side, and planar compass quality:
mean cosine(pred[..., :2], teacher[..., :2]) over frames with
`||teacher_planar|| >= 0.1` m/s.

Output: JSON (config, per-side metric dicts, pooled) + a readable printed
summary organized by the four signals.

Acceptance:
- ruff clean; smoke run completes and prints/writes plausible values:
  `uv run python scripts/diagnose_joint_compliance.py --checkpoint <reference checkpoint above> --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz --num-envs 8 --total-steps 60 --output /tmp/claude-1000/-home-yangl-twist2-safe-mimic/35b8dced-b6f5-4f85-b4a5-e05b7bdafddc/scratchpad/compliance_smoke.json`
- Cosine values in [-1, 1]; counts consistent (valid windows <= active
  frames); no NaNs when counts are nonzero (guard zero-count divisions).

## Task 3: termination-cause breakdown in `scripts/benchmark_policy_reaction_envelope.py` (edit)

Today the script only distinguishes collision / timeout / other (210 of 2048
episodes land in "other" unexplained). Add per-term cause tracking:

- Copy the pattern from `scripts/benchmark_policy_avoidance.py` lines
  294-320: before the loop, build
  `causes = {name: torch.zeros(num_envs, dtype=torch.bool, device=...) for name in raw_env.termination_manager.active_terms}`;
  on `new_done`, record `raw_env.termination_manager.get_term(name)[new_done]`
  for every term. Derive the existing `collision`, `timeout`, `other_failure`
  tensors from `causes` so current semantics are unchanged
  (collision = crowd | primary collision terms; timeout = `time_out`).
- Add to the JSON `summary`: `"termination_causes": {term_name: count}`
  counted over completed episodes.
- Add to each per-case record a `"termination_cause"` string: the first
  fired term in the order [primary_human_collision, crowd_collision, other
  terms alphabetically, time_out], or `null` if the episode never completed.
- Print a "Termination causes" table (term, count, share of completed).
- Change nothing else; all existing JSON keys keep their meaning.

Acceptance: ruff clean; `git diff -- scripts/benchmark_policy_reaction_envelope.py`
touches only this feature; smoke run with
`--num-envs 8 --episode-length-s 2 --output <scratchpad>/reaction_smoke.json`
using the reference checkpoint and motion completes and shows the new table.

## Task 4 (controller-only): full GPU runs and synthesis

Serial, after tasks land: (D, already running) correction ablation on
model_21000; (E) teacher envelope at 2048 envs seed 31; (F) compliance
diagnostics at 512 envs on model_21000; (G) reaction envelope re-run with
termination breakdown to
`artifacts/benchmarks/lidar_avoidance_auxiliary_joint_robust_model_21000_reaction_envelope_termbreakdown_2048.json`.
Synthesize into the Phase-1/Phase-2 decision gate.
