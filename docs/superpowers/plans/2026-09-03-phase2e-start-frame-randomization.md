# Phase 2e: randomize the motion start frame in training

Finding (final review of Phase 2d): `PlanarFilteredReplayMotionCommand._resample_command`
(src/safe_mimic/tasks/kinematic_replay_command.py) hard-codes `time_steps = 0` and
`reset_to_frame(env_ids, 0)` — inherited from the exact-replay demo class — so with
10 s episodes every training run has only ever seen the first 10 s of the 131 s
dance. Upstream `MotionCommand._resample_command` (mjlab commands.py:320-327) honors
`cfg.sampling_mode` in {"adaptive" (default), "uniform", "start"}; the demo cfgs set
"start" explicitly (env_cfg.py:952, :1103); the avoidance training cfgs never set it.
User direction (2026-09-03): randomize the starting point. FKC2 run stopped at
iteration 1 to relaunch with this fix.

## Global constraints
No git state changes; ruff clean hunks-only; full `uv run pytest -q` green (baseline
222); CPU tests; at most one 8-env smoke per task; never `uv run train`; existing demo
behavior (frame 0) preserved via their explicit `sampling_mode="start"`.

## Task 1: honor sampling_mode in the filtered command (+ training cfg)
1. In `PlanarFilteredReplayMotionCommand._resample_command`: replace the hard-coded
   frame-0 logic with start-time sampling per `self.cfg.sampling_mode` — "start" ->
   frame 0 (bit-identical current behavior); "uniform" -> uniform random time step in
   [0, time_step_total); "adaptive" -> raise NotImplementedError with a clear message
   (adaptive bin bookkeeping is not verified through this command's overrides — do NOT
   silently fall back). Prefer reusing upstream machinery (e.g. calling
   `MotionCommand._resample_command(self, env_ids)` when it does exactly the right
   thing, or its sampling helpers) over reimplementing; read commands.py:300-340 and
   `reset_to_frame` (:589) to decide, and document why.
2. Order of operations must remain correct: sampled `time_steps` set FIRST; robot reset
   to the sampled frame (with the cfg's pose/velocity/joint randomization as upstream
   does); `_update_reference_alignment(env_ids)` against the reset robot pose; filter
   state initialized from the RAW reference AT THE SAMPLED FRAME (`_filtered_root_xy_w`
   = raw root xy, velocities, joint filter state, residuals zero, obstacle history
   reset, arm-target caches zeroed + velocity hold set — everything the current method
   does, just at the sampled frame). Check whether `reset_to_frame` itself calls
   `_update_command`/`update_relative_body_poses` and avoid double work or stale
   alignment.
3. `KinematicReplayMotionCommand._resample_command` (parent, used by the kinematic demo
   cfgs which set "start"): honor sampling_mode the same way, or leave as-is if it is
   only ever used with "start" — state which and why.
4. Training cfg: in `unitree_g1_lidar_avoidance_tracking_env_cfg` set
   `filtered_command_cfg.sampling_mode = "uniform"` (propagates to range-rate,
   auxiliary, coadjust, FKC, FKC2). Demo cfgs untouched ("start").
5. Tests: "start" -> all time_steps 0 and bit-identical filter init vs today;
   "uniform" -> time_steps spread (not all equal, within range) and
   `_filtered_root_xy_w[env] == raw root xy at that env's sampled frame`;
   "adaptive" -> raises; cfg tests: avoidance-family cfgs have "uniform", demo cfgs
   "start".

## Task 2: pin evaluation to frame 0 for comparability
All existing benchmark artifacts were measured from frame 0. In
scripts/benchmark_policy_reaction_envelope.py, scripts/benchmark_teacher_envelope.py,
scripts/benchmark_policy_avoidance.py (in `_build_cfg`), scripts/analyze_policy_failures.py,
scripts/diagnose_joint_compliance.py: set `cfg.commands["motion"].sampling_mode = "start"`
right after the cfg is built (one line + comment each). Also add an opt-in
`--random-start` flag to reaction_envelope and diagnose_joint_compliance that sets
"uniform" instead (recorded in the output JSON) for future randomized-window evals.
ruff clean; no execution needed beyond an argparse sanity check (CPU).

## Task 3 (controller): gate + relaunch FKC2
pytest; reviews; relaunch `uv run train ...-CoAdjust-FKC2 --env.scene.num-envs 4096
--agent.run-name coadjust_fkc2_scratch_4096`; verify `Metrics/motion/sampling_*` or
time-step spread is live; standard gates at 10k.
