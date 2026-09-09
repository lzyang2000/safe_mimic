# Phase 2d: complete arm-compliance consistency

User direction (2026-09-03): fix arm compliance first. Post-mortem shows FKC
arm state compliance at t~0.57 / 51% of active frames. Remaining
inconsistencies (audited): body LINEAR velocity targets for arm bodies are
still raw (motion_body_lin_vel, weight 1.0, std 1.0 -> ~10-15% penalty while
an arm correction executes); the joint-space pull is diluted 29:1
(filtered_joint_position averages over all joints). Both fixes are reward-side
and deploy-safe. A parallel diagnosis (self- vs oracle-injection compliance)
decides whether the adjuster also needs strengthening (separate follow-up).

## Global constraints
- No git state changes; ruff clean hunks-only (no whole-file format on tracked
  files); full `uv run pytest -q` green (baseline 205). CPU tests only; at most
  one 8-env smoke per task; never `uv run train`. Default-off; existing tasks
  unchanged.

## Task 1: arm body VELOCITY target propagation (kinematic_replay_command.py)
When `propagate_arm_corrections_to_body_targets` is True, also correct the
arm bodies' linear velocity targets: cache the previous step's per-body
position displacement (the tensor already computed for `body_pos_w`) and set
`body_lin_vel_w` for arm cloud indices to raw + (disp_t - disp_{t-1}) /
step_dt. Reset the previous-displacement cache to the current displacement
(so the first step after reset/wrap contributes zero velocity correction)
on `_resample_command` and on wrap/uninitialized rows. Angular velocity
targets stay raw (std 3.14 rad/s makes them negligible; document). Tests:
constant displacement -> zero velocity correction; linearly growing
displacement -> exact finite-difference velocity; reset clears; non-arm
bodies untouched; default-off bit-identical `body_lin_vel_w`.

## Task 2: active-correction arm tracking reward + FKC2 task
1. `mdp.active_correction_joint_tracking_exp(env, command_name, std=0.2,
   activation_threshold_rad=0.05)`: per env, weights w_j = 1 where the
   teacher residual |filtered_j - raw_j| >= threshold else 0 (over ALL 29
   joints — legs included if the filter corrects them); error =
   sum_j w_j (q_robot_j - q_filtered_j)^2 / max(sum_j w_j, 1); reward =
   exp(-error/std^2) where any joint is active, else 0. Pure helper factored
   for CPU tests: zero when nothing active; 1 at the corrected pose; falls
   with error; inactive joints ignored.
2. Env cfg kwarg `active_correction_reward: bool = False` on
   `unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg`; when True adds
   reward `"active_correction_joint_tracking"` weight 1.5, params
   {command_name "motion", std 0.2, activation_threshold_rad 0.05}.
3. Register `LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID =
   "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-FKC2"`
   = FKC registration + `active_correction_reward=True` on env and play
   cfgs; experiment_name
   "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust_fkc2";
   add to __all__. Wire the id into the three eval scripts exactly as FKC
   was (task-id choices / "coadjust-fkc2" variant; set BOTH propagation and
   active-correction flags when selected).
4. Tests mirroring tests/test_coadjust_fkc_task.py (or its actual name).

## Task 3 (controller): gate + launch decision
pytest; reviews; then launch from scratch
`uv run train ...-CoAdjust-FKC2 --env.scene.num-envs 4096 --agent.run-name coadjust_fkc2_scratch_4096`
(standing authorization), gates at 10k: arm state compliance t (target >= 0.8,
>= 75% of active frames) and stable-frame variant; envelope; falls.
