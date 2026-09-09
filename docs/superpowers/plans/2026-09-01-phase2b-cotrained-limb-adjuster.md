# Phase 2b: co-trained limb adjuster (reference-adjustment through the command channel)

User direction (2026-09-01): no fine-tuning — co-train the motion adjuster and
the mimic policy FROM SCRATCH. Limbs only; no pelvis/root channel of any kind.

Architecture: the auxiliary head's predicted joint residuals are added to the
joint-position targets inside the actor's OBSERVED command (the reference the
mimic tracks), replacing the old ignorable-conditioning + action-residual
paths. During training rollouts the injected correction is
`mix * teacher + (1 - mix) * prediction` with mix annealing 1.0 -> 0.0, plus
the existing bounded conditioning noise. At inference the actor injects its
own prediction — deployment-correct by construction.

Evidence base (see the Phase 1/2a ledger): exposing filter-adjusted joints to
an untrained policy already doubled command-following and sped escape but
destabilized it (39% falls) — the mechanism works, obedience must be trained
in from the start.

## Global constraints

- No git commits or state changes. `uv run ruff check` clean on touched
  files; NO whole-file `ruff format` on tracked files (hunks-only styling).
- Full `uv run pytest -q` green (baseline 176) — new tests add to it.
- GPU is free until the training launch; smoke tests <= 16 envs, one at a
  time.
- Backward compatibility: every existing task, checkpoint contract, and
  benchmark must behave exactly as before when the new mode is off. Old
  checkpoints must still strict-load into actors built without the new mode.
- 2-space indent, mirror existing file conventions.

## Task 1: model + algorithm (command-slice adjustment path)

All in `src/safe_mimic/rl/perceptive_lidar.py` unless noted.

New `PerceptiveLidarModelCfg` fields (defaults preserve current behavior):
- `command_joint_pos_offset: int = -1` — start index of the 29 joint-position
  command values inside the flattened tracking ("actor") observation vector;
  -1 disables the adjustment path.
- `adjust_command_with_joint_prediction: bool = False` — master switch.
- Validation in the actor: if enabled, require `avoidance_joint_dim > 0`,
  `0 <= offset` and `offset + avoidance_joint_dim <= tracking_obs_dim`.

Actor forward semantics when the mode is enabled (`_forward` and every path
that produces actions):
1. Adjuster pass: encode the RAW observations exactly as today (tracking
   encoder on the raw tracking obs, LiDAR encoder, timing) -> base latent ->
   `prediction = tanh(head(latent)) * output_scale` (unchanged).
2. Injection: `adjusted_tracking = tracking.clone()`;
   `adjusted_tracking[..., offset:offset+29] += correction_joint_rad` where
   the correction is, in training-rollout paths, the CONDITIONING (teacher
   mix + noise, computed by the algorithm as today) and, in plain
   forward/inference paths, the actor's own `prediction[..., planar_dim:]`.
   Position targets only — the joint-velocity command values stay raw.
   The addition happens on the UNNORMALIZED observation; the empirical
   normalizer then applies to the adjusted vector for the mimic pass.
3. Mimic pass: tracking encoder applied to the adjusted tracking obs; mimic
   latent = cat(adjusted tracking latent, lidar latent, timing) — when the
   mode is enabled the 31-D conditioning is NOT concatenated to the MLP input
   (the correction is visible in the command; keep `_get_latent_dim`
   consistent), and `_joint_action_residual` is bypassed (the runner cfg will
   set gain 0 for the new task; additionally gate the residual off when the
   mode is on, with a comment). MLP -> distribution as today.
   Note this means two tracking-encoder evaluations per step (raw for the
   adjuster, adjusted for the mimic) — intended; the adjuster conditions on
   the nominal reference.
4. `update_normalization`: update statistics on the RAW tracking obs only
   (a single consistent convention; document it).

`_ExportPerceptiveLidarActor`: replicate the enabled-mode two-pass path
exactly (own prediction injected); verify actor-vs-export parity in a test.

`AvoidanceAuxiliaryPPO`: `act()` and `update()` route the conditioning into
the command slice via a shared actor method (e.g.
`_forward_with_command_adjustment(obs, correction, stochastic_output)`)
instead of `_forward_avoidance_latent` when the mode is enabled; the
supervised losses, diagnostics, and storage stay unchanged. New algorithm cfg
values are NOT new fields — reuse `avoidance_teacher_mix_start/end/decay`
(the new task's cfg will set 1.0 / 0.0 / 8000).

Checkpoint compatibility: no new persistent state. Any new buffers follow the
non-persistent pattern used for the action-scale buffers.

Tests (`tests/test_perceptive_lidar_actor.py` additions or a new file):
- Injection correctness: with a hand-built obs TensorDict, enabled-mode
  forward sees `tracking[offset:offset+29] + prediction_joint` in the mimic
  pass (verify via monkeypatched encoder capturing its input) and raw values
  in the adjuster pass.
- mix=1 routing: the algorithm-side path injects exactly the teacher target
  (noise zeroed) into the slice.
- Velocity targets untouched; values outside the slice untouched.
- Export parity: enabled-mode actor(obs) == exported(flat_obs) to 1e-6.
- Disabled mode is bit-identical to today's path (existing tests keep
  passing) and an old-style checkpoint state_dict strict-loads.

## Task 2: offset discovery, task registration, wiring

1. Discover the command slice: write a small utility (script or test helper,
   implementer's choice) that builds the auxiliary env cfg's actor
   observation group and computes each term's width and offset from the
   observation manager (constructing the real env on GPU at tiny scale is
   acceptable once, in a smoke run — record the numbers in the report). The
   command term is the motion command (58 values: 29 joint pos then 29 joint
   vel); we need the offset of its first joint-position value in the
   flattened actor vector. Hardcode the discovered offset in the runner cfg
   with a comment naming the derivation, AND add a runtime guard: a check at
   task-registration or runner-cfg construction is impossible without the
   env, so instead add an assert in the training smoke path of the report —
   plus a unit test pinning the actor-group term ORDER in the cfg (so a
   reordering of observation terms fails a test rather than silently
   corrupting the slice).
2. Register `LIDAR_AUXILIARY_COADJUST_TASK_ID =
   "SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust"`
   in `src/safe_mimic/tasks/__init__.py`, mirroring the auxiliary
   registration (same env cfg builder with `expose_filtered_command=False` —
   the raw reference stays the observed base; the adjustment happens in the
   model) with runner-cfg deltas:
   - `actor.adjust_command_with_joint_prediction = True`
   - `actor.command_joint_pos_offset = <discovered>`
   - `actor.avoidance_joint_action_residual_gain = 0.0`
   - `algorithm.avoidance_teacher_mix_start = 1.0`, `..._end = 0.0`,
     `..._decay_updates = 8000`
   - `experiment_name = "safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust"`
   Add to `__all__`.
3. Tests mirroring `tests/test_exposed_lidar_auxiliary_task.py`: registration
   values above, and the base auxiliary task unchanged.
4. One GPU smoke: 8 envs, a few steps of the new task's env+runner
   construction path (e.g. via the existing benchmark script pointed at the
   new task? No — simplest is a tiny `uv run train <new-task>
   --env.scene.num-envs 8 --agent.max-iterations 2` run, then delete the tiny
   log dir it creates under the new experiment name; record output in the
   report). Confirm no shape errors and that iteration 1 completes.

## Task 3 (controller): validation gate — DO NOT LAUNCH

USER DIRECTIVE (2026-09-01): do not launch training. Bring the work to
launch-ready only: full pytest, reviews, and the documented launch command
(from scratch, 4096 envs, run-name `coadjust_scratch_4096`) recorded here and
in the ledger for the user to run. Checkpoint evaluation protocol for
whenever the user trains: compliance pair + envelope at 10k/15k (the
envelope's student path needs no flag — inference injects the actor's own
prediction).
Success criteria vs the ttc_scratch baselines: execution cosine >> 0.19 on
active frames, falls ~0, envelope collision < 64.0% (and vs 52.2%
fall-censored) with arm-link contact share falling.
