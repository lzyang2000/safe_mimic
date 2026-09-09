# Safe Mimic LiDAR Avoidance Handoff

## CANONICAL SETUP (user decision 2026-09-05): the Leash task

Everything new builds on `SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash`
(`unitree_g1_lidar_unified_reference_tracking_env_cfg(active_joint_reward=True, root_lead_m=0.3, planar_filter_at_robot_root=True)`):

- Rewards: the nine nominal mjlab tracking terms + ONE `motion_active_joint_pos` (w 1.0, std 0.4, gate 0.05 rad). No bespoke avoidance rewards.
- Terminations: stock (`anchor_pos` z-only 0.25, `anchor_ori` 0.8, `ee_body_pos` z-only 0.25 on both ankles + both wrists, strict) + the two ray-only human collision terms.
- Reference: privileged planar + link CBF filters -> whole-body FK propagation -> closed-loop root target LEASHED to <= 0.3 m ahead of the robot, planar CBF evaluated at the robot root. Actor observes the RAW anchor orientation; critic the filtered one.
- Encounters: walking human spawn 0.75-4 m, TTC 0.5-4 s (ttc sampler), crowd annulus 2-4 m, obstacle-free 0.25. The slow / dense narrowings (Leash-Slow, Leash-Slow-Dense) evaluated worse than or equal to Leash and are NOT the base any more; the gated ee termination regressed arm compliance and stays default-off.
- Co-adjust actor: adjuster joint residuals injected into the observed command; teacher mix 1.0 -> 0.0 over 8000 updates.
- DEPLOYMENT NOTE (verified 2026-09-09): the exported policy is ONE network, one forward pass: deployable inputs -> shared latent -> avoidance head (2 planar values as an escape compass + 29 joint residuals added to the observed raw joint command) -> re-encode -> tracking MLP -> actions. The teacher mix exists only in training; the inference module has no teacher path. Neither the planner head nor the tracker sees privileged information: the actor group is the RAW reference command (joint pos/vel), the RAW anchor orientation relative to the robot, base angular velocity, joint pos/vel and last action; the LiDAR group is the torso-frame binned ranges + range rates + scan age. Filtered/privileged terms (filtered planar velocity, filtered joint command, human capsule vectors, privileged LiDAR ranges) are critic-only; the teacher and robustness-noise groups feed training losses only. The per-step live alignment of the reference to the robot's world XY/yaw does NOT require odometry at deployment: no actor term depends on world position, and the anchor-orientation term is relative with yaw re-aligned every step, so it reduces to the reference roll/pitch relative to the robot's IMU attitude. World-frame quantities are used only by rewards, terminations, the critic and the privileged filter. The reference frame is, by convention, the robot's current heading and footprint.
- Blind-NoHumans baseline complete 2026-09-09 (item 22). Best checkpoint on record (2026-09-06, run complete): BALLET `logs/rsl_rl/..._coadjust_unified_joint_leash_ballet/2026-09-05_17-41-54_coadjust_unified_joint_leash_ballet_scratch_4096/model_29999.pt` — dance envelope fair-regime failure 18.2 %, survival 81.8 %, envelope collision 43.8 %, 75 wrist trips, frontal fair failure 24 %; ballet reference 24.0 %. Low-collision alternative: model_15000 (fair collision 6.3 %, fair failure 18.9 %). Previous best: leash model_23000 (19.6 %). SLOW PRESET (the evaluation rule): ballet@30k collision 2.9 % / failure 5.0 % / survival 95 % on the dance clip (leash@23k 5.7 / 11.9 / 88.1; slow-TRAINED slow@15k 13.9 / 14.5; dense@30k 7.0 / 14.1); on the ballet reference 2.2 % collision, 16.0 % failure (wrist trips). The 20k/25k checkpoints look bad on the dance clip only because of wrist-height trips from ballet arm carriage; that resolved by 30k. EVALUATION PRESET (user, 2026-09-06): every checkpoint, whatever it trained on, is scored with `--encounter-preset slow` on the four eval scripts (spawn >= 1.8 m, approach 0.25-0.75 m/s uniform over TTC x speed bins, TTC 2.5-8 s, 10 s episodes; artifacts carry `_slow`); the standard frozen envelope is kept for continuity with the Phase 2-3 history. Headline metric: `summary.fair_regime.failure_rate` from `benchmark_policy_reaction_envelope.py` (approach <= 0.75 m/s, initial clearance >= 0.8 m, every non-timeout = failure): leash@23k 19.6 % (16k: 25.9 %; run-to-run noise ~2 pts), survival 80.4 %, envelope collision 51.1 %, arm compliance 1.07, fast-approach collisions 144. The run was stopped at ~23.4k while still improving; 30k never trained. Region split at 23k: frontal approaches fail 37 %, sides/back 10-15 %.
- Variants on this base: `...-Leash-Ballet` (whole G1 ballet library as the reference, clips chain in place); `...-Leash-Ballet-Slow` (same + slow encounter regime: spawn >= 1.8 m, approach <= 0.75 m/s) — training 2026-09-06 — ACTIVE RUN, see below. ballet@10k on the standard dance envelope: fair-regime failure 21.7 % (leash@10k 35.7, leash@23k 19.6), envelope collision 49.1 % (lowest on record), escape ratio 1.27 / arm compliance 1.42 (overshoots the filtered targets). Evaluate ballet checkpoints on BOTH the dance clip (comparability) and the ballet manifest (`--motion-file .../ballet.yaml`).

Updated: 2026-09-03 (Phase 3 launch). Supersedes the 2026-08-31 handoff (archived verbatim at
`docs/handoff-2026-08-31.md`). Everything below remains in the dirty working
tree on top of `addf087`; the working tree stays the source of truth. Do not
reset it.

## Executive summary

Since the previous handoff the project ran a diagnostics phase (Phase 0), a
complete from-scratch training iteration (Phase 1), a mechanism gate (Phase
2a), and an architecture change (Phase 2b). Net position:

1. **The privileged filter is not the bottleneck; execution is.** A robot
   kinematically slaved to the filtered reference collides in only 10.5% of
   the 0.5-4.0 s TTC envelope (17.9% before the planar-cap raise), while
   trained policies collide in 54-64%. The gap is 25-60 points in every TTC
   bin at or above 1 s. Legs/ankles dominate contacts (~71%); arms ~25%.
2. **Reshaping incentives did not close the gap.** The Phase 1 from-scratch
   run (TTC-explicit encounter curriculum with adaptive hard-case bins,
   urgency-weighted escape reward, raised planar caps, arm residual path)
   eliminated falls entirely but got WORSE on the collision envelope (64.0%
   at 15k iterations vs 54.4% for the previous 21k checkpoint, worse in every
   TTC bin), and time-to-first-useful-escape did not improve (p90 1.89 s vs
   1.56 s). PPO resolved the tracking-vs-evasion conflict by buying
   stability. Total failure was flat (64.0 vs 65.5 incl. falls).
3. **Adjusting the observed reference works mechanically.** Feeding the
   filter-adjusted joint targets into the actor's observed command with zero
   training (`--expose-filtered-command`) doubled command-following
   (execution cosine 0.099 -> 0.192), sped escape (outward p90 1.89 ->
   1.64 s), and cut envelope collisions 64.0 -> 52.2% — but destabilized the
   untrained policy (39% falls, which partially censors that collision
   number). Obedience must be trained in.
4. **Current architecture: co-trained limb adjuster (user-directed).** The
   auxiliary head's 29-joint residual prediction is injected into the joint-
   position targets of the actor's OBSERVED command; training anneals from
   teacher-forced filter corrections (mix 1.0) to the policy's own
   predictions (0.0 over 8000 updates) while the supervised loss anchors the
   head throughout. Limbs only — the pelvis/root path is deliberately
   unchanged (user decision: no velocity command, no displacement channel).
   Implementation is review-clean.
5. **Phase 2c (current run): FK-consistent task-space objective.** Root cause
   found for the compliance ceiling: the task-space body targets (and the
   z-only `ee_body_pos` termination, 0.25 m wrist-height threshold) tracked
   the UNCORRECTED arm pose, so full obedience to an arm-drop correction
   (0.4-0.6 m wrist drop) was punished up to termination — capping execution
   cosine at 0.10-0.19 in every lineage. Fix: the command's `body_pos_w` /
   `body_quat_w` now propagate the filtered ARM corrections via
   reference-frame chain FK (validated against `mujoco.mj_kinematics` to
   1e-7), making all task-space rewards, the termination, and metrics
   consistent from one source. Task:
   `SafeMimic-...-Lidar-Auxiliary-CoAdjust-FKC`.
   The first co-adjust run (`coadjust_scratch_4096`, stopped at 13.8k under
   the old inconsistent objective) already led every lineage honestly at 10k:
   envelope 59.5% with zero falls, packed 52.3%, best clearance response —
   but execution cosine stayed 0.114, which the FK fix targets.
   `coadjust_fkc_scratch_4096` trained 2026-09-02/03 and was KILLED at 11.3k
   by user decision after the absolute collision rate did not move (10k gate:
   envelope 53.4%, packed 47.2%, arm compliance 51% vs 9%). See the
   post-mortem section below. Checkpoints to `model_11000` are kept.
6. **Phase 2d/2e (STOPPED 2026-09-03 at iteration 11364): arm-compliance
   completion + randomized starts.** Diagnosis: FKC oracle-injection
   compliance (0.58) equals self-injection (0.59) — the remaining arm gap is
   the mimic's tracking of the adjusted command, not prediction. Fixes:
   arm body LINEAR-velocity targets now follow the corrected trajectory
   (finite-difference of the FK displacement, hold on reset/wrap);
   `active_correction_joint_tracking` reward (exp error over actively
   corrected joints, weight 1.5, std 0.2); training start frame randomized
   (`sampling_mode="uniform"`; demos/evals pinned to "start"). Task:
   `SafeMimic-...-Lidar-Auxiliary-CoAdjust-FKC2`, run
   `coadjust_fkc2_scratch_4096`. RESULT at 10k (frame-0 eval protocol, vs
   FKC@10k): envelope 69.6% (53.4), ee_body_pos terminations 157 (40), arm
   state compliance t 0.61/50% (0.57/51%), pelvis escape_ratio 0.01 (0.47).
   Verdict: more shaping rewards are not the lever. Checkpoints to
   model_11000 kept.
7. **Phase 3 (CURRENT RUN, launched 2026-09-03): unified filtered reference,
   nominal rewards only.** User-directed pipeline: LiDAR of randomized humans
   -> privileged live planar + link CBF filters -> FK of EVERY joint
   correction (legs, waist, arms; torso anchor included) and CLOSED-LOOP
   world-frame integration of the filtered root -> the stock mjlab tracking
   reward set (nine terms, stock weights) on that reference. Every bespoke
   reward is gone (planar velocity/progress/freeze, filtered-joint,
   urgent-escape, active-correction, survival, all proximity penalties); the
   two human collision terminations stay (ray-only humans); the nominal
   motion terminations consume the filtered reference by construction. The
   actor's `motion_anchor_ori_b` observes the RAW anchor
   (`mdp.raw_motion_anchor_ori_b`, deployment-consistent); critic/rewards use
   the filtered one. Co-adjust architecture and teacher schedule unchanged.
   Cfg flags: `propagate_joint_corrections_to_body_targets`,
   `closed_loop_root_target` (both default False everywhere else). Task
   `SafeMimic-...-Lidar-Auxiliary-CoAdjust-Unified`, run
   `coadjust_unified_scratch_4096` (1.8 s/iter). Plan + ledger:
   `docs/superpowers/plans/2026-09-03-phase3-unified-filtered-reference.md`,
   `.superpowers/sdd/2026-09-03-phase3-unified-filtered-reference/progress.md`.
   10K GATE RESULT (vs FKC@10k / FKC2@10k): envelope 64.9% (53.4 / 69.6);
   ee_body_pos terminations 222 (40 / 157); packed 69.9% (47.2 / 67.9); arm
   state compliance t 0.32 (0.59 / 0.61), falling since handover; pelvis
   escape_ratio 0.42 (0.48 / 0.01), escape cosine 0.59 (best). Reading: the
   nominal set moves the pelvis but has no joint-space term, so arm
   corrections carry almost no gradient (body_pos averages over 14 bodies),
   and the now-consistent ee_body_pos termination fires on NON-compliance.
   Run STOPPED at 10281 (checkpoints to model_10000).
8. **Phase 3b (CURRENT RUN, launched 2026-09-03): unified reference + one
   joint term on the actively filtered joints (user-directed).** Same task as
   Phase 3 plus `"motion_active_joint_pos"`
   (`mdp.active_correction_joint_tracking_exp`, weight 1.0, std 0.4 rad per
   active joint, threshold 0.05 rad) against the filtered joint targets. Std
   sized from data: per-active-joint rms error ~0.33 rad at the unified 10k
   gate -> ~0.5 reward today, 1.0 at compliance (FKC2's std 0.2 was inert).
   Task `SafeMimic-...-Lidar-Auxiliary-CoAdjust-Unified-Joint`, run
   `coadjust_unified_joint_scratch_4096`; eval via `--task-id` /
   `--task-variant coadjust-unified-joint`. 10K GATE RESULT: arm state
   compliance t 0.77 / 61% (best of any lineage; FKC 0.59), pelvis
   escape_ratio 0.56 (best; FKC 0.48), envelope 58.3% (FKC 53.4, unified
   64.9), ee_body_pos terminations 189, packed 63.0% (FKC 47.2). First
   lineage with compliance AND escape both best; absolute avoidance still
   trails FKC (frame-0-trained, matching this eval). Fast-approach collisions
   ~173/512 in EVERY lineage = the pelvis capability ceiling. Run continuing
   to 15k/20k; next diagnostic = wrist-vs-ankle split of ee_body_pos.
   Watch: `Episode_Termination/anchor_pos` (closed-loop root drift is bounded
   only while the reference planar speed < 0.4 m/s), `ee_body_pos` (targets
   now consistent for legs), aux planar loss (now has an irreducible
   recovery component; compare vs FKC2, not absolute).

## Launch and evaluation (NO ACTIVE RUN. Blind baseline finished at 30k (gated 10k-30k); ballet-lag finished, not adopted; canonical best remains ballet model_29999; ballet finished at 30k (model_29999, gated 10k-30k); dense FINISHED at 30k (model_29999; gates 5k/10k/20k recorded, 30k queued); leash-slow stopped at ~16.3k; leash run stopped at ~23.2k (model_23000 kept); joint run died at ~14790 (model_14000))

Launch from scratch (the current Phase 3 run; earlier lineages substitute
their task id and run name):

```bash
tmux new-session -d -s leash_ballet -c ~/twist2/safe_mimic \
  "uv run train SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint-Leash-Ballet \
   --env.scene.num-envs 4096 --agent.run-name coadjust_unified_joint_leash_ballet_scratch_4096 \
   2>&1 | tee logs/tmux/coadjust_unified_joint_leash_ballet_scratch_4096.log"
# attach: tmux attach -t leash_ballet
# eval/play for the ballet task: pass --motion-file artifacts/bones-seed/datasets/g1_ballet_v1/ballet.yaml
```

Play with the ADJUSTER's command as the ghost (blue; `--ghost both` adds the
teacher's green ghost; the stock `uv run play` ghost is the privileged teacher
reference, which does not exist at deployment):

```bash
uv run python scripts/play_adjuster_ghost.py --checkpoint <ckpt> \
  --task-id SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust-Unified-Joint \
  --ghost adjuster   # or: both | teacher
```

Eval scripts need the task selected for non-auxiliary checkpoints:
`--task-id SafeMimic-...-CoAdjust-Unified` (compliance / envelope / failure
analysis) or `--task-variant coadjust-unified` (benchmark_policy_avoidance);
the unified eval cfg is built by `unitree_g1_lidar_unified_reference_tracking_env_cfg`
so reward set and both flags match training. `benchmark_teacher_envelope.py`
still builds the auxiliary (open-loop, arm-only) cfg; its numbers are not
directly comparable to the unified run.

Checkpoint gates (compare against the baselines below; the co-adjust student
path needs no flags — inference injects the actor's own prediction):

```bash
uv run python scripts/diagnose_joint_compliance.py \
  --checkpoint <ckpt> --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 512 --total-steps 600 --output <out.json>

uv run python scripts/benchmark_policy_reaction_envelope.py \
  --checkpoint <ckpt> --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 1024 --output <out.json>
```

Success criteria: STATE compliance (compliance signal 5, `state_compliance`:
position t along the raw->corrected arm-pose segment on active frames, and
the fraction of frames with t >= 0.5) rising — FKC@10k: t 0.57 / 51%, vs
coadjust@10k 0.18 / 9%; falls (`ee_body_pos` in `termination_causes`) low;
envelope collision below 53.4% (FKC@10k) honestly, trending toward the 10.5%
kinematic ceiling. NOTE: the older execution-cosine signal (signal 3) ranks
lineages backwards for compliant policies (it measures motion TOWARD the
residual, which vanishes once the arm already sits at the corrected pose) and
is no longer the primary compliance metric.

## Baselines and key artifacts (all in artifacts/benchmarks/)

| Measurement | Number | Artifact |
|---|---|---|
| Teacher headroom, original caps | 17.9% collisions | reference_filter_teacher_reaction_envelope_2048.json |
| Teacher headroom, raised caps (2.0/2.3 m/s) | 10.5% | reference_filter_teacher_reaction_envelope_raised_caps_2048.json |
| Old model_21000 envelope (2048, with termination causes) | 54.4% + 10.6% falls | lidar_avoidance_auxiliary_joint_robust_model_21000_reaction_envelope_termbreakdown_2048.json |
| Old model_21000 correction ablation (packed/online) | oracle -9.2 / -18.4 pts, ~27% fall tax | ..._model_21000_correction_ablation{,_online}_1024.json |
| Old model_21000 compliance | pred cos 0.82 / exec cos 0.15 | ..._model_21000_joint_compliance_512.json |
| ttc_scratch model_15000 envelope | 64.0%, 0 falls | lidar_avoidance_ttc_scratch_model_15000_reaction_envelope_1024.json |
| ttc_scratch model_15000, exposure gate | 52.2% + 39% falls, exec cos 0.192 | ..._model_15000_reaction_envelope_exposed_1024.json, ..._joint_compliance{,_exposed}_512.json |
| ttc_scratch packed primary-only trend | 61.1% @10k -> 55.1% @15k | lidar_avoidance_ttc_scratch_model_{10000,15000}_primary_only_1024.json |
| coadjust model_10000 envelope (pre-FKC objective) | 59.5%, 0 falls | lidar_avoidance_coadjust_model_10000_reaction_envelope_1024.json |
| coadjust model_10000 packed / compliance | 52.3% / exec cos 0.114, pred cos 0.883, clearance 0.701 | lidar_avoidance_coadjust_model_10000_{primary_only_1024,joint_compliance_512}.json |

Checkpoints: old lineage
`logs/rsl_rl/safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_joint_robust/2026-08-31_00-16-59_auxiliary_joint_robust_4096/`
(to 21000); Phase 1
`logs/rsl_rl/safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_ttc_scratch/2026-09-01_01-32-33_ttc_scratch_4096/`
(stopped at 17000, deliberately). A few-minute exposed-command fine-tune stub
exists under `..._ttc_exposed_ft/` (superseded; user directed from-scratch).

## What was built since 2026-08-31

### Phase 0 diagnostics (all review-clean)

- `scripts/benchmark_teacher_envelope.py` — kinematic teacher-headroom
  benchmark (robot slaved to the filtered reference over the reaction
  envelope; also reports planar-cap saturation).
- `scripts/diagnose_joint_compliance.py` — the four compliance signals per
  arm side (teacher-vs-predicted residuals, mechanical action residual,
  achieved joint motion over 0.1/0.5 s, arm-clearance response), plus a
  planar-compass cosine.
- `scripts/benchmark_policy_reaction_envelope.py` — per-term termination-
  cause breakdown added (all previous "other failures" are `ee_body_pos`,
  i.e., tracking-divergence/falls).

### Phase 1 training changes (in the tree, used by the ttc_scratch run)

- `src/safe_mimic/tasks/encounter_sampling.py` + wiring in both human events:
  TTC-explicit encounter sampling — joint (TTC, speed) draws over a 3x2 bin
  grid, hard bins (TTC < 1.5 s or speed >= 1.5 m/s) curriculum-ramped
  0.1 -> 1.0 over 3000 sim-seconds, per-bin collision-failure EMAs
  reweighting sampling, attribution edge-triggered at schedule time on
  realized (TTC, speed). Spawn 0.75-4.0 m, TTC 0.5-4.0 s. Event param
  `encounter_sampling: "ttc" | "independent"`; all four benchmark scripts pin
  `"independent"` plus the historical bounds (2-4 m / 1-3 s) so evaluation
  distributions stay comparable to every existing artifact.
- `mdp.urgent_escape_progress_reward` — privileged TTC-gated outward-progress
  bonus (auxiliary cfg only, weight 1.0); command exposes
  `obstacle_velocities_w` / `obstacle_entity_slices`.
- Planar filter caps raised in the avoidance family:
  `max_intervention_speed_mps` 1.5 -> 2.0, `max_planar_speed_mps` 2.0 -> 2.3
  (validated pre-launch: teacher headroom 17.9 -> 10.5%); actor planar head
  scale 2.0; planar aux loss coef 2.0.
- Online-composer fix: deactivated humans can no longer be revived
  mid-episode by path expiry (`scheduled` flag in the sampler).

### Phase 2a/2b: reference adjustment (current mainline)

- `--expose-filtered-command` flag on the envelope + compliance scripts (the
  zero-training gate).
- `SafeMimic-...-Lidar-Auxiliary-Exposed` task (env-side exposure; kept, but
  superseded by co-adjust).
- **Co-adjust path** in `src/safe_mimic/rl/perceptive_lidar.py`: cfg fields
  `adjust_command_with_joint_prediction` / `command_joint_pos_offset`. In
  adjust mode the actor runs an adjuster pass on the raw observations,
  injects the correction's joint part into the command slice (position
  targets only, on the unnormalized obs; functional `torch.cat`, stored
  rollouts untouched), keeps a 2-D planar-compass concat, disables the joint
  concat and action-residual routes, and the mimic pass tracks the adjusted
  reference. Training rollouts inject the teacher-mixed conditioning;
  inference and ONNX export inject the actor's own prediction (parity ~3e-8).
  Normalizer statistics update on raw observations and apply to the adjusted
  vector (deliberate: mix-independent, deployment-consistent).
- `SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-CoAdjust`
  task: adjust on, offset 0 (command term is first in the actor group —
  discovered live, pinned by a term-order test), residual gain 0.0, teacher
  mix 1.0 -> 0.0 over 8000 updates, experiment name
  `safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_coadjust`.

## Standing user directives

- Train FROM SCRATCH; no fine-tuning.
- Limbs only: NO pelvis/root command channel of any kind (velocity and
  displacement variants both declined).
- Do NOT launch training without the user's explicit instruction (the
  co-adjust launch was explicitly authorized 2026-09-02: "we are free now
  launch as you wish").
- No git commits; the dirty working tree is the source of truth.

## Training-data caveat discovered 2026-09-03: frame-0 starts

`PlanarFilteredReplayMotionCommand._resample_command` hard-coded `time_steps = 0`
(inherited from the exact-replay demo class), so with 10 s episodes EVERY
training run to date has only ever replayed the first 10 s of the 131 s dance
(TensorBoard `Metrics/motion/sampling_entropy` was 0 in all runs for this
reason). All benchmark artifacts were likewise measured from frame 0. Phase 2e
makes training honor `sampling_mode` ("uniform" for the avoidance family;
demos keep "start") and pins the evaluation scripts to frame 0 for
comparability (opt-in `--random-start`). The left/right arm-compliance
asymmetry probe averaged the full motion and should be re-checked on the
trained window. Plan: `docs/superpowers/plans/2026-09-03-phase2e-start-frame-randomization.md`.

## Post-mortem (2026-09-03): why the absolute number never moved

Read `docs/postmortem-2026-09-03-avoidance-plateau.md`. Summary: the FKC run
was killed at 11.3k after the user judged the absolute collision rate
unchanged. All four lineages realize only 0.33-0.51 of the teacher's
commanded planar escape velocity and keep the root closer to the nominal
dance than to the escape target; first-useful-escape timing is identical
across lineages; collisions are 64-71% lower-body; limb compliance
substituted arm contacts for leg contacts (no net change); and with oracle
teacher information the policy still collides 52% where the teacher collides
~10%. The binding constraint is pelvis escape execution. Two candidate
causes (no closed-loop root error signal vs. unrehearsed escape skill) are
separable by one experiment: a self-generated 2-D anchor-displacement
observation, teacher-forced, trained from scratch (previously declined by
the user; re-raised with this evidence). Current best checkpoints for any
follow-up: FKC `model_10000`/`model_11000`.

## The open problem: pelvis/root escape

The co-adjust run addresses limb compliance (~25% of contacts). The dominant
failure — legs/root, needing whole-body escape — currently has no command
channel and relies on LiDAR-inferred, reward-shaped behavior, which two runs
have shown insufficient. Options documented and deliberately deferred:
(a) a 2-D body-frame displacement command (`motion_anchor_pos_b`-style,
self-generated by the adjuster so no odometry needed) — declined for now;
(b) deploying the CBF filter at runtime on perceived obstacles;
(c) synthesizing escape stepping into the reference itself. The teacher-
headroom artifacts bound what any of these can achieve.

## Decision record and known minor issues

Full ledgers (rulings, review verdicts, deferred minors):

```text
.superpowers/sdd/2026-08-31-phase0-teacher-headroom-and-compliance-diagnostics/progress.md
.superpowers/sdd/2026-08-31-phase1-ttc-scratch-training/progress.md
.superpowers/sdd/2026-09-01-phase2b-cotrained-limb-adjuster/progress.md
```

Plans: `docs/superpowers/plans/2026-08-31-phase0-*.md`, `...phase1-*.md`,
`2026-09-01-phase2b-cotrained-limb-adjuster.md`.

Deferred minors (all triaged non-blocking by final reviews): bin-EMA state is
not checkpointed (restarts at 0.5 on resume); the far-fast encounter bin
realizes only ~2% occupancy; ratio tests use a constant teacher mix (schedule
skew structurally guarded, unpinned); a comment misattributes the command
property's class; the intra-command pos/vel layout is pinned only by code.

## Benchmark correctness notes (carried forward, still true)

- Checkpoint loading restores `common_step_counter`; the three policy
  benchmark scripts reset once after loading. Do not trust initial-TTC data
  from artifacts generated before this fix.
- Same-seed GPU reruns vary by ~±2 points; per-bin binomial SE at n≈130-320
  is ±3-4 points.
- The kinematic teacher benchmark is an optimistic bound (no dynamics); its
  clearance sampling is one control step stricter than the policy
  benchmark's termination — both asymmetries are conservative for the
  "execution gap dominates" conclusion.

## Last validation

- `uv run ruff check src/ scripts/ tests/`: clean.
- Full test suite: **191 passed** (152 at the previous handoff).
- All work reviewed: per-task reviews, scoped fix-round re-reviews, and final
  whole-branch reviews for Phases 0, 1, and 2b (verdicts in the ledgers).

9. **Root-tracking diagnosis (2026-09-03).** `scripts/diagnose_root_tracking.py` on joint@14000 -> `artifacts/benchmarks/lidar_avoidance_coadjust_unified_joint_model_14000_root_tracking_512.json`. Root-pos reward is dead (value 0, gradient 0) once the ghost leads by >0.6 m (19 % of frames, 61 % of collisions); the filtered reference moves at 1.0-1.1 m/s but the robot plateaus at ~0.6 m/s; in 77 % of robot-in-danger frames the ghost already has >=0.8 m clearance so the CBF (evaluated at the ghost) stops pushing; following costs body_pos (ankle error .05->.26 m). Actor cannot observe the root target, so behaviour is invariant to the ghost construction. Proposed: leash ghost to <=0.3 m of robot + evaluate planar CBF at robot root (see progress ledger). Not built, no run launched.

10. **Phase 3c: Unified-Joint-Leash (launched 2026-09-03).** Same task as Unified-Joint plus two reference-generator flags (`max_root_lead_m=0.3`, `planar_filter_at_robot_root=True`): the closed-loop root target may lead the robot by at most 0.3 m and the planar CBF is evaluated at the robot. Motivation in item 9. Task id `...-CoAdjust-Unified-Joint-Leash`; eval `--task-id` accordingly, avoidance `--task-variant coadjust-unified-joint-leash`. Root-tracking diagnostic: `scripts/diagnose_root_tracking.py --task-id <leash id>`.

11. **Play panel: human approach-speed cap (2026-09-04).** The viser panel gained a "Safe Mimic" folder with the checkbox "Human approach <= robot speed". It stretches each scheduled encounter so the primary human never approaches faster than the robot's recent peak planar speed (decayed running peak, half-life 1 s, floored at 0.3 m/s); the spawn ring, crossing angle and intersection point are unchanged. Default OFF, so training and recorded benchmarks are unaffected; only the online `HumanCapsuleMotion` is involved, the packed training event is untouched. The speed is a scheduling quantity (`spawn_radius / intersection_delay`), not a property of the source clips, which are time-warped to whatever the schedule asks. Use it from `scripts/play_adjuster_ghost.py --viewer viser`.

12. **Phase 3d: Unified-Joint-Leash-Slow (launched 2026-09-04).** Leash plus (a) the walking human spawns outside the critical distance in training and eval (min spawn radius 1.8 m: 99.7 % of spawns start with >= 0.8 m surface clearance), (b) slow-regime encounter sampler (approach <= 0.75 m/s, TTC 2.5-8 s, no hard bins; crowd unchanged, already >= 2 m), (c) `ee_body_pos` replaced by `mdp.bad_motion_body_pos_z_only_filter_gated`: 0.25 m strict, 0.5 m on a limb while the link filter is correcting it (joint@16k: 124/126 trips were wrists lagging the FK-propagated arm target). Motivation: slow-bin anatomy (item 9/ledger): 57/65 slow collisions started inside reach, 15 % of slow episodes ended on the wrist z check. Task id `...-CoAdjust-Unified-Joint-Leash-Slow`; `--task-variant coadjust-unified-joint-leash-slow`. Headline metric going forward: envelope slow bin (<= 0.75 m/s) conditioned on initial clearance >= 0.8 m (leash@16k: 13.3 %).

13. **Leash-Slow gate and Phase 3e (2026-09-05).** Leash-Slow@15k on the frozen envelope: 68.0 % collisions (leash 53.9), fair regime 27.3 % failures = identical survival to leash@16k once wrist terminations count as failures (leash: 21 collisions + 18 ee of 143; slow: 39 collisions). Arm compliance regressed 0.856 -> 0.618 with the adjuster head unchanged (prediction cos 0.85 both): the gated wrist termination removed the main training pressure for arm compliance. Training saw only 0.7 collisions/batch (leash 9.2). New task `...-Leash-Slow-Dense`: strict stock ee termination again, TTC cap 5 s (2-3 encounters per episode), obstacle-free probability untouched. HEADLINE METRIC now `summary.fair_regime.failure_rate` from the envelope script (approach <= 0.75 m/s, initial clearance >= 0.8 m, any non-timeout = failure); baseline 27.3 %.

14. **Ballet library reference (2026-09-05, built, NOT yet trained).** `...-Unified-Joint-Leash-Ballet` = LEASH (canonical) with the whole G1 ballet library (`artifacts/bones-seed/datasets/g1_ballet_v1/ballet.yaml`, 348 clips, every split) as the reference. The filtered replay command now accepts a manifest as `motion_file` (`LibraryMotionLoader`, per-env clip bounds); when a clip ends the env chains to a freshly drawn clip's first frame and the live alignment keeps the robot where it is (smoke: reference root within 0.044 m of the robot at every clip change). Eval/play scripts: pass the manifest as `--motion-file`, e.g. `--motion-file artifacts/bones-seed/datasets/g1_ballet_v1/ballet.yaml`; avoidance variant `coadjust-unified-joint-leash-ballet`. The "restart where it died" idea was dropped: both human spawners are robot-centred, so it would not change human distances.

15. **Failure regions (2026-09-05).** Envelope artifacts now carry per-episode approach bearings (spawn and closest approach, robot heading frame) and `summary.fair_regime_by_initial_bearing` / `fair_regime_by_closest_bearing` (front/left/back/right). Finding across every lineage: frontal approaches fail 44-56 % in the fair regime versus 9-28 % from the sides or behind; the back is the safest region, so it is not a LiDAR coverage issue. Wrist-height terminations cluster in front (dance arms held forward). Dense@20k fair-regime failure 30.1 % vs leash@16k 25.9 % (re-run, same conditions): dense is not better than leash; ballet run pending the user's swap decision.

16. **LiDAR-on vs blinded video (2026-09-06).** `scripts/render_avoidance_comparison.py` picks the median frontal encounter from a seeded batch, records it with the policy, then rebuilds the identical scene and records it with the LiDAR fed "no returns" (the benchmarks' blind mode). Output for ballet@30k, dance clip, slow preset, no crowd: `artifacts/videos/no_crowd/avoidance_seed7_env17_{lidar_on,lidar_blind,side_by_side}.mp4` (LiDAR on survives at 0.73 m minimum clearance; blinded collides at 2.0 s). With the crowd present, median frontal encounters on the dance clip often end on the wrist-height check because crowd members inside the 0.8 m link-filter band move the reference arms; use `--no-crowd` to match the crowd-free benchmark scene.

17. **Wrist check (2026-09-06).** Evaluation now splits `safety_failure_rate` (collisions + falls) from `tracking_termination_rate` (ee_body_pos wrist/ankle height trips) in `fair_regime_summary`; `failure_rate` still counts both. Under the slow preset ballet@30k is 2.9 % safety / 2.1 % tracking on the dance clip and 2.2 % / 13.8 % on the ballet reference. A lag-aware termination (`mdp.bad_motion_body_pos_z_only_lag_aware`, bound 0.25 m + 0.2 s x reference vertical speed, cap 0.6 m) is built as task `...-Leash-Ballet-Lag`, not yet trained. Ballet-Slow@10k lost to Ballet@10k on every preset including the slow one (safety 6.5 vs 5.3 %).

18. **Ballet-Lag verdict (2026-09-07).** The lag-aware ee check (bound widening with reference vertical speed) trained to 30k. In distribution it gained 2-4 pts survival at equal safety (ballet ref, slow preset: 88.1 % at 25k vs ballet's 84.0); on the dance clip it was far worse (tracking terminations 16-50 % vs 2.1 %, safety 4.1 vs 2.9 %) because the allowance let the arms ride further from the filtered target (left arm compliance 1.6-1.8). Not adopted; flag and task kept default-off. Canonical best: ballet model_29999.

19. **Blind baseline (2026-09-07, training).** `...-Leash-Ballet-Blind` = Leash-Ballet with the actor's LiDAR term (`BlindDirectionalHeldLidarRangeRate`) reading "no returns" every step; same network, rewards, reference, terminations; critic still privileged. The honest no-perception lower bound to quote against ballet@30k (slow preset safety 2.9 % dance / 2.2 % ballet ref) and against the eval-blinded videos. Variant `coadjust-unified-joint-leash-ballet-blind`.

20. **Blind baseline result (2026-09-08).** Blind-trained@30k under the slow preset: 61.9 % safety failures / 2.5 % survival on the dance clip (53.8 % / 19.0 % on the ballet reference), escape ratio ~0 (it stops reacting once the teacher mix hits zero). Sighted ballet@30k: 2.9 % / 95 %. That x21 gap is the number for the LiDAR's value; the eval-blinded sighted policy (41.5 % safety, 0 % survival, 58 % wrist trips) is an OOD artefact and belongs only in video captions. Three-way video (sighted / eval-blinded / blind-trained) via `render_avoidance_comparison.py --baseline-checkpoint ... --baseline-task-id ...-Ballet-Blind` under `artifacts/videos/three_way/`.

21. **Viser recorder (2026-09-08).** `scripts/record_viser_comparison.py` records the comparison through the play viewer's viser scene (skinned SOMA humans, LiDAR hit points, teacher ghost) using a headless Chromium via playwright (`uv pip install playwright && uv run playwright install chromium`); use `--gpu` (software GL is ~3 s/frame, GPU ~0.1 s). Same scenario logic and flags as the offscreen renderer (`--force-crowd`, `--lidar-points`, `--drop-tracking-termination`, `--baseline-checkpoint`, continuous mode with tally). Shared helpers in `src/safe_mimic/video_tools.py`. Final video: `artifacts/videos/viser_ballet_sighted_vs_blind_dense_high/` (use `--dense-crowd --elevation -52 --distance 4.0 --fov-deg 58 --client-settle-s 0.1 --gpu`).
22. **Blind-NOMINAL and Blind-NOHUMANS baselines (2026-09-08).** The Blind task (item 19) still trained on the CBF-filtered reference, so it kept an avoidance signal. `nominal_reference=True` (`PlanarFilteredReplayMotionCommandCfg.disable_filters`) turns both CBFs off: raw live-aligned reference, zero teacher residual, zero arm target offsets; only the clearance metric is still logged. `...-Leash-Ballet-Blind-Nominal` (blind actor + filters off) was launched, then stopped at ~6k when the user asked for a baseline with "no humans whatsoever": `...-Leash-Ballet-Blind-NoHumans` adds `training_humans=False`, which removes the human animation events and the collision terminations from the TRAINING cfg only (entities stay parked at -100 m; the play cfg keeps the populated scene, so every eval script scores it in the same scene as the others). Variants `coadjust-unified-joint-leash-ballet-blind-nominal` / `-nohumans`. ACTIVE RUN: `coadjust_unified_joint_leash_ballet_blind_nohumans_scratch_4096` (tmux `blind_nohumans`), gates in tmux `blind_nohumans_gates`, video waiter in tmux `viser_nohumans_wait` (`logs/tmux/viser_record_nohumans.sh`, titles 5x via `--title-scale`). RESULT (2026-09-09, 30k, slow preset, safety/tracking/survival %): dance 91.8 / 0.0 / 8.2, ballet ref 74.0 / 0.3 / 25.7 (standard envelope: dance 100 % collisions); flat from 10k to 30k; no wrist trips because nothing perturbs the arm reference. Headline: LiDAR takes safety failures 91.8 % -> 2.9 % (x32), survival 8.2 % -> 95 %. This is THE blind baseline for the Safe Mimic vs Blind video: FINAL `artifacts/videos/viser_safe_mimic_vs_blind_nohumans_env9/viser_seed11_env9_side_by_side_web.mp4` (seed 11 env 9 from the sweep `sweep_seed11.json`; Safe Mimic survives 25 s at 0.64 m min clearance, the blind robot is hit 6 times, first at 4.2 s). Recorder rule: ONE panel per process (`--reuse-lidar-on` for the second); two panels in one process hang at the first teardown and the headline "value of the LiDAR" number; the filtered-reference Blind (61.9 %) is the "same environment, no perception" ablation. No active run; GPU free.
