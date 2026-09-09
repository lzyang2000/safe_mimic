# Phase 1: TTC-explicit encounter training, from scratch

User decisions (2026-08-31): train FROM SCRATCH (no fine-tune of model_21000).
Controller rulings: fold in a moderate planar-filter cap raise (validated by a
teacher-envelope re-run BEFORE launching training; reverted if it does not
improve the fast tail); no extra fall penalty — training with the residual
path from step 0 plus the existing `ee_body_pos` termination prices falls in;
arm residual gain stays 0.5.

Phase 0 evidence this plan acts on (see
`.superpowers/sdd/2026-08-31-phase0-teacher-headroom-and-compliance-diagnostics/progress.md`):
teacher headroom is large at TTC >= 1.5 s (execution gap dominates); the
teacher's planar caps bind only at >= 2 m/s approaches; prediction quality is
good while execution is near-orthogonal; the training distribution (delay
1-3 s, radius 2-4 m, packed-runtime speeds) under-covers the eval envelope.

## Global constraints

- **No git commits, no `git add/restore/reset/checkout`.** Dirty working tree
  is the source of truth.
- Style: 2-space indent; `uv run ruff check <files>` and
  `uv run ruff format <files>` clean on every touched file.
- **The full test suite must stay green**: `uv run pytest -q` (baseline: 152
  passed per handoff; new tests add to that). Tests are CPU-only — follow the
  conventions in `tests/` (construct small tensors directly; no GPU, no env
  construction unless an existing test already does it).
- GPU discipline: implementers run NO GPU jobs at all in this plan (unit
  tests are CPU); the controller runs all GPU validation.
- Implementers never spawn subagents.
- Exact interface names below are binding — sibling tasks depend on them.

## Task 1: TTC-explicit encounter sampling with curriculum + adaptive bins

New module `src/safe_mimic/tasks/encounter_sampling.py` plus wiring into both
primary-human events and the training env cfg.

### The sampler module

A GPU-resident helper owning all sampling state; pure torch; no env import:

```python
class EncounterSampler:
  def __init__(self, num_envs: int, device, *,
               ttc_bin_edges_s: tuple[float, ...] = (0.5, 1.5, 2.5, 4.0),
               speed_bin_edges_mps: tuple[float, ...] = (0.75, 1.5, 3.0),
               spawn_radius_clamp_m: tuple[float, float],
               hard_ttc_below_s: float = 1.5,
               hard_speed_above_mps: float = 1.5,
               curriculum_ramp_s: float = 3000.0,
               bin_failure_ema_alpha: float = 0.05,
               bin_base_weight: float = 0.15) -> None: ...
  def note_collision_state(self, collided_now: torch.Tensor) -> None: ...
      # OR a (num_envs,) bool into a collided-since-last-schedule buffer
  def sample(self, env_ids: torch.Tensor, global_time_s: float
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
      # returns (ttc_s, speed_mps, radius_m) for env_ids
```

`sample` semantics, in order:
1. For each env in `env_ids` that has a previously assigned bin, update that
   bin's failure EMA with the env's collided-since-schedule flag
   (`ema = (1-alpha)*ema + alpha*flag`; EMA initialized to 0.5), then clear
   the flag and the assignment. Bin EMA state is global (shared across envs).
2. Compute bin weights over the (len(ttc_edges)-1) x (len(speed_edges)-1)
   grid (3 x 2 = 6 bins with the defaults):
   `w = bin_base_weight + failure_ema`, then multiply HARD bins (bin's ttc
   lower edge < hard_ttc_below_s OR speed lower edge >= hard_speed_above_mps)
   by `0.1 + 0.9 * min(1.0, global_time_s / curriculum_ramp_s)`; floor every
   weight at 0.05; normalize.
3. Sample a bin per env (categorical), then ttc ~ U(bin ttc edges) and
   speed ~ U(bin speed edges); store the sampled bin as the env's assignment.
4. `radius = clamp(ttc * speed, *spawn_radius_clamp_m)`; recompute the
   effective speed as `radius / ttc` (the bin assignment keeps the SAMPLED
   bin — note this in a comment). Return `(ttc_s, effective_speed, radius)`.

### Event wiring (both classes, identical semantics)

`PackedHumanCapsuleMotion` (`src/safe_mimic/tasks/packed_human_event.py`) and
`HumanCapsuleMotion` (`src/safe_mimic/tasks/human_capsule_event.py`):

- New event param `"encounter_sampling"`: `"independent"` (default —
  bit-for-bit current behavior) or `"ttc"`.
- When `"ttc"`: construct an `EncounterSampler` in `__init__` (clamp =
  `(min_initial_spawn_radius_m, max_initial_spawn_radius_m)`; other knobs
  overridable via optional params named exactly as the constructor kwargs).
  In `_schedule`, replace the independent draws of `requested_delay` and
  `spawn_radius` with `ttc, speed, radius = sampler.sample(env_ids, now)`,
  using `requested_delay = ttc` and `spawn_radius = radius`; every downstream
  line (delay-step rounding, playback-speed computation, placement) stays
  unchanged.
- In `__call__`, each step: if the env's termination manager has terms named
  `primary_human_collision` / `crowd_collision`, OR them into
  `sampler.note_collision_state(...)`; if neither term exists (demo cfgs),
  skip. Read via `self._env.termination_manager.get_term(name)` guarded by
  `name in self._env.termination_manager.active_terms`.
- `HumanCapsuleMotion` requires spawn-radius bounds to be set when mode is
  `"ttc"` (raise ValueError otherwise).

### Env cfg wiring

In `unitree_g1_lidar_avoidance_tracking_env_cfg`
(`src/safe_mimic/tasks/env_cfg.py`), update the primary event params (this
propagates to range-rate and auxiliary cfgs):
- `min_initial_spawn_radius_m` 2.0 -> 0.75 (max stays 4.0)
- `min_intersection_delay_s` 1.0 -> 0.5, `max_intersection_delay_s` 3.0 -> 4.0
  (these now act as the independent-mode fallback and documentation of range)
- add `"encounter_sampling": "ttc"` (defaults for all other sampler knobs).

### Tests (new `tests/test_encounter_sampling.py`)

CPU tensors, seeds fixed: (a) weights normalize to 1 and respect the 0.05
floor; (b) hard-bin multiplier is 0.1 at t=0 and 1.0 at t>=ramp; (c) EMA
update moves toward the observed flag and only fires for envs with an
assignment; (d) radius clamp keeps ttc*speed inside bounds and effective
speed = radius/ttc; (e) sampled (ttc, speed) always lie inside the assigned
bin's edges; (f) `"independent"` mode leaves both event classes' scheduling
draws untouched (test at the param level: constructing the sampler is skipped
— verify via a small unit on the param plumbing if an env-free test is
possible; otherwise cover the sampler API only and assert the env cfg sets
`encounter_sampling: "ttc"` in `tests/test_task_runner_cfg.py` style).

## Task 2: pin benchmark scripts to independent sampling

The training cfg now sets `encounter_sampling: "ttc"`; evaluation must keep
its controlled independent draws. In each of
`scripts/benchmark_policy_reaction_envelope.py`,
`scripts/benchmark_policy_avoidance.py`,
`scripts/analyze_policy_failures.py`, `scripts/benchmark_teacher_envelope.py`:
immediately where the script already overrides the primary event's
radius/delay params, add `params["encounter_sampling"] = "independent"`
(in `benchmark_policy_avoidance.py` the override lives in its scenario/cfg
configuration path — put it wherever the primary event params are already
touched, once). No other changes. ruff clean on the four files.

## Task 3: urgency-weighted escape reward

1. Expose obstacle velocity from the filter command: add a read-only property
   `obstacle_velocities_w` on `PlanarFilteredReplayMotionCommand`
   (`src/safe_mimic/tasks/kinematic_replay_command.py`) returning
   `self._obstacle_velocity_w`, and a property `obstacle_entity_slices`
   returning a dict `{entity_name: slice}` built from
   `cfg.obstacle_entity_names` and `self._obstacle_geom_counts`.
2. New reward in `src/safe_mimic/tasks/mdp.py`:

```python
def urgent_escape_progress_reward(env, command_name, human_entity,
                                  ttc_horizon_s=2.5,
                                  normalization_speed=0.5,
                                  min_closing_speed_mps=0.3,
                                  vertical_gate_m=1.0,
                                  robot_radius=0.35) -> torch.Tensor
```

   Per env: run `planar_capsule_geometry` for the primary human's capsules
   (via the command's slice into its obstacle tensors, so geometry matches
   the filter's view); take the nearest ACTIVE capsule; planar outward unit
   normal n = (robot_xy - closest_xy)/norm; closing speed
   `c = (v_obs_xy - v_robot_xy) . (-n)` using the command's
   `obstacle_velocities_w` for that capsule and the robot root velocity;
   `ttc_est = clearance / c` where `c > min_closing_speed_mps` else inf;
   `urgency = clamp(1 - ttc_est / ttc_horizon_s, 0, 1)`;
   reward `= urgency * clamp((v_robot_xy . n) / normalization_speed, -1, 1)`.
   Zero when no active capsule or clearance <= 0 handling: clamp clearance
   to >= 0 before the division and guard zeros.
3. Wire into `unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg` (the
   auxiliary cfg only) as reward `"urgent_escape_progress"` with
   `weight=1.0`, params `{"command_name": "motion", "human_entity":
   PRIMARY_HUMAN_ENTITY_NAME}`.
4. Tests in `tests/test_human_aware_mdp.py` style: synthetic tensors driving
   the math directly if the function can be factored into a pure helper —
   factor the geometry-independent core
   (`urgency_weighted_outward_progress(clearance, closing_speed, outward_speed, ...)`)
   into a pure function and unit-test: urgency 0 beyond horizon, 1 at contact
   pace, sign of reward follows outward speed, receding humans give 0.

## Task 4: filter caps, planar head scale, planar loss weight, run name

1. `src/safe_mimic/tasks/env_cfg.py`,
   `unitree_g1_lidar_avoidance_tracking_env_cfg`:
   `filtered_command_cfg.planar_filter.max_intervention_speed_mps = 2.0`
   (from default 1.5) and `.max_planar_speed_mps = 2.3` (from default 2.0),
   set right where the cfg already sets the two `safe_clearance_m = 0.8`
   lines.
2. `src/safe_mimic/tasks/__init__.py`, `_perceptive_lidar_runner_cfg`
   auxiliary branch: `cfg.actor.avoidance_planar_output_scale = 2.0` (must
   cover the new planar intervention cap — the teacher planar delta can now
   reach 2.0 m/s); `cfg.algorithm.avoidance_planar_loss_coef = 2.0` (Phase 0
   measured the planar compass as the weakest prediction);
   `experiment_name` for the auxiliary registration ->
   `"safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_ttc_scratch"`.
3. Update any tests asserting the old values (check
   `tests/test_task_runner_cfg.py`, `tests/test_reference_filter.py`,
   `tests/test_nominal_debug_cfg.py`, `tests/test_human_aware_mdp.py`); the
   suite must pass.

## Task 5 (controller-only): validation gate and launch

1. `uv run ruff check .` on touched files; `uv run pytest -q` full suite.
2. Teacher-envelope re-run with the raised caps (same seed/envelope) ->
   expect a material drop in the >= 2 m/s teacher collision bins vs
   `reference_filter_teacher_reaction_envelope_2048.json`; if not, revert
   Task 4's cap change and re-run pytest before launch.
3. Launch from scratch:
   `uv run train SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary --num-envs 4096` (+ run-name flag per `train <task> --help`).
4. Early monitoring (first ~500-1000 iters): episode length, collision/
   timeout termination shares, `Loss/avoidance_*` present and falling, no
   NaN; report status and leave the run training.

## Task ordering

Task 1 first (defines `encounter_sampling` and touches env_cfg). Task 2 may
run in parallel with Task 1 (scripts only; the param name is fixed by this
plan). Task 3 after Task 1 (env_cfg conflict). Task 4 after Task 3 (env_cfg
conflict). Task 5 last.
