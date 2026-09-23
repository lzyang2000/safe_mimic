# Safe Mimic

An installable mjlab plug-in for Unitree G1 motion imitation that preserves a
reference dance while avoiding moving people.

## Install

The project consumes the published package; it does not clone or vendor mjlab.

```bash
uv sync
uv run list-envs | grep SafeMimic
```

mjlab training requires an NVIDIA GPU. The dependency is pinned to `mjlab==1.6.0`
and the complete resolution is recorded in `uv.lock`.

## Environment setup

The plug-in registers fourteen environments:

| Task | Purpose | Crowd | Full action human | Policy input |
|---|---|---|---|---|
| `SafeMimic-Tracking-ExampleDance-NoStateEst-Unitree-G1` | Single-motion baseline trained only on the bundled mjlab example dance path | none | none | state-estimation-free 154-value actor input |
| `SafeMimic-Tracking-ExampleDance-ImplicitState-Unitree-G1` | Single-motion implicit-estimator experiment on the same example dance | none | none | 154 current values plus six predicted state values and a 16D history latent |
| `SafeMimic-Tracking-MotionLib-Unitree-G1` | First-cut generalized raw mimic teacher over the packed G1 library | none | none | state-estimation-free 154-value actor input |
| `SafeMimic-Tracking-MotionLib-ImplicitState-Unitree-G1` | Learned-state ablation without direct translational state | none | none | 154 current values plus six predicted state values and a 16D history latent |
| `SafeMimic-Tracking-MotionLib-StateEst-Unitree-G1` | Generalized raw mimic teacher with translational state estimation | none | none | upstream 160-value actor input |
| `SafeMimic-Tracking-Obstacles-Unitree-G1-Lidar` | Crowd-only obstacle-aware training | 0--packed-capacity people, five collidable randomized capsules each | none | tracking state + 1,080 LiDAR ranges |
| `SafeMimic-Tracking-Crowd-Human-Unitree-G1-Lidar` | Recommended optimized training task | 0--packed-capacity people, five ray-only randomized capsules each | one collidable 18-capsule human | tracking state + 1,080 LiDAR ranges |
| `SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar` | From-scratch no-state avoidance policy with privileged safe-reference teaching | randomized 2--4 m ring, including empty scenes | one collidable 18-capsule human | 154 tracking + 2,160 dual-scan directional ranges + scan age |
| `SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-RangeRate` | Revised avoidance task with explicit closing speed and dense link-clearance shaping | randomized 2--4 m ring, including empty scenes | one moving 18-capsule human | 154 tracking + 1,080 directional ranges + 1,080 range rates + scan age |
| `SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Sparse` | Fast low-ray-count ablation with the same directional encoding | randomized 2--4 m ring, including empty scenes | one collidable 18-capsule human | 154 tracking + 144 dual-scan directional ranges + scan age |
| `SafeMimic-Tracking-Flat-Unitree-G1-Lidar-Debug` | Visualize the complete scene with an upstream nominal checkpoint | five-capsule randomized crowd | one 18-capsule human | unchanged upstream 160-value actor input |
| `SafeMimic-Reference-Replay-Crowd-Human-Unitree-G1-Lidar-Demo` | Exact kinematic reference replay for developing the reference filter | five-capsule randomized crowd | one 18-capsule human | unchanged upstream input; policy actions are ignored |
| `SafeMimic-Reference-Filter-Crowd-Human-Unitree-G1-Lidar-Demo` | Tune deterministic planar and link/joint CBF reference filters | five-capsule randomized crowd | one 18-capsule human | unchanged upstream input; policy actions are ignored |
| `SafeMimic-Reference-Filter-Policy-Crowd-Human-Unitree-G1-Lidar-Demo` | Run the no-state policy on a live-aligned, filtered reference | five-capsule randomized crowd | one 18-capsule human | checkpoint-compatible 154-value no-state actor input |

List the installed registrations with:

```bash
uv run list-envs | grep SafeMimic
```

All Safe Mimic tasks use TensorBoard logging by default, disable model uploads,
and save a checkpoint every 1,000 training iterations. Runs remain under
`logs/rsl_rl/<experiment_name>/<timestamp>/`. View all runs with:

```bash
uv run tensorboard --logdir logs/rsl_rl
```

### From-scratch LiDAR avoidance policy

The revised live-aligned task trains from random initialization on the example
dance by default. Launch the 4,096-environment run with:

```bash
uv run train SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-RangeRate \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz
```

The task without explicit range rate remains registered as the paired baseline:

```bash
uv run train SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz
```

For a lower-fidelity ray-casting ablation, use the separate 120 x 4 task. It
casts 96 rays per policy step while retaining the same directional actor input:

```bash
uv run train SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Sparse \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz
```

The actor never receives measured base linear velocity, global reference-root
position, filtered CBF commands, capsule geometry, or clean LiDAR. It receives
the nominal 154-value live-aligned mimic input and a separate perceptive branch:
the current completed scan and explicit closing speed, pooled to closest-return
angular cells, plus scan age. Positive range rate means that the closest return
in a cell is approaching. Both channels are computed from current and previous
10 Hz scans. The full task uses `120` azimuth by `9` elevation cells; it still
ray casts every point in each `185 x 27` scan before applying point-level noise
and minimum pooling. The baseline supplies current and previous range instead,
and the sparse `120 x 4` ablation retains `24 x 3` cells per scan. The 2,160
directional values are projected to a 128D latent before fusion with the
tracking MLP.

Pooling happens once when the 10 Hz sensor publishes a complete scan, and that
corrupted snapshot is held for all five 50 Hz policy steps. PPO therefore stores
2,161 LiDAR values per transition rather than both 4,995-point raw scans. At
4,096 environments and 24 rollout steps this reduces LiDAR rollout storage from
3.66 GiB to 810.4 MiB without lowering ray-cast density or changing PPO settings.

The critic and rewards can use the clean current scan, nearest-human state, and
privileged filtered planar/joint targets. The filter is set to 0.8 m proactive
clearance. The revised task adds a dense squared penalty when the existing
robot-link CBF metric falls below 0.8 m; its weight is `-3.0`. This reuses the
batched link calculation and does not add another geometry pass. Each reset
samples a 2--4 m rounded-ring crowd with randomized density and 1.3--1.9 m human
height; 25% of environments contain no people. The upstream PPO rollout,
minibatch, epoch, and optimizer settings are unchanged. Crossing the analytical
0.1 m robot-link-to-human clearance terminates the episode; physical contact
sensors are disabled. A directional progress reward uses the privileged CBF
escape velocity to teach whole-root translation without exposing that target to
the actor. The inherited global root-position reward follows the filtered safe
reference. All inherited reward weights remain at the upstream MJLab defaults,
including `0.5` for global root position and orientation.

Evaluate a revised checkpoint against a correctly blinded paired control with:

```bash
uv run python scripts/benchmark_policy_avoidance.py \
  --task-variant range-rate \
  --checkpoint logs/rsl_rl/safe_mimic_g1_live_aligned_lidar_avoidance_range_rate_link_reward/<run>/model_<iteration>.pt \
  --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --scenarios primary-only \
  --modes normal blind
```

### Raw motion-library mimic teacher

For a controlled single-motion comparison, train only the example dance with
state estimation disabled:

```bash
uv run train SafeMimic-Tracking-ExampleDance-NoStateEst-Unitree-G1 \
  --env.scene.num-envs 4096
```

This uses mjlab's normal single-NPZ command rather than the packed motion
library and logs separately under
`safe_mimic_g1_example_dance_no_state_est_tracking`. The motion defaults to
`artifacts/motions/lafan1_dance1_subject1_demo_motion.npz`; override it with
`--env.commands.motion.motion-file /absolute/path/to/motion.npz` if the cache is
elsewhere.

Train the no-LiDAR first cut with:

```bash
uv run train SafeMimic-Tracking-MotionLib-Unitree-G1 \
  --env.scene.num-envs 4096
```

For locomotion-capable tracking, the recommended state-estimation baseline
restores pelvis linear velocity and reference-anchor position in the actor:

```bash
uv run train SafeMimic-Tracking-MotionLib-StateEst-Unitree-G1 \
  --env.scene.num-envs 4096
```

It logs separately under the
`safe_mimic_g1_motion_library_state_est_tracking` experiment. Its 160-value
actor is intentionally checkpoint-incompatible with the 154-value ablation.

To test learned translational state without exposing either position or linear
velocity directly to the actor, train:

```bash
uv run train SafeMimic-Tracking-ExampleDance-ImplicitState-Unitree-G1 \
  --env.scene.num-envs 4096
```

This single-trajectory variant uses only
`artifacts/motions/lafan1_dance1_subject1_demo_motion.npz`. For the corresponding
generalized experiment over all 27,514 training motions, use:

```bash
uv run train SafeMimic-Tracking-MotionLib-ImplicitState-Unitree-G1 \
  --env.scene.num-envs 4096
```

This experiment keeps the ordinary 154-value no-state actor observation and
adds ten 50 Hz frames (0.2 s) of noisy angular velocity, projected gravity,
joint position/velocity, and previous actions. The history encoder produces a
16D latent, and a small state head predicts six explicit values (body-frame
linear velocity and reference-root position error). The policy consumes both
the six predictions and latent. A training-only target encoder embeds the
actual next proprioceptive observation for a balanced prototype loss. PPO
freezes the history representation components; the alternating representation
update minimizes the two supervised losses plus the successor-state
contrastive loss. Deployment runs the history encoder, six-value state head,
and policy; the target encoder, prototypes, critic, and simulator targets are
omitted. No local reference velocity or LiDAR is included in this ablation.
Single-dance logs are written to
`safe_mimic_g1_example_dance_implicit_state_tracking`; generalized logs use
`safe_mimic_g1_motion_library_implicit_state_tracking`.

The task defaults to
`artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1/conversion_manifest.jsonl`
and loads only its `train` split: 27,514 motions and 10,689,485 frames. Override
the library when necessary with:

```bash
uv run train SafeMimic-Tracking-MotionLib-Unitree-G1 \
  --env.commands.motion.motion-file /absolute/path/to/conversion_manifest.jsonl \
  --env.scene.num-envs 4096
```

This keeps the stock mjlab flat G1 mimic MDP aside from its packed command
backend and state-estimation-free actor setting. It has no LiDAR, people,
obstacle rewards, reference filter, or new observation terms. The actor omits
pelvis linear velocity and reference-anchor position while retaining pelvis
angular velocity and reference-anchor orientation; actor and critic shapes are
154 and 286. Reset
pose/velocity/joint perturbations, pushes, COM, encoder-bias, foot-friction, and
observation corruption remain active in training.

Each environment owns an independent motion ID and continuous motion time.
Adaptive mode draws clip-local one-second phases from the duration-weighted
WBC/dance prior, records tracking terminations as failures and completed clips
or timeouts as successes, and resamples difficult phases more often while
retaining the 10% random floor. References teleport only during environment
reset or after completing their own clip; interpolation never crosses NPZ
boundaries.

The active train split contains 9.557 GiB of packed frame tensors. Startup takes
roughly two minutes because the compressed NPZs are streamed from SSD into the
final CUDA allocation.

Play the newest local checkpoint from this task with:

```bash
uv run python scripts/play_motion_library.py
```

An explicit checkpoint and single motion can be supplied to avoid loading the
complete library for a short visualization:

```bash
uv run python scripts/play_motion_library.py /absolute/path/to/model.pt \
  --motion-file /absolute/path/to/motion.npz
```

Play mode uses mjlab's normal deterministic overrides: observation corruption,
reset perturbations, and pushes are disabled, and clips advance from their
starts. This first cut uses the generic runner because the upstream tracking
ONNX exporter assumes one reference trajectory. Adaptive state is available
through the command's `adaptive_state_dict()` API, but the generic runner does
not yet include it in checkpoints; resuming a run therefore restarts the
difficulty statistics while preserving the learned policy and optimizer.

The recommended training environment is the combined task:

```bash
uv run train SafeMimic-Tracking-Crowd-Human-Unitree-G1-Lidar \
  --env.commands.motion.motion-file artifacts/motion.npz \
  --env.scene.num-envs 4096
```

Play a checkpoint trained with the same observation contract:

```bash
uv run play SafeMimic-Tracking-Crowd-Human-Unitree-G1-Lidar \
  --checkpoint-file artifacts/lidar_policy.pt \
  --motion-file artifacts/motion.npz \
  --num-envs 1 \
  --viewer viser
```

The crowd-only physical-contact variant remains available for ablations:

```bash
uv run train SafeMimic-Tracking-Obstacles-Unitree-G1-Lidar \
  --env.commands.motion.motion-file artifacts/motion.npz \
  --env.scene.num-envs 4096
```

### Nominal-policy scene demo

The debug task keeps the upstream checkpoint's 160-element actor observation
unchanged. It is therefore useful for scene, motion, self-occlusion, and LiDAR
debugging, but the nominal policy cannot react to the people:

```bash
uv run play SafeMimic-Tracking-Flat-Unitree-G1-Lidar-Debug \
  --checkpoint-file /tmp/mjlab_cache/demo_ckpt.pt \
  --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 1 \
  --viewer viser
```

Checkpoint and motion files are not shipped by this repository. Replace the
`/tmp/mjlab_cache` paths with local artifacts when necessary. Viser prints the
selected localhost port at startup. Stop it with `Ctrl+C`.

An upstream flat checkpoint cannot be loaded directly into either LiDAR training
task because adding the 1,080 ranges changes the actor input. Use the nominal
checkpoint as a teacher or initialize a separately defined perceptive student.

### Reference-filter tuning demo

This stage tunes the reference itself; it does not train PPO or any other
policy. Run the privileged-geometry planar filter with:

```bash
uv run play SafeMimic-Reference-Filter-Crowd-Human-Unitree-G1-Lidar-Demo \
  --checkpoint-file /tmp/mjlab_cache/demo_ckpt.pt \
  --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 1 \
  --viewer viser
```

The checkpoint only satisfies the standard play interface; its actions cannot
move the robot. The first filter reads the current crowd and action-human capsule
geometry, computes an additive planar CBF velocity correction, integrates a
persistent world-space translation, and applies it to the complete reference.
Its surface-to-surface safety clearance is 0.1 m. Height, roll, pitch, and yaw
remain those of the source motion. The persistent translation is important: the
next source frame does not snap the robot back onto the unsafe unfiltered path.

A second filter covers the selected shoulders, elbows, wrists, knees, ankles,
and torso in 3D. For arm threats it first compares four vectorized 0.5 s
lookahead candidates: preserve the reference, lower both arms, lower the left
arm, or lower the right arm. A finite-rotation FK rollout propagates every
candidate through the current G1 kinematic tree and scores only the capsule
paired with each current or predicted threatened arm link. The score averages
capped pairwise clearance, so an unchanged shoulder cannot mask a useful
forearm or wrist move. The semantic candidates drive the selected side's
shoulder, elbow, and wrist joints toward the default tucked arm pose;
shoulder-roll supplies the pronounced downward motion while distal joints can
also move when finite FK predicts better clearance. The subsequent hard CBF may
still use any joint when required for immediate safety. While an arm-tuck
candidate is active, reference recovery is gated on those moving arm DOFs so it
cannot cancel the selected move; inactive DOFs and the complete arm after the
threat passes still recover to the source motion. The posture velocity is
capped at 1.5 rad/s. Both semantic posture selection and the hard link CBF use a
0.8 m surface activation margin. Each selected side is latched for 1.0 s, long
enough to lower a fully horizontal arm even if the nearest moving human clears
the wrist quickly.
This lets lowering a horizontal arm win even when that motion is orthogonal to
the instantaneous separation normal, without an extra MuJoCo forward pass.
Environments and candidates are CUDA tensor dimensions; only the fixed
29-joint tree is traversed in Python, so the filter remains parallel across
thousands of environments. A sequential
joint-space projection then adds only the motion required to satisfy every
active link-CBF half-space. Outside the threat region, the preferred correction
is zero and the reference remains the source motion. The default link surface
clearance is 0.1 m. Shoulder, elbow, and wrist joints may
deviate by up to 1.5 rad, while waist, hip, knee, and ankle joints may deviate by
up to 0.75 rad. All corrections are capped at 1.5 rad/s and the robot's soft
joint limits remain the hard positional bound.

The parameters live under `env.commands.motion.planar_filter` and
`env.commands.motion.link_filter`. The environment reports current and episode
tuning metrics for planar and joint intervention, reference offset/residual,
CBF violation, and clearance violation. These are measurements, not reward
terms used for learning.

This first filter intentionally uses privileged capsule geometry so filtering
quality can be tuned independently of perception error. The dense 5 Hz LiDAR is
still active for visualization and for the subsequent step of replacing the
oracle obstacle state with tracked LiDAR obstacles.

### Nominal policy tracking the filtered reference

To leave physics and actuation enabled and test the trained no-state tracker on
the crowd and full action human, run:

```bash
uv run play SafeMimic-Reference-Filter-Policy-Crowd-Human-Unitree-G1-Lidar-Demo \
  --checkpoint-file logs/rsl_rl/safe_mimic_g1_example_dance_no_state_est_tracking/2026-08-28_01-12-15/model_29999.pt \
  --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 1 \
  --viewer viser
```

This task keeps the checkpoint-compatible 154-value no-state actor observation
and normal 29-joint action interface. At every 50 Hz control step, the current
source frame is translated onto the live robot root and yaw-rotated onto its
tracking anchor before
the planar and link/joint filters run. This removes accumulated global tracking
drift from the filter's starting point while preserving the reference's local
body pose and world-relative obstacle geometry. The filtered reference is never
written into the simulated robot after reset. Gravity, contacts, actuators,
rewards, and tracking terminations remain enabled. LiDAR is visualization-only;
obstacle reactions come from the privileged reference filter.

The existing no-state checkpoint does not observe reference-root translation or
base linear velocity. It can therefore respond immediately to the filtered
joint targets, but the planar root displacement is not directly controllable by
this frozen actor. A learned step-away response will eventually require either
a deployable local planar command input or conversion of the planar correction
into a locomotion/joint-reference segment.

### Exact reference-replay baseline

Use the replay task to separate reference-filter behavior from tracking-policy
error:

```bash
uv run play SafeMimic-Reference-Replay-Crowd-Human-Unitree-G1-Lidar-Demo \
  --checkpoint-file /tmp/mjlab_cache/demo_ckpt.pt \
  --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 1 \
  --viewer viser
```

The task accepts the nominal checkpoint because its observation and action spaces
are unchanged, but the checkpoint's joint targets do not control the displayed
robot. At every 50 Hz control step, the motion command writes the exact reference
root pose/velocity and joint pose/velocity, forwards MuJoCo kinematics, and only
then samples LiDAR. Gravity, contacts, actuation, pushes, and all task
terminations are disabled. The moving crowd and full action human remain active.

The dense `185 x 27` scan publishes at 5 Hz by default: ten interleaved phases
cast 513 padded rays per control step. A 10 Hz option remains available to code
that constructs the config directly:

```python
from safe_mimic.tasks.env_cfg import (
  unitree_g1_kinematic_reference_lidar_demo_env_cfg,
)

cfg = unitree_g1_kinematic_reference_lidar_demo_env_cfg(scan_hz=10)
```

The nominal tracker does use global pose, but not simply as a termination signal.
Its actor receives the desired anchor position and orientation expressed relative
to the robot torso frame. Its reward also scores global anchor position,
orientation, and global body velocities. By contrast, the standard termination
rules deliberately leave planar position and heading unconstrained: anchor
position and end-effector checks are Z-only, while anchor orientation checks
tilt through projected gravity. This replay task clears even those termination
rules so it can serve as a clean reference-filter test fixture.

## Sensing choice

mjlab supports batched ray casting but does not ship a spherical Mid-360 pattern.
This package adds an interleaved body-mounted pattern and held-scan timing:

- The legacy obstacle tasks use a sparse `180 x 6 = 1,080` grid over 360 degrees
  horizontally and `0, -10, ..., -50` degrees vertically.
- The from-scratch avoidance task uses the true quarter-resolution
  `185 x 27 = 4,995` grid: approximately 2 degrees horizontally and vertically.
  Five interleaved phases cast 999 padded rays per 50 Hz policy step and publish
  a completed scan at 10 Hz. The observation is held for five policy steps.
- Ranges at or below 0.3 m and beyond 5 m are invalidated. This rejects the
  near-body region and limits the obstacle horizon.
- The avoidance actor receives normalized, corrupted current and previous
  completed scans plus scan age; its critic receives the clean current scan and
  compact privileged human/teacher state.
- The sensor is attached to the `mid360` MJCF site and uses the robot's full base
  orientation. Group-3 G1 proxies, the head-side pillars, and the shoulder plank
  reproduce hardware self-occlusion without physical contacts.
- The debug task uses the same `185 x 27 = 4,995` grid for visual inspection.
  Red Viser points are ray hits; ray arrows are disabled.
- The exact replay demo uses the same dense pattern at 5 Hz by default, with a
  10 Hz configuration option. Its robot pose is written before every LiDAR phase.

The exact Livox non-repeating scan pattern and packet timing are not simulated.
The control design and camera/LiDAR trade study are in
[docs/sensing.md](docs/sensing.md), and measured stepping costs are in
[benchmark.md](benchmark.md).

## Full G1 mimic motion library

The BONES-SEED Unitree G1 source data lives once on this machine, at
`~/twist2/seed/g1/csv/` (all 142,220 extracted G1 CSV trajectories, about
49 GB). `artifacts/bones-seed/g1` is a symlink to that directory, so the
paths below still resolve. The original `g1.tar.gz` and `soma_uniform.tar.gz`
archives and the Git LFS object cache were removed to save space (2026-09-19
and 2026-09-22). Only the 3,000 SOMA BVH clips named in
`datasets/walk_punch_kick_1000/manifest.jsonl` remain extracted under
`artifacts/bones-seed/soma_uniform/`; re-download the archives from
`https://huggingface.co/datasets/bones-studio/seed` if a wider motion
selection is ever needed. `artifacts/bones-seed/` is ignored by Git.

Build the deterministic full-corpus filter manifests with:

```bash
uv run python scripts/build_g1_motion_library.py --workers 16
```

The filter keeps original, upright motions and deliberately retains dances.
Mirrored clips are omitted from the stored corpus because reflection is cheaper
as online augmentation. Ground, inverted, stunt, and assistive-device motions
are rejected from metadata. The remaining clips must satisfy the G1 joint
limits (with 0.02 rad tolerance), 35 rad/s joint-speed and 5 m/s planar
root-speed limits, and a 100-degree root-tilt limit. Normal motions use a
0.40--0.95 m root-height envelope; dance and jump motions use the wider
0.35--1.20 m envelope.

The current full run produced:

| Result | Count |
|---|---:|
| Metadata-compatible original clips | 58,934 |
| Accepted clips | 54,525 |
| Kinematically rejected clips | 4,409 |
| Accepted dance-tagged clips | 6,636 |
| Training split | 49,394 |
| Validation split | 5,131 |
| Accepted duration | 108.0 hours |

The split is deterministic by source take, so related clips do not leak between
training and validation. Filtering outputs live under
`artifacts/bones-seed/datasets/g1_general_mimic_v1/`; `filtered_all.jsonl` is
the conversion input, while `train.jsonl`, `validation.jsonl`,
`rejected_kinematic.jsonl`, `csv_manifest.yaml`, and `summary.json` retain the
decisions and statistics.

Convert every accepted original to the exact mjlab tracker schema at 50 Hz:

```bash
uv run python scripts/convert_g1_tracker_npz.py --workers 16
```

The converter interpolates the native 120 Hz root pose and 29 joint angles onto
an exact 50 Hz time grid, runs MuJoCo forward kinematics for all 30 G1 bodies,
and writes compressed NPZs containing `fps`, `joint_pos`, `joint_vel`,
`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`. Existing
files are resumable; add `--validate-existing` to recheck them during a rerun.

The completed conversion contains 54,525 NPZ files, 19,467,141 frames, and zero
conversion errors. It occupies 31,720,546,078 bytes (about 30 GiB) under
`artifacts/bones-seed/datasets/g1_general_mimic_v1/npz_50hz/`. Use
`npz_manifest_50hz.yaml` as the complete train/validation index;
`npz_conversion_summary.json` and `npz_conversion_errors.jsonl` record the
batch result. Every NPZ can be passed directly as mjlab's `--motion-file`.

The full float32 tensor payload is about 32.5 GiB before loader overhead, so it
does not fit simultaneously on this workstation's 24 GiB GPU. This does not
limit single-motion tracking. Training over the complete manifest will require
a multi-motion loader with a GPU-resident sampled subset/cache, or a more compact
bank that reconstructs body kinematics online. Do not duplicate mirrored NPZs;
apply left/right reflection when that loader samples a motion.

### Expanded paired general-mimic bank

The recommended stored-mirror bank assigns 70% of its 10 GiB original payload
to the active WBC filtered list and 30% to accepted dances, then includes every
selected motion's actual `_M` partner. Build the deterministic selection with:

```bash
uv run python scripts/build_g1_paired_sample.py \
  --policy wbc-dance \
  --target-gib 10
```

The deterministic size-budgeted selection initially produced 11,927 WBC pairs
and 3,928 dance pairs. A post-conversion stationary-jump filter removes paired
jump/hop/leap clips that remain within 0.35 m of their starting pelvis XY, plus
clips explicitly labelled “in place.” The active result has 11,459 WBC pairs
and 3,645 dance pairs: 15,104 pairs, 30,208 motions, and 19.350 GiB of full
30-body float32 tracker payload. Its payload mixture is 70.467% WBC and 29.533%
dance. Materialize the remaining conversions with:

```bash
uv run python scripts/convert_g1_tracker_npz.py \
  --manifest artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1/conversion_manifest.jsonl \
  --output-dir artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1/npz_50hz \
  --workers 16
```

The active manifests index 30,208 NPZs, 11,594,251 frames, and 17.594 GiB of
compressed NPZ data. The 1,502 excluded clips are recorded in
`excluded_stationary_jumps.jsonl`; their cached NPZs, original archive,
extracted CSVs, earlier banks, and reused NPZs remain available.

`PackedNpzMotionLib` loads the bank in a WBC-style flat tensor layout while
keeping every clip boundary explicit. It selects the default 14 tracker bodies
on the CPU and copies each clip directly into one preallocated final CUDA slice;
there is no list of full-body CUDA clips followed by a second `torch.cat` copy.
The active all-split bank contains 10.366 GiB of packed 14-body frame tensors.
The pre-pruning 31,710-motion benchmark occupied 10.711 GiB, loaded in about two
minutes from SSD, and sampled/interpolated 4,096 references in 0.370 ms.

`AdaptiveMotionSampler` supplies boxing-teacher-style randomized RSI and
difficulty practice. It uses 254,390 clip-local one-second bins, maintains an
EMA failure rate, retains a 10% duration-weighted random floor, couples mirrored
phases, and normalizes WBC and dance independently so the curated 70/30 mixture
cannot drift. Adaptive sampling plus interpolation costs 0.374 ms per 4,096;
updating difficulty for and resampling an artificial batch in which all 4,096
environments reset together costs 2.484 ms. Updates occur only for completed
episodes in training, so the latter is a conservative non-amortized cost.

The standard mjlab pose, velocity, joint-position, push, COM, encoder-bias, and
foot-friction randomizations remain active in training, alongside Safe Mimic's
crowd, human-size, LiDAR, and scene randomization. Adaptive sampling determines
which reference phase is practiced; it does not replace domain randomization.

See [docs/g1_wbc70_dance30_10g.md](docs/g1_wbc70_dance30_10g.md) for the
selection, loader API, adaptive update contract, measurements, and current
integration boundary. The earlier 5 GiB-original bank is retained as a smaller
ablation and documented in
[docs/g1_wbc70_dance30_sample.md](docs/g1_wbc70_dance30_sample.md).

For an ablation that instead follows the complete unfiltered SEED distribution,
run:

```bash
uv run python scripts/build_g1_paired_sample.py
```

That result has 8,176 original/mirror pairs: 16,352 motions and 9.995 GiB
of materialized float32 tracker tensors. It is stratified by package and
category against all 71,088 mirror-paired originals. This intentionally samples
the **unfiltered** distribution, so 23.44% of its pairs fail the complete Safe
Mimic upright/kinematic filter. See
[docs/g1_paired_sample.md](docs/g1_paired_sample.md) for the distribution,
safety breakdown, manifests, storage use, and reproduction commands.

## Balanced human-motion dataset

Build the deterministic BONES-SEED manifest with 1,000 original trajectories
per family:

```bash
uv run python scripts/build_bones_seed_dataset.py
```

The output is written to
`artifacts/bones-seed/datasets/walk_punch_kick_1000/`. `manifest.jsonl`
contains all 3,000 entries, with separate `walk.jsonl`, `punch.jsonl`, and
`kick.jsonl` files alongside it. The families are disjoint and do not use
mirrored copies. Here, punch and kick are broad kinematic labels for suitable
outward arm and leg actions; the punch filter rejects sustained overhead arms.

## Packed human trajectories

Prebuild every retained temporal event as a six-second-or-shorter capsule path:

```bash
uv run python scripts/build_capsule_path_bank.py
```

The current analysis bank contains 8,737 paths from all 3,000 source clips.
Float16 capsule poses occupy about 630 MiB on disk. Build the transition-ready
motion graph separately:

```bash
uv run python scripts/build_transition_index.py
```

The default graph keeps upright, supported, active, non-prop segments with at
least eight walking connectors at both action boundaries. It currently retains
1,574 walk, 745 arm-action, and 356 leg-action segments. Every decision and
rejection reason is saved under `transition_index_v3/`.

Build the compact skeleton bank after pruning:

```bash
uv run python scripts/build_skeleton_path_bank.py
```

It stores only the 2,675 approved paths and the 24 joints required by the
capsule fit. The current float16 source bank is 257 MiB. Compile all approved
transition chains into direct-playback keypoints once:

```bash
uv run python scripts/build_packed_human_trajectory_bank.py \
  --kind primary \
  --output artifacts/bones-seed/datasets/packed_primary_human_trajectory_v1

uv run python scripts/build_packed_human_trajectory_bank.py \
  --kind crowd \
  --output artifacts/bones-seed/datasets/packed_crowd_human_trajectory_v1
```

The primary output contains 8,808 transition-matched
`walk -> punch/kick -> walk` chains and occupies 0.81 GiB in float16. The crowd
output contains 800 three-action chains and occupies 0.038 GiB. Root alignment,
velocity-aware 0.2 s inertialization, and the 24-joint forward kinematics run
only in this offline compiler. At training startup the packed arrays are copied
to VRAM. Runtime work is an indexed frame gather, interpolation, placement,
capsule fit, and direct geom-pose write; it performs no motion reads from SSD or
system RAM. The action human updates at 10 Hz, the crowd at 5 Hz, and LiDAR
publishes at 10 Hz. Viser play configurations retain the original skeleton
composer because it supplies the complete joint pose needed to skin the SOMA
debug mesh.

Only entry walks with enough natural root travel are sampled. At reset, the
full human starts 2--4 m from the robot, its walking phase and playback rate are
selected to cover the approach without foot-skating, and the midpoint of the
annotated punch/kick is placed on the predicted robot path. The reference-filter
demo uses the same 2--4 m spawn and one-to-three-second approach distribution as
training; the nominal calibration scene keeps its fixed 3 m, three-second
approach.

Transitions are inertialized after forward kinematics in aligned global skeleton
space during compilation. This is important for the SOMA hierarchy, which splits
root translation between synthetic `Root` and `Hips` joints: decaying offsets in
the clips' local coordinate frames can otherwise create multi-meter jumps after
`walk -> action` and `action -> walk` boundaries. Runtime primary placement uses
the packed root path to choose the 2--4 m entry frame without online bisection.
Crowd reset placement reads the newly written robot root directly from `qpos`, so
the ring is centered on the robot before mjlab's post-reset forward-kinematics
pass.

Do not split the dataset into thousands of `.npz` files. Fixed-shape bank arrays
give one contiguous startup transfer and cheap indexed CUDA gathers; one-file
`.npz` exports remain useful only for videos and individual debugging cases.

## Stationary crowd boundary

Build the reproducible 100-segment standing arm-action subset and its compact
skeleton bank with:

```bash
uv run python scripts/select_stationary_arm_actions.py
uv run python scripts/build_skeleton_path_bank.py \
  --transition-index artifacts/bones-seed/datasets/standing_arm_actions_100 \
  --output artifacts/bones-seed/datasets/skeleton_path_bank_standing_arm_actions_100
```

The selector requires an approved arm-action segment, at least 18 cm of hand
excursion, no more than 25 cm of local pelvis drift, no sustained overhead-arm
motion, and no sitting, jumping, prop, or environment-dependent metadata. The
current deterministic result has 100 segments from 69 source motions and 45
actors. Its float16 skeleton poses occupy about 10 MB and are copied to VRAM
once at startup.

Training samples a 2--4 m boundary scale and a superellipse exponent from 2
through 8 per environment, covering circles through rounded-square layouts. It
then samples the background occupancy uniformly from zero through that exact
boundary's packed capacity. This includes empty, sparse, and fully crowded
episodes without forcing overlapping people onto a small ring. Active members
are resampled by boundary arc length with even stratification; the packed end is
approximately 0.62 m center-to-center, with up to +/-0.25 m radial jitter. The
reference-filter demo uses this same complete spatial and occupancy distribution;
only its mesh rendering and denser visualization rays differ from training. The
nominal calibration scene remains dense and fixed at 3 m for repeatable visual
comparisons.

The robot starts at the boundary's center. Slots remain fixed in world space;
people do not orbit or follow the robot. Every member is aligned by anatomical
facing toward the robot without heading jitter. Each person's standing height is
sampled uniformly from 1.3--1.9 m against the physical 1.7605 m SOMA bind mesh;
width/depth proportions, proxy thickness, motion, phase, and 0.8--1.2x playback
speed are independent per member. The same height randomization applies to the
full action human. Pelvis XY is fixed to its slot. Packed crowd chains use the
same globally inertialized transitions and ping-pong at their endpoints, avoiding
a loop-boundary pose teleport while updating at 5 Hz in the optimized task.

Crowd ray casting currently uses five inflated animated capsules per
person. Standing height, width/depth, radius scale, and radius margin make its
dimensions independent per person. The optimized combined task uses that same
proxy for analytical proximity while keeping it out of physical contact
generation. The independently approaching/action human retains all 18 capsules.
The full SOMA skin is visualization-only; Viser clusters it on a 2 cm grid and
updates it at 5 Hz so debug rendering does not dictate training throughput.

The 5 m LiDAR range remains unchanged, so the complete 2--4 m crowd boundary is
within nominal range at reset unless self-occlusion blocks a return.

Compile a single annotated clip to cross an mjlab robot reference:

```bash
uv run python scripts/compile_human_intersection.py \
  --robot-motion artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --family walk \
  --intersection-time 5
```

For a PHP-style `walk -> action -> walk` trajectory, use the pruned graph and
skeleton-level inertialization:

```bash
uv run python scripts/compile_composed_human_intersection.py \
  --robot-motion artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --action-family punch \
  --action-query jab \
  --intersection-time 8
```

The offline command is useful for inspecting selected source clips. The packed
runtime filters for sufficiently long approaches, gathers the closest natural
2--4 m entry frame, and adjusts playback speed so the action human reaches the
predicted robot path at the annotated punch/kick midpoint. The walking segments
provide the approach and departure.

## Validation

Run formatting/static checks and the test suite with:

```bash
uv run ruff check .
uv run python -m pytest -q
```

The transition regression tests check both exact boundary continuity and the
post-boundary decay that previously produced visible root jumps. The current
4,096-environment packed-playback result is recorded in
[benchmark.md](benchmark.md).
