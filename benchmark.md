# LiDAR stepping benchmark

## Offline-packed human playback

The action human and crowd now load offline-stitched float16 keypoint banks into
VRAM instead of composing three source motions and running 24-joint forward
kinematics during every environment reset. The primary bank contains 8,808
`walk -> action -> walk` chains (0.81 GiB); the crowd bank contains 800
three-action chains (0.038 GiB). Human size/proportion, thickness, playback
speed, phase, crowd layout/density, and robot-relative placement remain online
randomizations.

The preceding phase profile measured the old mature 24-step rollout at 8.65 s:

| Old collection phase | Time | Share |
|---|---:|---:|
| Primary/crowd reset composition | 3.76 s | 43.5% |
| Human step animation | 3.71 s | 42.9% |
| CBF | 0.38 s | 4.4% |
| MuJoCo simulation | 0.25 s | 2.9% |
| LiDAR sensing | 0.16 s | 1.8% |
| Observations | 0.11 s | 1.3% |

A six-iteration end-to-end PPO run with 4,096 environments and the unchanged
24-step rollout measured warmed collection times of
`1.683, 1.674, 1.635, 1.694, 1.676 s`: **1.672 s** mean with a 0.020 s population
standard deviation. This is **5.17x faster** than the profiled 8.65 s rollout
and removes **80.7%** of collection time. This final measurement updates capsule
half-lengths, bounds, centers, and orientations at every 5/10 Hz pose refresh;
the speedup does not freeze the animated limb silhouette. The new run was a
from-scratch policy with mean episode lengths near 6.5 steps, so it exercised
resets substantially more often than the old approximately 118-step
mature-policy profile; the result is not obtained by hiding reset cost.

The packed compiler and runtime were also checked on both banks for finite
values and transition continuity. Viser play tasks deliberately retain the
online full-skeleton composer for mesh skinning; headless training uses packed
playback.

## PPO collection-path optimization

The dense avoidance task still casts a complete `185 x 27` scan at 10 Hz and
retains the current and previous completed scans. Moving deterministic angular
minimum pooling ahead of PPO rollout storage changes only where preprocessing
occurs:

| Collection representation | LiDAR values/transition | 4,096 env x 24-step storage |
|---|---:|---:|
| Raw current + previous scans + age | 9,991 | 3.659 GiB |
| Two `24 x 3` directional grids + age | 145 | 54.4 MiB |
| Two `72 x 6` directional grids + age (previous) | 865 | 324.4 MiB |
| Two `120 x 9` directional grids + age (adopted) | 2,161 | 810.4 MiB |

The adopted `120 x 9` representation is 4.62x smaller than raw rollout storage
and removes about 2.87 GiB of GPU writes per rollout. Point-level Gaussian
noise, isolated-ray dropout, and sector dropout are still applied to the dense
scans before pooling. The pooled actor and critic observations are cached by
complete-scan publication, so a 10 Hz snapshot—including its corruption—is
immutable for the following five 50 Hz policy steps.

The `24 x 3` clean six-iteration run measured collection times of
`5.888, 5.597, 5.638, 5.573, 5.398, 5.336 s`, for a five-iteration warmed mean
of **5.508 s**. The previous `72 x 6` run measured
`5.887, 5.588, 5.641, 5.629, 5.670, 5.623 s`, for a warmed mean of **5.630 s**.
Six times the directional resolution therefore costs only 2.2% collection
throughput. It remains faster than the prior 5.719 s compact-critic best and the
early 5.82--5.94 s range. Policy-free environment stepping previously took
2.16 s per rollout, so most remaining collection time lies in policy/value
evaluation, normalization, and rollout bookkeeping rather than LiDAR storage.

Measured on an NVIDIA GeForce RTX 4090 with 4,096 MJLab environments. The
environment ran headless with the same scene and physics configuration in every
case. The benchmark measures environment stepping without policy inference.

## Unified 4,096-environment relative comparison

All human-bearing rows below use the sparse 180 x 6 LiDAR, either at
5 or 10 Hz. The baseline is upstream nominal tracking with no LiDAR and no
humans. Every row uses 4,096 environments, 20 warmup steps, three trials of 40
policy steps, headless execution, and no policy inference.

Another workload occupied the GPU. To reduce drift in the denominator, nominal
tracking was measured before and after the sweep at 15.120 and 14.656 ms/step;
their midpoint, **14.888 ms/step**, is the 1.00x baseline below. Ratios are the
most useful result; absolute timings remain approximate.

| Configuration | LiDAR | Human proxies | Step time | Relative to nominal | Throughput |
|---|---:|---:|---:|---:|---:|
| Nominal tracking; no added sensors or humans | none | 0 | 14.888 ms | 1.00x | 275,121 transitions/s |
| LiDAR only | 5 Hz | 0 | ~16.605 ms | ~1.12x | ~246,673 transitions/s |
| LiDAR only | 10 Hz | 0 | 17.004 ms | 1.14x | 240,884 transitions/s |
| One full crossing/fighting human | 5 Hz | 18 | ~25.651 ms | ~1.72x | ~159,682 transitions/s |
| One full crossing/fighting human | 10 Hz | 18 | ~26.329 ms | ~1.77x | ~155,570 transitions/s |
| Dense 30--58-person crowd; five proxies/person | 5 Hz | 320 compiled | 110.437 ms | 7.42x | 37,089 transitions/s |
| Dense 30--58-person crowd; five proxies/person | 10 Hz | 320 compiled | 113.463 ms | 7.62x | 36,100 transitions/s |
| Five-proxy crowd + one full 18-proxy human | 5 Hz | 338 compiled | ~126.176 ms | ~8.48x | ~32,463 transitions/s |
| Five-proxy crowd + one full 18-proxy human | 10 Hz | 338 compiled | ~124.755 ms | ~8.38x | ~32,832 transitions/s |
| Dense crowd before simplification; 18 proxies/person | 10 Hz | 1,152 compiled | 261.126 ms | 17.54x | 15,686 transitions/s |

The first auxiliary LiDAR-only, single-human, and mixed runs accidentally
retained training-mode push events. A corrected 10 Hz LiDAR-only run measured
17.004 instead of 21.604 ms. Values marked with `~` use that 4.600 ms paired
difference as a rough correction, as requested; the nominal, corrected 10 Hz
LiDAR-only, dense-crowd, and pre-simplification rows are direct measurements.

The 5 Hz and 10 Hz human-heavy results are effectively tied under the observed
GPU contention. Proxy count, broadphase work, privileged observations, and
proximity/contact evaluation dominate; LiDAR cadence is not the main remaining
bottleneck. The five-proxy crowd cuts the previous dense crowd from 17.54x to
7.62x nominal cost at 10 Hz. Adding one fully articulated crossing human raises
that to roughly 8.38x.

- Physics timestep: 5 ms
- Policy timestep: 20 ms (50 Hz)
- LiDAR valid range: 0.3-5 m
- LiDAR field of view: 360 degrees horizontally, 0 through -52 degrees vertically
- Warmup: 60 steps
- Measurement: median of five trials, 60 steps per trial

| Resolution | Scan rate | Rays cast per policy step | Step time | Relative to sweep baseline | Throughput |
|---|---:|---:|---:|---:|---:|
| No LiDAR | - | 0 | 10.964 ms | 1.00x | 373,600 transitions/s |
| 185 x 27 (4,995 points/scan) | 5 Hz | 513 padded | 12.398 ms | 1.13x | 330,374 transitions/s |
| 185 x 27 (4,995 points/scan) | 10 Hz | 999 | 14.045 ms | 1.28x | 291,630 transitions/s |
| 360 x 52 (18,720 points/scan) | 5 Hz | 1,872 | 18.355 ms | 1.67x | 223,150 transitions/s |
| 360 x 52 (18,720 points/scan) | 10 Hz | 3,744 | 24.260 ms | 2.21x | 168,836 transitions/s |

The 360 x 52 grid has 1 degree horizontal spacing and approximately 1.02 degree
vertical spacing. This corresponds to approximately 8.7-8.9 cm sample spacing at
5 m on a perpendicular surface.

At 5 Hz, a completed scan is accumulated over ten policy steps and then held for
ten policy steps. At 10 Hz, it is accumulated over five policy steps and held for
five. The 5 Hz publication cadence was verified directly: the held scan changed
only at steps 9 and 19 in a 22-step check.

The uniform spherical grids approximate Mid-360 point density. They do not model
the exact, non-repeating Livox scan pattern or packet timing.

Reproduce a run with:

```bash
MPLCONFIGDIR=/tmp/safe-mimic-mpl uv run python scripts/benchmark_lidar.py \
  --case lidar \
  --scan-hz 5 \
  --resolution full \
  --motion-file /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 4096 \
  --warmup-steps 60 \
  --steps-per-trial 60 \
  --trials 5
```

## Online human-path smoke test

The runtime now keeps the complete transition-approved 24-joint skeleton bank
on CUDA. A 4,096-environment smoke test sampled a transition-matched
`walk -> action -> walk` triplet per environment, ran skeleton-space transition
setup, placed each action on its target crossing, performed forward kinematics,
and generated the first 18-capsule pose batch.

- CUDA-resident immutable bank: 257.27 MiB
- Peak allocation after bank construction: 308.67 MiB
- Active composed humans: 4,096 / 4,096
- Schedule plus first pose batch: 223.52 ms

This is reset/setup work, not per-step work. Pose refreshes occur at 10 Hz and
held calls do not run FK. The timing is a one-run correctness and memory smoke
test, not a representative training-throughput result.

Reproduce the smoke test with:

```bash
uv run python scripts/benchmark_composed_human.py \
  --bank artifacts/bones-seed/datasets/skeleton_path_bank_transition_v3 \
  --num-envs 4096 \
  --device cuda
```

## Dense crowd relative benchmark

Measured with another workload occupying the GPU, so only the within-session
ratio is meaningful. Both cases used 4,096 obstacle-task environments, the
180 x 6 LiDAR at 10 Hz, 20 warmup steps, and three trials of 40 policy steps.
Policy inference was not included.

| Crowd layout | Compiled people | Expected active people | Capsule geoms | Step time | Relative to current nominal* | Relative to sparse |
|---|---:|---:|---:|---:|---:|---:|
| Sparse annulus, 4--12 active | 12 | 8 | 216 | 101.715 ms | 6.83x | 1.00x |
| 3--6 m ring, 0.75 m spacing | 48 | about 38 | 864 | 236.573 ms | 15.89x | 2.33x |
| 3--6 m ring, 0.62 m spacing | 64 | about 43 | 1,152 | 285.606 ms | 19.18x | 2.81x |

\* The nominal-relative column uses the newer 14.888 ms bracketing baseline;
these historical crowd rows came from an earlier contended-GPU session, so use
them as approximate context rather than a strict paired ratio.

The final gap-safe 0.62 m ring retained 35.61% of the sparse configuration's
throughput: 14,341 versus 40,269 environment transitions per second. Trial
times were 11.964, 11.424, and 10.406 seconds for the final ring; 9.463, 9.490,
and 9.394 seconds for the 0.75 m ring; and 4.069, 4.141, and 4.057 seconds for
the sparse crowd. The external GPU workload changed noticeably during the
final trials, so treat the 2.81x slowdown as approximate; two 64-slot runs
bracketed the slowdown at roughly 2.6--2.8x.

This includes simulation, sensors, observations, rewards, and the 10 Hz crowd
update, but not actor/critic network execution. At this stage, the dense task
grew the privileged critic capsule-vector input from 648 to 3,456 scalars; the
five-proxy optimization below later reduced it to 960.

### Five-proxy crowd optimization

The stationary crowd was subsequently reduced from 18 articulated capsules per
person to five inflated LiDAR/contact proxies: one merged body/head capsule,
left/right arm capsules, and left/right leg capsules. The SOMA mesh remains a
visualization-only skin and does not participate in physics or ray casting.

A same-session paired benchmark used the final 64-slot, 0.62 m ring with the
sparse 180 x 6 LiDAR at 10 Hz. Both runs used 4,096 environments, 20
warmup steps, three trials of 40 steps, and no policy inference.

| Crowd proxy | Crowd geoms | Critic proxy scalars | Step time | Throughput |
|---|---:|---:|---:|---:|
| 18 capsules/person | 1,152 | 3,456 | 261.126 ms | 15,686 transitions/s |
| 5 capsules/person | 320 | 960 | 113.463 ms | 36,100 transitions/s |

The five-proxy layout is **2.30x faster** and removes **56.5%** of the dense
crowd's step time in this paired run. It is only about 12% slower than the
historical 12-person sparse crowd result, while retaining roughly 30--58 active
people in a dense ring. GPU contention still makes absolute timings approximate.

### Optimized five-proxy crowd plus full human

The final target is the dense five-capsule crowd plus one independently composed,
physically collidable 18-capsule action human. All rows below use 4,096
environments, the sparse 180 x 6 LiDAR, 20 warmup steps, three trials of
40 policy steps, and no policy inference. A separate GPU workload remained active,
so the median and within-session ratios matter more than the absolute timings.

| Optimization stage | LiDAR | Crowd pose | Crowd bodies | Crowd critic scalars | Step time | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Physically collidable five-proxy crowd + full human | 10 Hz | 10 Hz | 320 | 960 | 116.176 ms | 35,257 transitions/s |
| Ray-only crowd; exact ray geometry retained | 10 Hz | 10 Hz | 320 | 960 | 91.603 ms | 44,715 transitions/s |
| Trim unreachable slots, 64 -> 58 | 10 Hz | 10 Hz | 290 | 870 | 90.079 ms | 45,471 transitions/s |
| Also hold crowd poses at 5 Hz | 10 Hz | 5 Hz | 290 | 870 | 87.509 ms | 46,806 transitions/s |
| Drop redundant crowd critic vectors | 10 Hz | 5 Hz | 290 | 0 | 85.437 ms | 47,942 transitions/s |
| One ray-only crowd body; direct per-world geom poses | 10 Hz | 5 Hz | 1 | 0 | 71.483 ms | 57,300 transitions/s |
| Also omit Viser-only retained joints in headless training benchmark | 10 Hz | 5 Hz | 1 | 0 | **70.945 ms** | **57,735 transitions/s** |
| Same final configuration, 5 Hz LiDAR | 5 Hz | 5 Hz | 1 | 0 | **70.550 ms** | **58,058 transitions/s** |

A fresh nominal-tracking measurement in the same contended session was 14.489
ms/step and 282,699 transitions/s. The recommended 10 Hz LiDAR configuration is
therefore about **4.90x nominal step cost**. Relative to the physically collidable
mixed case, it removes **38.9% of step time** and raises throughput by **1.64x**.
Relative to the first ray-only implementation, it removes another **22.5%**.

The main win is structural. A ray-only crowd does not need one mocap body per
capsule. The optimized asset stores all 290 geoms under one mocap root and writes
per-environment `geom_pos`, `geom_quat`, and `geom_size` directly. This leaves the
five-capsule silhouettes unchanged for MuJoCo Warp ray casting while avoiding 290
body transforms per environment on every physics substep. A 32-environment CUDA
check measured zero center and size error against the sampler and a maximum
quaternion error of 2.98e-7.

The production tradeoff is deliberate:

- The 58-slot capacity is lossless for the configured ring. Its mathematical
  maximum is `floor(2*pi*(6.0 - 0.25)/0.62) = 58` active people.
- The crowd remains exact for LiDAR and all-five-capsule analytical proximity,
  but it no longer produces physical contacts. The separate full action human
  remains physically collidable and keeps its contact penalty.
- The critic already receives clean, noiseless LiDAR. Removing its redundant 870
  crowd capsule vectors does not change the actor observation or sensor fidelity.
- Holding background crowd animation at 5 Hz saves about 1--2%; the full action
  human and hardware-matched LiDAR remain at 10 Hz.
- Reducing LiDAR from 10 to 5 Hz saves less than 1% in the final scene, so 10 Hz is
  the recommended default.

Reproduce the recommended case with:

```bash
MPLCONFIGDIR=/tmp/safe_mimic_mpl uv run python scripts/benchmark_lidar.py \
  --case lidar \
  --task combined-ray-only \
  --scan-hz 10 \
  --crowd-update-hz 5 \
  --resolution sparse \
  --motion-file /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 4096 \
  --warmup-steps 20 \
  --steps-per-trial 40 \
  --trials 3
```

## Checkpoint collision-avoidance ablation

Checkpoint `model_16000.pt` from the
`2026-08-29_01-46-09` live-aligned LiDAR run was evaluated on the first episode
of 1,024 identically seeded environments. Inference was deterministic. Pushes
and observation corruption were disabled to isolate perception, while startup
mass, center-of-mass, encoder-bias, friction, human-size, human-motion, and
scene-layout randomization remained enabled.

The paired control replaces the actor's 2,160 current/previous directional
LiDAR values with `1.0` (max range/no return), while preserving the real scan
age. This all-miss input is not an exotic input: obstacle-free scenes were 25%
of the training distribution. The critic is irrelevant during inference.

| Scenario | Actor LiDAR | Collision, all episodes | Collision, human-present episodes | Crowd collision | Primary-human collision | Timeout | Minimum-clearance p05 |
|---|---|---:|---:|---:|---:|---:|---:|
| Training distribution | normal | 0.20% | 0.26% | 0.00% | 0.20% | 99.80% | 0.578 m |
| Training distribution | blinded | 25.88% | 34.06% | 25.68% | 0.20% | 74.41% | 0.076 m |
| Humans always enabled | normal | 0.68% | 0.68% | 0.00% | 0.68% | 99.32% | 0.554 m |
| Humans always enabled | blinded | 34.38% | 34.38% | 33.98% | 0.39% | 65.82% | 0.062 m |
| Primary moving human only | normal | 0.39% | 0.39% | 0.00% | 0.39% | 99.61% | 4.192 m |
| Primary moving human only | blinded | 0.59% | 0.59% | 0.00% | 0.59% | 99.41% | 4.192 m |

On the default distribution, 263 paired environments collided only when
blinded and none collided only with normal LiDAR. With humans forced on, those
counts were 347 versus 2. The corresponding approximate paired McNemar z values
were 16.2 and 18.5. This is strong evidence that the checkpoint learned to use
LiDAR to avoid crossing the stationary annular crowd boundary, rather than
merely surviving due to the reference filter or termination distribution. The
ring is sampled 2--4 m from the robot, so this should not be interpreted as
evidence of close-range reactive dodging.

The benchmark does **not** establish avoidance of the independently moving
18-capsule human. Once the crowd is removed, that human's fifth-percentile
minimum clearance is about 4.19 m in both arms; the scene did not create a
meaningful encounter. The packed primary event chose its future intersection
from the source motion's raw world position plus environment origin. The robot
task, however, live-aligns the source reference to the actual robot each step.
The resulting coordinate-frame mismatch placed the moving human around the old
source trajectory instead of the robot.

An additional 256-environment test used the exact online human composer selected
by Viser play mode, with the crowd disabled. This reproduces the visually close
encounter:

| Online/play primary human | Collision | Timeout | Minimum-clearance p05 |
|---|---:|---:|---:|
| Normal LiDAR | 30.86% | 69.14% | -0.142 m |
| Blinded LiDAR | 33.59% | 66.80% | -0.140 m |

There were 36 paired episodes that collided only when blinded and 29 that
collided only with normal LiDAR (`z ~= 0.87`). The small 2.73-point difference
is not meaningful evidence of avoidance. Therefore the current checkpoint has
**not** learned useful avoidance of the close moving human seen in play.

The original online/play and packed/training discrepancy had two sources. Play mode uses
the online composer and forces the robot motion to start at frame zero with RSI
pose/velocity randomization disabled; training uses the packed human bank and
adaptive/randomized source frames. Both primary schedulers targeted raw
source-motion world positions, so frame zero happened to produce the visible
encounter while randomized training frames generally did not.

Both schedulers now snapshot the robot's actual free-joint qpos at scheduling
time and place the human action point there. The target remains fixed afterward,
so the robot can move away while the human continues toward the intended
intersection. This direct qpos read is important during reset: the new robot
state has already been written, but derived link poses are stale until the next
forward pass.

Collision termination and the benchmark metric now use 3D clearance from the
11 robot links configured for the joint CBF to every selected human capsule.
Each link uses the CBF's 0.10 m sphere radius, and termination retains the 0.10 m
surface margin. This catches arm, leg, and torso encounters that the old 0.35 m
root-only planar proxy missed. Link and geom indices are resolved once and all
environment/link/capsule pairs remain batched on the device.

### Robot-relative human fix validation

The old checkpoint was rerun after the placement and clearance fixes in 256
primary-only environments with seed 17:

| Runtime | Actor LiDAR | Primary collision | Mean episode length | Link-clearance p05 | Link-clearance median |
|---|---|---:|---:|---:|---:|
| Packed training | normal | 100.00% | 70.0 steps | -0.109 m | 0.025 m |
| Packed training | blinded | 100.00% | 69.8 steps | -0.108 m | 0.031 m |
| Online/play | normal | 100.00% | 73.4 steps | -0.119 m | 0.020 m |

The packed fifth-percentile clearance fell from the broken approximately 4.19 m
to -0.109 m. Packed and online encounter timing and clearance are now close,
which validates the shared placement semantics. The 100% collision rate is also
the expected negative control: `model_16000.pt` was trained with the distant
packed human and did not learn close reactive avoidance. Normal and blinded
LiDAR remain indistinguishable, so a new training run is required.

Reproduce the paired default and forced-human runs with:

```bash
uv run python scripts/benchmark_policy_avoidance.py \
  --checkpoint logs/rsl_rl/safe_mimic_g1_live_aligned_lidar_avoidance/2026-08-29_01-46-09/model_16000.pt \
  --num-envs 1024 \
  --seed 17 \
  --scenarios default forced-human \
  --modes normal blind \
  --output artifacts/benchmarks/lidar_avoidance_model_16000_paired.json
```

The isolated moving-human diagnostic uses `--scenarios primary-only`; its raw
result is saved in
`artifacts/benchmarks/lidar_avoidance_model_16000_primary_only.json`.
Add `--human-runtime online` to evaluate the play composer; that result is saved
in `artifacts/benchmarks/lidar_avoidance_model_16000_primary_only_online.json`.
Post-fix results are saved in
`artifacts/benchmarks/lidar_avoidance_model_16000_primary_only_fixed.json` and
`artifacts/benchmarks/lidar_avoidance_model_16000_primary_only_online_fixed.json`.

## Moving-human reaction envelope

The auxiliary joint-robust checkpoint `model_21000.pt` was evaluated in 2,048
primary-human-only online/play encounters. Initial root radius was sampled from
0.75--4.0 m and the human's annotated action midpoint was scheduled to intersect
the robot's initial position 0.5--4.0 s later. The crowd, pushes, observation
noise, and visualization were disabled. Startup dynamics and human motion/size
randomization remained enabled. Collision is the configured 0.10 m
robot-link-to-human surface clearance; a timeout means the complete six-second
episode remained collision-free. `Other` denotes a normal tracking termination,
so it should not be counted as a successful avoidance episode.

Checkpoint loading restores the training `common_step_counter`. The environment
must be reset once after loading so time-based human trajectories are rescheduled
against that restored clock. The reaction-envelope and existing diagnostic
scripts now do this; results produced before this fix must not be used to infer
initial TTC.

| Scheduled intercept TTC | Episodes | Collision | Collision-free timeout | Other |
|---|---:|---:|---:|---:|
| <1.5 s | 557 | 86.7% | 5.7% | 7.5% |
| 1.5--2.0 s | 292 | 60.6% | 27.1% | 12.3% |
| 2.0--2.5 s | 307 | 55.7% | 33.2% | 11.1% |
| 2.5--3.0 s | 295 | 42.7% | 47.5% | 9.8% |
| 3.0--3.5 s | 305 | 37.4% | 52.5% | 10.2% |
| >=3.5 s | 292 | 26.7% | 60.3% | 13.0% |

The nominal average approach speed below is initial root distance divided by
the scheduled intercept time. It is more stable than a single 10 Hz finite
difference through the composed walk/action transition.

| Nominal approach speed | Episodes | Collision | Collision-free timeout | Other |
|---|---:|---:|---:|---:|
| <1.0 m/s | 1,005 | 40.7% | 46.5% | 12.8% |
| 1.0--1.5 m/s | 450 | 56.2% | 35.3% | 8.4% |
| 1.5--2.0 m/s | 198 | 63.1% | 24.2% | 12.6% |
| >=2.0 m/s | 395 | 91.6% | 3.8% | 4.6% |

Distance alone is not monotonic because a far human assigned the same intercept
time must move faster. The two-dimensional result separates them: below 0.75
m/s, collision was 16% at 2.0--2.5 m and 26% at 1.5--2.0 m, but 50% inside
1.5 m. At 2.0--3.0 m/s, collision was 81--100% across every populated distance
bin. Thus both close initialization and speed matter, but TTC captures their
interaction better than either marginal statistic.

The deterministic normal-versus-blinded policy action differed by L2 >=0.10 at
0.08 s for the 10th, 50th, and 90th percentiles: one completed interleaved LiDAR
publication. The robot first developed at least 0.25 m/s velocity away from the
human at 0.14/0.46/1.56 s (p10/p50/p90). From that first outward motion, remaining
time strongly predicted collision:

| Time remaining after first outward motion | Collision |
|---|---:|
| 0--0.5 s | 98.4% |
| 0.5--1.0 s | 84.2% |
| 1.0--1.5 s | 54.7% |
| 1.5--2.0 s | 37.9% |
| 2.0--2.5 s | 28.4% |
| >=2.5 s | 21.0% |

Therefore the principal failure is not a two-second perception delay. The
policy notices the observation at the first 10 Hz scan, but typically needs
roughly 1.5--2.5 additional seconds of physical maneuvering margin. Scheduled
TTC must exceed about 2.5 s before collision falls below 50%, and 3.5 s before
it falls below 30%. Even with long warning, the residual 21--27% collision floor
shows that maneuver quality, limb response, and motion-dependent failures still
need training rather than merely earlier detection.

Reproduce with:

```bash
uv run python scripts/benchmark_policy_reaction_envelope.py \
  --checkpoint logs/rsl_rl/safe_mimic_g1_live_aligned_lidar_avoidance_auxiliary_joint_robust/2026-08-31_00-16-59_auxiliary_joint_robust_4096/model_21000.pt \
  --motion-file artifacts/motions/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 2048 \
  --seed 31 \
  --min-radius-m 0.75 \
  --max-radius-m 4.0 \
  --min-intercept-s 0.5 \
  --max-intercept-s 4.0 \
  --episode-length-s 6.0 \
  --output artifacts/benchmarks/lidar_avoidance_auxiliary_joint_robust_model_21000_reaction_envelope_2048.json
```
