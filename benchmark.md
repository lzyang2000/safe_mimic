# LiDAR stepping benchmark

Measured on an NVIDIA GeForce RTX 4090 with 4,096 MJLab environments. The
environment ran headless with the same scene and physics configuration in every
case. The benchmark measures environment stepping without policy inference.

## Unified 4,096-environment relative comparison

All human-bearing rows below use the quarter-density 180 x 6 LiDAR, either at
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
quarter-density 180 x 6 LiDAR at 10 Hz. Both runs used 4,096 environments, 20
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
environments, the quarter-density 180 x 6 LiDAR, 20 warmup steps, three trials of
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
  --resolution quarter \
  --motion-file /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
  --num-envs 4096 \
  --warmup-steps 20 \
  --steps-per-trial 40 \
  --trials 3
```
