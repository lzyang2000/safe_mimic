# Expanded SEED G1 WBC-70 / dance-30 bank

## Selection

The expanded bank doubles the previous 5 GiB-original selection while retaining
its intent: broad motions from the active `wbc_mjlab` allowlist provide 70% of
the payload and accepted dance-tagged motions provide 30%. The pools are
disjoint, and every selected original is stored with its published mirrored
trajectory.

| Measurement | Value |
|---|---:|
| WBC original/mirror pairs | 11,459 |
| Dance original/mirror pairs | 3,645 |
| Total pairs | 15,104 |
| Total motions | 30,208 |
| Train pairs / motions | 13,757 / 27,514 |
| Validation pairs / motions | 1,347 / 2,694 |
| Original 50 Hz frames | 5,797,398 |
| Combined 50 Hz frames | 11,594,251 |
| Full 30-body float32 payload | 19.350 GiB |
| Active compressed NPZ bytes | 17.594 GiB |
| Realized WBC / dance payload | 70.467% / 29.533% |

The active original duration is about 32.21 hours. Original and mirror data are
both retained because actual retargeted mirrors are preferable to implementing
online joint/body reflection for the first training run.

After conversion, run the reproducible stationary-jump prune with:

```bash
uv run python scripts/prune_g1_stationary_jumps.py --apply
```

It removes both pair members when jump/hop/leap metadata coincides with at most
0.35 m pelvis XY excursion; explicit “in place” and “one place” descriptions
are always removed. The current prune excludes 751 pairs (1,502 clips), while
preserving traveling jumps. `excluded_stationary_jumps.jsonl` is the audit log;
source CSVs and cached NPZs are not deleted.

Rebuild the manifests with:

```bash
uv run python scripts/build_g1_paired_sample.py \
  --policy wbc-dance \
  --target-gib 10
```

The target size is part of the default output directory name, so this writes
`g1_wbc70_dance30_paired_10g_v1` rather than overwriting the earlier 5 GiB bank.
Convert or resume with:

```bash
uv run python scripts/convert_g1_tracker_npz.py \
  --manifest artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1/conversion_manifest.jsonl \
  --output-dir artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_10g_v1/npz_50hz \
  --workers 16
```

The completed conversion reused 24,424 NPZs through hardlinks, converted 7,286
new files, and reported zero errors. Reuse does not change the logical contents
of either bank, and all source archives and CSVs remain available.

## Packed CUDA loader

`PackedNpzMotionLib` accepts the rich JSONL conversion manifest, the generated
YAML NPZ manifest, or one NPZ. JSONL is preferred because it already contains
exact frame counts plus pool, pair, split, and description metadata. YAML files
without counts require only an NPZ-header pass before allocation.

For the complete bank the loader:

1. Computes every clip's final offset without decompressing it.
2. Allocates the six final float32 tensors once on CUDA.
3. Sequentially decompresses one clip on the CPU.
4. Selects the 14 G1 tracking bodies on the CPU while retaining all 29 joints.
5. Copies those arrays directly into the assigned final CUDA slice.

This avoids `wbc_mjlab`'s startup pattern of holding per-clip CUDA tensors and
then allocating a second concatenated bank. Continuous-time frame lookup clamps
both interpolation indexes to the selected clip, so neither linear
interpolation nor quaternion SLERP can cross an NPZ boundary.

```python
from safe_mimic.motions import PackedNpzMotionLib

tracked_body_indexes = (0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29)
library = PackedNpzMotionLib(
  "artifacts/bones-seed/datasets/"
  "g1_wbc70_dance30_paired_10g_v1/conversion_manifest.jsonl",
  tracked_body_indexes,
  device="cuda",
  splits="train",
)
```

Use `splits="train"` for optimization and construct a separate validation
library when evaluating. The all-split measurement below deliberately loads the
complete bank to establish its upper memory bound.

## Random RSI and adaptive difficulty

`AdaptiveMotionSampler` takes the useful structure from the boxing tracking
teacher: random reference-state initialization, approximately one-second
difficulty bins, a uniform floor, and more starts near failed phases. It adapts
that structure to a large separate-file library:

- a bin ends at its source clip boundary, even when shorter than one second;
- difficulty is an EMA failure rate, not a raw failure count, avoiding a
  frequently-sampled-bin feedback loop;
- 10% of probability remains on the duration-weighted random prior;
- WBC and dance probabilities are normalized independently, preserving the
  curated pool mass;
- original/mirror phases share their difficulty signal; and
- all update and sampling operations are CUDA-vectorized over environments.

```python
from safe_mimic.motions import AdaptiveMotionSampler

sampler = AdaptiveMotionSampler(library)
sample = sampler.sample(4096)
reference = library.get_frame(sample.motion_ids, sample.motion_times)

# At episode completion, before drawing replacement references:
sampler.update(failed, sample.motion_ids, sample.motion_times)
```

The adaptive tensors expose `state_dict()` and `load_state_dict()` and must be
saved with the training checkpoint. Otherwise a resumed policy forgets which
phases it has found difficult.

Reference selection complements the existing training randomizations. mjlab's
training configuration perturbs reset position/orientation, base velocity, and
joint position; applies interval pushes; and randomizes COM, encoder bias, and
foot friction. Safe Mimic additionally randomizes crowd density/shape, human
height and motion, and LiDAR timing/noise. Play mode disables the standard
tracking corruption and reset perturbations as before.

## RTX 4090 measurement

Command:

```bash
uv run python scripts/benchmark_packed_motion_lib.py \
  --batch-size 4096 \
  --warmup 20 \
  --iterations 200
```

Measured with the pre-pruning 31,710-motion bank while unrelated GPU work was
present; it remains a conservative upper bound for the active bank:

| Measurement | Result |
|---|---:|
| Load time | 113.82 s |
| Packed motion tensors | 11,500,479,360 bytes (10.711 GiB) |
| CUDA allocated after load | 11,505,864,704 bytes |
| CUDA reserved after load | 11,507,073,024 bytes |
| Adaptive bins | 254,390 |
| Adaptive state/index tensors | 15,770,772 bytes |
| Uniform sample + interpolation, 4,096 | 0.370 ms |
| Adaptive sample + interpolation, 4,096 | 0.374 ms |
| Difficulty update + adaptive sample + interpolation, 4,096 | 2.484 ms |

The 2.484 ms path intentionally updates and replaces all 4,096 episodes every
iteration. Normal training only performs that work for environments that ended,
so it should be charged at reset frequency rather than every policy step.

## Raw mimic integration

`PackedMotionCommand` now supplies the storage/command integration described
above. Every environment owns a motion ID and continuous motion time; standard
tracking observations, rewards, terminations, randomized simulator reset, and
ghost visualization consume the packed reference through the same properties
as mjlab's original `MotionCommand`. Clip completion resamples that environment
without ever indexing the following NPZ.

The first registered task intentionally has no perceptive additions:

```bash
uv run train SafeMimic-Tracking-MotionLib-Unitree-G1 \
  --env.scene.num-envs 4096
```

It loads the 27,514-motion train split and uses the upstream state-estimation-free
actor contract: pelvis linear velocity and reference-anchor position are omitted,
while pelvis angular velocity and reference-anchor orientation remain. The actor
and critic contain 154 and 286 values, respectively, and use the standard flat G1 rewards,
terminations, actions, self-collision sensor, and domain randomization. A CUDA
run initialized, reset, and trained all 4,096 environments with the pre-pruning
split; the active split contains 9.557 GiB of packed frame tensors.

The command updates adaptive statistics before replacing a completed reference:
tracking termination is failure, whereas timeout or natural clip completion is
success. The statistics expose a checkpointable `adaptive_state_dict()`, but
the generic first-cut runner does not yet persist it. Training resumes retain
the policy and optimizer but restart the difficulty EMA. The future perceptive
task can reuse this command and add unfiltered future-reference observations,
LiDAR, people, and avoidance objectives separately.
