# SEED G1 paired 10 GiB sample

## Result

The deterministic sample contains 8,176 non-mirrored SEED G1 motions and their
8,176 actual `_M` partners. It targets the in-memory float32 mjlab tracker
payload, not compressed file size.

| Measurement | Originals | Mirrors | Combined |
|---|---:|---:|---:|
| Motions | 8,176 | 8,176 | 16,352 |
| Float32 tracker payload | 4.998 GiB | 4.997 GiB | 9.995 GiB |
| Duration at 50 Hz | 16.64 h | 16.64 h | 33.27 h |
| Compressed NPZ size | — | — | 9.065 GiB |

The original and mirror payloads differ slightly because a small number of
published pairs have different metadata frame counts. All 16,352 NPZs converted
successfully. The manifest contains exactly one file per entry and no temporary
or failed outputs. Seven motions sampled across the bank also passed direct
mjlab `MotionLoader` validation.

The sample uses 5 GiB (`5 * 1024**3` bytes) rather than 5 decimal GB. Of the
compressed bank, 6,260 selected originals are hardlinks to the full converted
bank. This avoids duplicating their disk blocks; converting the remaining 1,916
originals and all mirrors added 6,070,923,608 bytes to disk.

## Selection

The source population is the 71,088 non-mirrored motions that have an actual
raw `_M.csv` partner. Another 44 originals without a published mirror are left
out. Package/category strata receive byte quotas proportional to their share of
the complete source payload. A stable SHA-256 ordering within every stratum
selects the closest prefix to its quota. The fixed seed is `20260827`.

This targets duration/payload distribution rather than equal clip counts. The
largest package payload-share error relative to the complete paired source is
0.0123 percentage points; the largest category error is 0.0091 percentage
points.

| Package | Pairs | Sample payload | Source payload | Difference |
|---|---:|---:|---:|---:|
| Locomotion | 4,284 | 52.20% | 52.20% | +0.007 pp |
| Communication | 1,268 | 11.50% | 11.50% | -0.003 pp |
| Interactions | 826 | 10.74% | 10.73% | +0.011 pp |
| Dances | 628 | 7.57% | 7.57% | +0.005 pp |
| Gaming | 493 | 7.51% | 7.50% | +0.002 pp |
| Everyday | 338 | 6.00% | 6.02% | -0.012 pp |
| Sport | 225 | 2.67% | 2.68% | -0.007 pp |
| Other | 114 | 1.80% | 1.80% | -0.003 pp |

The broad dance tag also catches dance-like motions outside the `Dances`
package: 843 pairs, or 10.31% of the sample, are dance-tagged. The split has
7,363 training pairs and 813 validation pairs, assigned deterministically by
source take. It spans 504 actors. A total of 5,803 pairs (70.98%) also occur in
the active WBC filtered list.

## Unfiltered-content warning

Because this sample deliberately represents the unfiltered distribution, it is
not an upright-only tracking library. Only 6,260 pairs (76.56%) occur in Safe
Mimic's fully accepted kinematic bank. The metadata-only screen gives the
following breakdown:

| Metadata outcome | Pairs | Share |
|---|---:|---:|
| Upright metadata pass | 6,774 | 82.85% |
| Injured | 630 | 7.71% |
| Sitting | 352 | 4.31% |
| Crawl | 114 | 1.39% |
| Kneel | 88 | 1.08% |
| On all fours | 88 | 1.08% |
| Stunt category | 48 | 0.59% |
| Crutch | 54 | 0.66% |
| Lying | 11 | 0.13% |
| Cartwheel | 9 | 0.11% |
| Handstand | 8 | 0.10% |

The 514-pair difference between the metadata pass and the 6,260 fully accepted
motions consists of kinematic-filter failures. Training the nominal upright
tracker on the entire paired sample without motion-aware initialization or
recovery handling would therefore mix incompatible objectives. The manifests
keep the metadata and `pair_role` fields so this portion can be excluded or
curriculum-gated without resampling.

## Reproduce

Build the deterministic pair selection:

```bash
uv run python scripts/build_g1_paired_sample.py
```

The output directory is
`artifacts/bones-seed/datasets/g1_unfiltered_paired_5g_v1/`. Its key files are:

- `originals.jsonl` and `mirrors.jsonl`: separately labeled halves.
- `conversion_manifest.jsonl`: interleaved original/mirror pairs.
- `selection_summary.json`: complete source and sample distributions.
- `npz_manifest_50hz.yaml`: the materialized mjlab tracker bank.
- `npz_conversion_summary.json` and `npz_conversion_errors.jsonl`: conversion
  result.

Materialize or resume missing NPZs with:

```bash
uv run python scripts/convert_g1_tracker_npz.py \
  --manifest artifacts/bones-seed/datasets/g1_unfiltered_paired_5g_v1/conversion_manifest.jsonl \
  --output-dir artifacts/bones-seed/datasets/g1_unfiltered_paired_5g_v1/npz_50hz \
  --workers 16
```
