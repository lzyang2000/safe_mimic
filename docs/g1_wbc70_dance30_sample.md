# SEED G1 WBC-70 / dance-30 paired sample

## Result

This bank uses the active `wbc_mjlab` filtered SEED file list for 70% of its
original-motion payload and Safe Mimic's accepted dances for the remaining 30%.
The pools are disjoint: the 145 accepted dances that also occur in the WBC list
belong only to the dance pool. Every selected original is paired with its actual
published `_M` trajectory.

| Measurement | Value |
|---|---:|
| WBC original/mirror pairs | 6,014 |
| Dance original/mirror pairs | 1,968 |
| Total pairs | 7,982 |
| Total tracker motions | 15,964 |
| Original float32 payload | 5.001 GiB |
| Original + mirror float32 payload | 10.002 GiB |
| Compressed NPZ size | 9.096 GiB |
| Approximate paired duration | 33.30 h |
| Training pairs | 7,274 |
| Validation pairs | 708 |

The realized original-payload mixture is 70.018% WBC and 29.982% dance. Both
pools are independently stratified by package and category, using stable hashed
prefixes against proportional payload quotas. The WBC selection's largest
package payload-share error is 0.0213 percentage points and its largest category
error is 0.1016 percentage points. The corresponding dance-pool error is 0.0152
percentage points.

## Combined distribution

The 30% dance pool is based on the broad accepted dance tag, not only the
top-level `Dances` package. Consequently, package-level `Dances` occupies
16.46% of combined payload even though dance-tagged motions occupy 29.98%.

| Package | Pairs | Payload share |
|---|---:|---:|
| Locomotion | 3,562 | 47.94% |
| Dances | 1,402 | 16.46% |
| Communication | 1,265 | 11.57% |
| Interactions | 808 | 9.75% |
| Gaming | 374 | 5.49% |
| Everyday | 279 | 5.00% |
| Sport | 192 | 2.24% |
| Other | 100 | 1.57% |

## Filter comparison

All dance-pool motions already pass Safe Mimic's complete upright and kinematic
filter. The active WBC list is broader: 7,660 of the complete 7,982-pair bank
(95.97%) also occur in Safe Mimic's accepted bank, while 322 pairs do not.

At metadata level, 7,744 pairs pass and 238 have an explicitly non-upright tag:

| Metadata rejection | Pairs |
|---|---:|
| Injured | 111 |
| Sitting | 63 |
| On all fours | 35 |
| Kneeling | 27 |
| Lying | 2 |

The remaining 84-pair gap between the metadata pass and complete acceptance is
from kinematic limits. The manifests retain `sampling_pool`, `pair_role`, and all
metadata needed to exclude or curriculum-gate these 322 pairs without rebuilding
the bank.

## Files and reproduction

The generated bank is under
`artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_5g_v1/`:

- `originals.jsonl` and `mirrors.jsonl`: separately labeled selections.
- `conversion_manifest.jsonl`: interleaved pairs and pool labels.
- `selection_summary.json`: quotas and complete package/category distributions.
- `npz_manifest_50hz.yaml`: all 15,964 tracker motions.
- `npz_conversion_summary.json` and `npz_conversion_errors.jsonl`: materialized
  conversion result.

Rebuild the deterministic selection with:

```bash
uv run python scripts/build_g1_paired_sample.py --policy wbc-dance
```

Materialize or resume its NPZ files with:

```bash
uv run python scripts/convert_g1_tracker_npz.py \
  --manifest artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_5g_v1/conversion_manifest.jsonl \
  --output-dir artifacts/bones-seed/datasets/g1_wbc70_dance30_paired_5g_v1/npz_50hz \
  --workers 16
```

The completed bank has zero conversion errors and no partial files. The manifest
and directory contain exactly the same 15,964 unique paths. Sampled motions from
both pools and both pair roles pass direct mjlab `MotionLoader` validation.
