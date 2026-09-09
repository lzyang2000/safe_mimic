# Post-mortem: why collision avoidance plateaued (2026-09-03)

Scope: four training lineages (old auxiliary_joint_robust to 21k; ttc_scratch
to 17k; coadjust to 13.8k; coadjust_fkc to 11.3k) all land at 53-69%
collisions on the human-approach protocols, against a kinematic teacher
ceiling of 10.5%. The user killed the FKC run and asked why nothing moved the
absolute number. This document is the evidence-backed answer.

## 1. The headline is flat because the pelvis never learned to escape

Every lineage, measured identically (reaction envelope, 0.5-4 s TTC, online):

| Lineage | collision | first useful outward motion p50 / p90 | root escape realized (planar compliance) |
|---|---:|---:|---:|
| old model_21000 | 54.4% (+11% falls) | 0.46 / 1.56 s | 0.51 |
| ttc_scratch @15k | 64.0% | 0.44 / 1.89 s | (n/a, same signature) |
| coadjust @10k | 59.5% | 0.50 / 1.64 s | 0.33 |
| coadjust_fkc @10k | 53.4% (+4% falls) | 0.48 / 1.63 s | 0.47 |

- "Root escape realized" = fraction of the teacher's commanded planar escape
  velocity that the robot actually produces on frames where the filter
  intervenes (>= 0.1 m/s, ~70% of frames). It is 0.33-0.51 everywhere, and
  in every lineage the robot's root velocity is CLOSER to the nominal dance
  velocity (0.50-0.56 m/s error) than to the escape velocity (0.71-0.74 m/s
  error). Interventions of ~1 m/s are realized at ~0.4-0.5 m/s, in a
  partially right direction (cosine 0.41-0.53).
- The time to first useful outward motion has not changed by more than noise
  across four architectures. Nothing we built touched pelvis behavior, by
  design (user scope: limbs only) — and the pelvis is where the collisions
  are.

## 2. Collisions are lower-body, and fixing arms only moves the contact point

Failure attribution, 512 online episodes, 10 s (`analyze_policy_failures.py`):

| | old model_21000 | coadjust_fkc @10k |
|---|---:|---:|
| collisions | 343 | 352 |
| legs (ankles + knees) | 221 (64%) | 249 (71%) |
| arms | 113 (33%) | 94 (27%) |
| torso | 9 | 9 |

Arm contacts fell 17% under the limb-compliance work; leg contacts rose 13%;
the total is unchanged. When the arm gets out of the way but the pelvis stays
on the human's path, the collision simply lands on the next link (ankles alone
account for ~205 of 352). Limb compliance is real (see 4) but cannot reduce
collisions on its own.

## 3. Perfect information does not rescue it: the constraint is execution

Online correction ablation (primary-only, historical distribution, 10 s):

| mode | old-21k | coadjust_fkc @10k |
|---|---:|---:|
| normal | 68.9% | 69.0% |
| aux-oracle (true teacher corrections fed to the policy) | 50.6% | 52.1% |
| aux-zero | 82.8% | 86.0% |
| blind | 98.4% | 97.1% |

With the privileged teacher's planar compass and limb corrections handed to
the policy at every step, it still collides 52% of the time where a robot
that simply followed that teacher would collide ~10%. The information is
present; the policy cannot convert it into a large enough, fast enough
whole-body displacement. (One real change: under oracle the FKC/co-adjust
policy no longer falls — 1.5% vs 27% for the old forced action residual.
Reference-based injection is the stable way to deliver corrections.)

## 4. What did work, and what it is worth

- FK-consistent objective (Phase 2c) removed the arm-compliance ceiling: arm
  state compliance rose from 9% of active frames near the corrected pose
  (coadjust) to 51% (FKC); prediction quality 0.85 cosine; arm-clearance
  response 71%. The old policy's higher 70% figure was forced by the 0.5-gain
  action residual and came with 11% falls — the wrong kind of compliance.
- Envelope improved 59.5 -> 53.4% between coadjust and FKC at equal
  iterations, entirely consistent with the arm share of contacts shrinking.
- The execution-cosine metric (signals 3) ranks lineages backwards for
  compliant policies and is retired; state compliance (signal 5, arms) and
  planar state compliance (signal 6, root) are the mechanism metrics.

## 5. Why the pelvis does not escape — the two candidate causes

The root's rewards are already consistent (all body-position terms track the
planar-filtered targets), so unlike the arms this is not a reward conflict.
Two explanations remain, and they call for different fixes:

A. **Observability (no closed-loop error signal).** The no-state actor
   observes no root position or root-velocity target — the escape enters
   only through LiDAR and a 2-D learned compass (direction, no magnitude, no
   progress feedback). An open-loop "human there -> lean away" response
   cannot know when it has moved enough, so it stays conservative: realize
   ~half, then hedge back toward the dance. The oracle result (perfect
   compass, still 52%) is consistent with this: direction alone is not an
   error signal.
B. **Capability (skill absence).** The tracker has learned one dance clip.
   A 1-2 m/s lateral displacement initiated from an arbitrary dance pose
   within ~0.5 s is a locomotion maneuver it has never rehearsed; rewards can
   only select among behaviors the motor repertoire contains.

The two are separable with one experiment (next section).

## 6. Recommendation

1. **Decisive experiment (one training run):** give the tracker a closed-loop
   root error signal — the 2-D body-frame displacement of the filtered
   reference anchor relative to the live robot (`motion_anchor_pos_b`-style,
   self-generated by the filter/adjuster; no odometry, zero at rest during
   nominal motion, so it does not interfere with diverse motions). Train
   from scratch with it teacher-forced. If planar compliance rises toward
   1.0 and collisions fall toward the teacher, cause A is confirmed and the
   deployable adjuster only has to fill 2 more channels it already predicts.
   If compliance stays ~0.5, cause B is confirmed and the fix moves to the
   motion data: add escape/stepping skills (e.g., filter-generated escape
   trajectories) to the tracking library.
   This is the channel the user declined earlier (motions too diverse); the
   evidence above is why it is being re-raised: a displacement offset is
   defined as deviation FROM whatever motion is playing, and is exactly the
   standard tracking observation the state-estimation variant of this
   tracker already uses across diverse motion libraries.
2. Keep the FKC objective and the co-adjust limb path — they are correct and
   stable; they will show their value once the pelvis moves.
3. Extend FK propagation to legs only after the root question is settled
   (leg residuals still fight the ankle half of `ee_body_pos`).
4. Sub-1 s TTC is physically hopeless even for the teacher (60% at <0.75 s);
   exclude it from success criteria.

## Artifacts

- Envelope: `lidar_avoidance_coadjust_fkc_model_10000_reaction_envelope_1024.json` and lineage counterparts.
- Ablation: `lidar_avoidance_coadjust_fkc_model_10000_correction_ablation_online_1024.json`.
- Failure attribution: `lidar_avoidance_{coadjust_fkc_model_10000,auxiliary_joint_robust_model_21000}_failure_analysis_512_online.json`.
- Compliance v2/v3 (signals 5, 6): `*_joint_compliance_v2_512.json`, `*_joint_compliance_v3_512.json` for the three checkpoints.
- Teacher ceiling: `reference_filter_teacher_reaction_envelope_raised_caps_2048.json`.
