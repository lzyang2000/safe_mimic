# Safe Mimic: internalized collision avoidance plan

Decision date: 2026-08-26. Target stack: `mjlab==1.6.0`, Unitree G1,
reference-conditioned motion tracking, and body-mounted 360-degree 3D LiDAR.

## Objective

Train one policy that tracks a specific reference motion while directly producing
the minimum whole-body deviation needed to avoid obstacles. The policy must preserve
the reference's local pose, velocities, rhythm, and phase; global root position and
yaw are negotiable when clearance requires them to change.

The deployed control path is:

```text
reference window + phase + proprioception + LiDAR history
                              |
                              v
               reference-conditioned policy
                              |
                              v
                        joint targets
```

Obstacle avoidance is internalized in this policy. It is not performed by a route
planner, reference-root warper, diffusion replanner, or required runtime CBF filter.
A separate hardware emergency layer may remain as defense in depth, but it is not the
learned avoidance mechanism and is not active in the primary evaluation.

## Research hypothesis

A tracking policy can learn anticipatory, minimal-deviation collision avoidance when:

1. the actor observes both the motion reference and deployment-realistic obstacle
   sensing;
2. privileged simulation geometry supplies a dense per-link barrier signal;
3. a training-only safety filter corrects unsafe exploration; and
4. the policy is penalized for both barrier violations and reliance on that filter.

This combines CBF-RL's dual training recipe with PAC-MAN's per-link safety objective,
then replaces PAC-MAN's generic adversarial motion prior with exact, phase-aligned
motion tracking.

## Architectural decisions

### One obstacle-aware tracking actor

Extend mjlab's original G1 tracking actor rather than placing a second avoidance
policy above it. The policy outputs joint-position targets at the normal tracking
frequency. It receives the existing tracking observations plus a temporally stacked
LiDAR representation.

The future reference window remains important: it tells the policy which links the
reference is about to move toward an obstacle, allowing avoidance to begin before the
current pose becomes unsafe. This is reference conditioning, not replanning.

### Asymmetric actor-critic

The actor receives only signals available on hardware:

- reference phase and the normal mjlab tracking reference terms;
- joint positions and velocities, projected gravity, base angular velocity, and
  previous action;
- noisy, delayed LiDAR history in the robot frame; and
- optionally, sensor-derived body-relative clearance features computed from the
  local LiDAR field.

The critic and reward functions may additionally receive:

- exact obstacle geometry and velocities;
- exact robot link poses, velocities, and collision shapes;
- exact per-link barrier values; and
- the difference between proposed and training-filtered actions.

Privileged quantities must never enter the deployed actor observation.

### 3D 360-degree LiDAR

Use the existing torso-mounted spherical ray-cast LiDAR scaffold. Do not reduce the
problem to a single horizontal scan: arms, legs, torso, and head occupy different
heights, and the reference motion may turn away from a forward-facing sensor.

The initial actor can consume the ordered range image directly. A later encoder may
turn a short scan history into an egocentric occupancy or signed-distance field
(SDF). Global SLAM is not required for the initial reactive task.

Training corruptions should cover range noise, ray and sector dropout, latency, held
frames, extrinsic error, self-occlusion, and scan-rate mismatch. Moving-obstacle
experiments additionally require temporal observations from which relative velocity
is observable.

## Whole-body barrier reward

The current root-only planar proximity penalty is insufficient: it can report a safe
pelvis while an arm, leg, torso, or head collides. Replace it with barriers over all
relevant robot collision links.

For PAC-MAN's sphere approximation, the clearance for link `i` and obstacle `j` is:

```text
h_ij = ||p_obstacle,j - p_link,i||
       - (radius_obstacle,j + radius_link,i + margin)
```

For general boxes, cylinders, walls, and clutter, use the obstacle SDF:

```text
h_ik = d_sdf(p_ik) - radius_ik - margin
```

`p_ik` is a sphere sample on link `i`. Multiple spheres should approximate long links
and collision capsules; one center point per link is not accurate enough for dancing
limbs.

The closing-rate barrier is:

```text
g_ik = h_dot_ik + alpha * h_ik
h_dot_ik = grad(d_sdf(p_ik)) dot (v_link,ik - v_obstacle)
```

The first Link-CBF reward should reproduce PAC-MAN's most-binding formulation:

```text
r_link_cbf = min_ik clip(g_ik, -c, 0)
```

Initial reference parameters from PAC-MAN are `alpha = 1`, `c = 2`, and reward weight
`0.27`; they are starting points, not assumed final values. Compare the hard minimum
against a differentiable soft minimum or mean of the worst `k` links if training is
too noisy.

Gate the reward by relevance. For static obstacles, apply it when a link is within an
alert distance or its reference/current velocity is closing on an obstacle. This
prevents distant geometry from encouraging the robot to abandon the dance or freeze.

## Training-only CBF guidance

Follow CBF-RL's dual method during simulation:

1. The policy proposes an action `u_policy`.
2. A CBF projection produces the minimally changed safe action `u_safe`.
3. The simulator executes `u_safe` during the guided-training stage.
4. The reward penalizes the unsafe proposal and the filter intervention magnitude.
5. Evaluation and deployment execute `u_policy` directly, with no CBF projection.

For one linearized constraint, CBF-RL uses the closed-form projection:

```text
u_safe = u_policy                                      if a^T u_policy >= b
u_safe = u_policy + ((b - a^T u_policy) / ||a||^2) a otherwise
```

Its dense internalization reward combines direct violation and correction distance:

```text
r_correction = min(a^T u_policy - b, 0)
               + exp(-||u_policy - u_safe||^2 / sigma^2) - 1
```

The whole-body problem has many simultaneous link constraints. Implement Link-CBF
reward-only training first. Then add a vectorized training-only projection over the
most active constraints and verify its throughput before enabling it in all parallel
environments. The projection must operate in a representation consistent with the
policy output and robot kinematics, such as desired joint velocity derived from the
joint-position target. Do not silently project only the planar base velocity and call
that whole-body safety.

The required ablation is:

- nominal tracking plus collision penalty;
- Link-CBF reward only;
- training filter only;
- dual Link-CBF/correction reward plus training filter; and
- dual policy with a runtime filter, reported only as a privileged upper bound.

The primary system is the dual-trained policy evaluated without a runtime filter.

## Reward composition

Use the original tracking task as the nominal objective:

```text
r_total = r_mimic
          + lambda_link * r_link_cbf
          + lambda_correction * r_correction
          + r_collision
          + r_regularization
```

Preserve strongly:

- local per-body position and orientation relative to the reference anchor;
- per-body linear and angular velocities;
- joint configuration and motion phase;
- action smoothness and physical regularization; and
- balance and valid contacts.

Relax selectively:

- global root position;
- global root yaw; and
- individual links only when their barrier becomes active.

Keep enough global-root reward to discourage gratuitous wandering and to draw the
robot back toward the nominal trajectory after an obstacle clears. Do not introduce
a separate replanned root path.

Collision termination remains useful, but it is a sparse final failure signal rather
than the main avoidance reward. Count contact on any protected link as a collision.

## Implementation phases

### Phase 0: protect the nominal tracking baseline

- [ ] Obtain the actual reference motion and checkpoint artifacts.
- [ ] Record obstacle-free tracking metrics for the upstream mjlab task.
- [ ] Add a deterministic evaluation seed set and checkpoint metadata.
- [ ] Verify which upstream reference and reward terms are local versus global.

Exit gate: the baseline dance is reproducible and its local/global tracking errors are
recorded before avoidance changes begin.

### Phase 1: privileged per-link geometry

- [ ] Define protected G1 links and sphere/capsule samples for each link.
- [ ] Compute exact link-obstacle clearance, normals, relative velocity, `h`, `h_dot`,
  and `g` in batched Torch operations.
- [ ] Add the Link-CBF reward and any-link collision metric.
- [ ] Replace the existing pelvis-only `obstacle_proximity_penalty` in the primary
  task; retain it only as an explicit baseline.
- [ ] Add unit tests for clearance sign, closing-rate sign, margins, aggregation,
  moving obstacles, and gradients/batch shapes.

Exit gate: scripted motions receive a penalty before contact at the correct link, and
unthreatened links produce zero penalty.

### Phase 2: end-to-end perceptive tracking policy

- [ ] Train with privileged per-link rewards while the actor receives the existing
  noisy LiDAR history.
- [ ] Start from obstacle-free or distant-obstacle scenes and progressively move
  obstacles into the reference swept volume.
- [ ] Mix obstacle-free episodes so the policy cannot improve reward by constantly
  deviating from the reference.
- [ ] Randomize obstacle azimuth, height, size, shape, and multiplicity.
- [ ] Track per-link intervention and mimic errors during training.

Exit gate: without a runtime filter, the policy improves any-link collision rate over
the nominal tracker while retaining an agreed fraction of baseline reference fidelity.

### Phase 3: dual CBF-RL training

- [ ] Implement a vectorized training-only action projection.
- [ ] Add the correction-distance and raw barrier-violation rewards.
- [ ] Ensure the actor can observe every threat property needed to reproduce the
  filter through LiDAR history; keep exact geometry critic-only.
- [ ] Anneal or probabilistically remove the training filter so the policy cannot
  depend on corrected rollout dynamics.
- [ ] Run reward-only, filter-only, and dual ablations.

Exit gate: the dual policy retains its avoidance success when the training filter is
removed, unlike the filter-only baseline.

### Phase 4: richer obstacles and sim-to-real sensing

- [ ] Add narrow passages, overhead obstacles, low obstacles, and clutter that
  threaten different body links.
- [ ] Add moving obstacles with randomized velocities and intermittent appearance.
- [ ] Replace the provisional beam table with the selected physical LiDAR pattern.
- [ ] Match scan aggregation, latency, minimum range, extrinsics, and failure modes.
- [ ] Evaluate recovery toward the original reference after threats pass.

Exit gate: avoidance works across obstacle types and sensor corruption without
systematic freezing, fleeing, or sacrificing the reference when no threat exists.

## Evaluation

Report safety and imitation together rather than reducing the result to one success
rate.

Safety metrics:

- any-link collision rate and collision count;
- minimum whole-body clearance and minimum barrier value;
- time spent with `g < 0`;
- training-filter intervention rate and correction magnitude;
- fall rate; and
- success under static, moving, occluded, and rear-approach obstacles.

Reference-fidelity metrics:

- local MPJPE or per-body position error;
- local body orientation error;
- joint and velocity tracking error;
- phase error;
- global root translation and yaw deviation;
- time to return toward the nominal root trajectory; and
- obstacle-free degradation relative to the upstream tracker.

Behavior metrics:

- reaction time before predicted collision;
- unnecessary motion when no threat is present;
- freeze/flee rate;
- clearance efficiency; and
- smoothness of departure from and recovery toward the reference.

Plot a safety-versus-fidelity Pareto curve over barrier and tracking weights. A policy
that never collides because it stops dancing is a failure, as is a perfect tracker that
clips obstacles.

## Required ablations

- pelvis-only proximity versus per-link Link-CBF;
- LiDAR actor versus privileged obstacle-state actor;
- single scan versus temporal LiDAR history;
- 2D horizontal scan versus 3D vertical coverage;
- exact tracking rewards versus a generic AMP-style prior;
- strong versus soft global-root tracking;
- reward only versus training filter only versus dual CBF-RL;
- dual policy with and without the runtime filter; and
- clean sensing versus deployment-level corruption.

## Known risks

- **Insufficient observability:** PAC-MAN shows that stronger CBF structure can hurt
  when the actor cannot infer the threat. Barrier sophistication must match LiDAR
  coverage, history, and resolution.
- **Freeze or flee behavior:** use relevance gating, obstacle-free episodes, saturated
  penalties, and recovery/global-root rewards.
- **Tracking dominates safety:** curriculum the obstacle difficulty and increase
  per-link/correction rewards only after obstacle-free tracking is stable.
- **Safety dominates tracking:** retain phase and local body rewards and measure
  obstacle-free degradation continuously.
- **Projection dependence:** anneal the training filter and always evaluate the raw
  policy.
- **Incorrect link geometry:** use several sphere samples or capsules per long link
  and validate against MuJoCo contacts.
- **Parallel-training cost:** benchmark vectorized link distance and projection code
  before expanding obstacle count or link samples.

## Immediate code changes

1. Add protected-link geometry and batched per-link barrier terms to
   `src/safe_mimic/tasks/mdp.py`.
2. Configure Link-CBF rewards and critic-only barrier observations in
   `src/safe_mimic/tasks/env_cfg.py`.
3. Add geometry/reward unit tests before training.
4. Preserve the current pelvis-only penalty as a named baseline rather than the
   primary task objective.
5. Add the training-only action projection only after the Link-CBF reward path is
   correct and profiled.

## Literature and code anchors

- [CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier
  Functions](https://arxiv.org/html/2510.14959): dual training-time filtering and
  barrier/correction rewards; deployment without a runtime filter.
- [PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid
  Dodgeball](https://arxiv.org/html/2607.28623v1): deployment-realistic perception,
  privileged per-link barrier guidance, and any-link evaluation.
- [Constrained Whole-Body Tracking for Humanoid Robots
  (ConstrainedMimic)](https://arxiv.org/abs/2606.00374): post-hoc constrained tracking
  baseline and privileged runtime-filter upper bound, not the primary architecture.
- [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab): the repository currently linked
  by PAC-MAN. At the time of inspection, its `main` branch contains the base AMP
  locomotion/recovery task but not the PAC-MAN dodgeball task; the paper states that
  its benchmark and training pipeline will be released.
