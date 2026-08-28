# Sensing and control decision: obstacle-aware mimic

Decision date: 2026-08-26. Target package: `mjlab==1.6.0`.

## Desired behavior

The robot should keep performing a reference motion while treating its global root
trajectory as negotiable. Near an obstacle it may sidestep, turn, retreat, or dance
around it; after clearance opens, it should drift back toward the reference path.

This fits mjlab's tracking task better than it may first appear. The strong body-pose
rewards are evaluated relative to the robot's current anchor transform, so they retain
the motion's joint pose, body shape, and rhythm even when the root translates or yaws.
Only the absolute anchor position/orientation rewards need to become soft.

## Recommendation

Use a single body-mounted 360-degree **3D** LiDAR as the first obstacle sensor. Feed
the policy an ordered range image rather than an unordered point cloud. The simulated
baseline uses six downward elevations and 180 azimuth samples: 1,080 ranges shaped
logically as `[batch, 6, 180]`. The 2-degree azimuth spacing is approximately
17.5 cm at 5 m.

For obstacle avoidance this is a cleaner first choice than the four directional depth
cameras used by RPL:

- there is no front/back/left/right seam or hard camera handoff;
- coverage remains available when the reference motion turns unexpectedly;
- geometry is largely invariant to lighting and texture;
- the same sensor can later serve mapping and non-policy safety monitors.

A single horizontal 2D scanner is workable for tall walls but misses low obstacles
and can be occluded by the moving body. Use a real 3D unit with vertical coverage and
replace the provisional simulated beam table with its calibrated beams.

Depth becomes attractive if the task later needs semantics, object identity, or fine
near-field shape. A hybrid is reasonable then, but four depth cameras are not needed
for this obstacle-only baseline.

## Does mjlab support it?

Yes at the ray-casting layer. `RayCastSensorCfg` provides batched GPU ray casting,
body/site/geom attachment, body-relative alignment, geom filtering, ranges, hit
positions, and normals. It casts against the current MuJoCo-Warp scene, including
moving geometry.

mjlab 1.6.0 does not ship a native spherical/spinning pattern. Its `RingPatternCfg`
places ray origins in a ring and shoots them in one direction; it is a terrain-height
sensor, not a 360-degree LiDAR. This project adds `SphericalLidarPatternCfg`, which
implements the small `generate_rays()` pattern protocol consumed by
`RayCastSensorCfg`.

## Implemented task

`SafeMimic-Tracking-Obstacles-Unitree-G1-Lidar` starts from mjlab's original
`Mjlab-Tracking-Flat-Unitree-G1` configuration and adds:

- one 18-capsule moving human per environment, driven by a CUDA-resident SOMA
  skeleton motion bank;
- transition-matched `walk -> punch/kick -> walk` composition with skeleton-space
  inertialization; the annotated punch/kick midpoint, rather than either walking
  connector, is aligned with a future point on the robot reference path;
- a six-ring, 360-degree ray-cast LiDAR mounted on `torso_link`;
- Gaussian range noise, individual-ray dropout, random 90-degree blind sectors,
  20-60 ms latency, and two-frame range history;
- exact human capsule vectors for the critic only (asymmetric actor-critic);
- privileged clearance reward, hard contact penalty, and collision termination;
- softer global root position/orientation tracking while preserving the original
  relative body-pose, velocity, action, and joint-limit objectives.

The human is represented by 18 independently driven mocap bodies. Its transition-
approved 24-joint float16 bank occupies 257 MiB and is copied to VRAM once at
startup. Composition, inertialization, forward kinematics, domain randomization,
and capsule fitting stay on-device; raw BVHs and meshes are preprocessing inputs
and are never touched by the training loop.

## Reward behavior

The task deliberately creates three priorities:

1. Avoid collision and maintain a 0.1 m nominal surface clearance.
2. Preserve the reference motion's relative pose and rhythm.
3. Match the reference's global root path softly, which pulls the robot back only
   when it is safe.

If training yields "freeze near everything," increase progress/global-anchor reward
or curriculum the scheduled crossing time/clearance. If it clips the human to
preserve the dance, raise proximity/collision costs or initially offset crossings
into near misses. Those are behavior tradeoffs, not sensor problems.

## Policy architecture and checkpoint compatibility

Adding 1,080 ranges plus history changes the actor's first-layer input. The original
checkpoint cannot be loaded unchanged into the new task. Recommended training is
teacher-student distillation:

1. Run the existing tracking policy in obstacle-free scenes as the style teacher.
2. Train the LiDAR student on both imitation targets and obstacle rewards.
3. Initialize/copy compatible proprioceptive layers where possible, with new LiDAR
   input weights zero-initialized, or add an explicit residual avoidance branch.
4. Fine-tune with PPO once the student reproduces the reference motion.

If the existing tracker should remain frozen, the cleanest production architecture
is hierarchical: encode LiDAR in a slower avoidance policy, output a bounded planar
`(x, y, yaw)` offset or velocity, and use that to warp the reference root frame seen
by the nominal mimic policy. The tracker then keeps the original observation/action
contract and continues producing the dance, while the high-level controller moves
the dance around hazards. A joint-action residual is the second choice; it is easier
to attach but more likely to fight the tracker or disturb balance.

The scaffold currently registers an end-to-end student task because stock
mjlab/RSL-RL can train it immediately. A frozen-tracker reference-warp command or
residual runner requires the actual checkpoint and its state-dict/input contract,
neither of which is present in this repository yet. In either architecture, keep an
independent emergency-stop/contact layer outside the learned policy on hardware.

## Simulation contract

- Mount: `mid360_site`, injected under `torso_link` from the deployment URDF at
  position `(0.0002835, 0.00003, 0.41618)` m and pitch `0.0401426` rad.
- Training pattern: 180 azimuths x elevations `[0, -10, -20, -30, -40,
  -50]` degrees (1,080 rays per 50 Hz step).
- Nominal debug-viewer pattern: 185 azimuths x 27 evenly spaced elevations
  from `0` through `-52` degrees (4,995 rays), one quarter of the Mid-360's
  typical first-return density per 10 Hz frame.  Angular spacing is nearly square:
  `1.946` degrees azimuth by `2.000` degrees elevation, corresponding to about
  17 cm spacing at 5 m on a perpendicular surface.
- Valid range: 0.3-5 m; nearer self/near-field returns and farther returns are
  discarded, and misses encode as normalized range `1.0`.
- Visible groups: human/ground group 0 and G1 collision-proxy group 3.  The
  torso, arms, and legs therefore self-occlude the scan.  The one exception is
  mjlab's coarse head collision sphere: it contains the optical origin, so it is
  moved to ray-invisible group 5 while retaining its normal contact behavior.
  Two orange, contact-free pillars approximate the left and right sides of the
  head.  Relative to the optical origin, their centers are `(0.04, +/-0.035,
  -0.0375)` m and their full XYZ size is `(0.022, 0.013, 0.075)` m.  Mirrored
  roll angles of `-15` and `+15` degrees make their lower ends lean inward; a
  shared `-5` degree Y-axis pitch makes both lower ends lean forward.
- Timing: 50 Hz control and 10 Hz LiDAR publication.  Each 100 ms scan is built
  from five interleaved 999-ray phases, one per policy step, without deskew.  The
  completed 4,995-ray scan is then held unchanged for five policy steps while the
  next scan is collected.  The lower-density training pattern uses the same
  five-phase schedule (216 rays per step, 1,080 rays per published scan), followed
  by 1-3 steps of randomized observation latency and two-frame history.

Before hardware transfer, match the exact beam table, scan rate, range limits,
minimum range, mount transform, timing, and packet/frame aggregation. Add extrinsic
jitter, correlated bias, intensity/material dropout, motion distortion, and held
frames at the real scan rate.

## Sources

- [mjlab repository](https://github.com/mujocolab/mjlab)
- [mjlab RayCast Sensor documentation](https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/raycast_sensor.html)
- [mjlab RGB-D Camera documentation](https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/rgbd_camera.html)
- [RPL project page](https://rpl-humanoid.github.io/)
- [RPL paper](https://arxiv.org/abs/2602.03002)
