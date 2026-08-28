# Local policy artifacts

Place local motion and checkpoint files here when they are available, for example:

- `motion.npz` — the reference motion used by the tracking command
- `policy.pt` — a checkpoint trained for `Mjlab-Tracking-Flat-Unitree-G1`
- `lidar_policy.pt` — a checkpoint trained for the obstacle-aware Safe Mimic task

Large artifacts are intentionally ignored by Git. A flat-tracking checkpoint is not
input-compatible with the LiDAR task: use it with the upstream task, or use it as a
teacher while training a new perceptive student.
