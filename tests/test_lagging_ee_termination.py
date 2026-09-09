"""End-effector height termination whose bound widens while the reference moves."""

from types import SimpleNamespace

import torch

from safe_mimic.tasks import mdp
from safe_mimic.tasks.kinematic_replay_command import PlanarFilteredReplayMotionCommand

BODIES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def _env(*, z_error, ref_vz):
  """One env per row; ``z_error`` and ``ref_vz`` are (env, body) lists."""
  z_error = torch.tensor(z_error, dtype=torch.float32)
  ref_vz = torch.tensor(ref_vz, dtype=torch.float32)
  n = z_error.shape[0]
  robot_pos = torch.zeros(n, 4, 3)
  ref_pos = robot_pos.clone()
  ref_pos[..., 2] = z_error
  ref_vel = torch.zeros(n, 4, 3)
  ref_vel[..., 2] = ref_vz
  command = PlanarFilteredReplayMotionCommand.__new__(PlanarFilteredReplayMotionCommand)
  command.cfg = SimpleNamespace(body_names=BODIES)
  cls = type(
    "Stub",
    (PlanarFilteredReplayMotionCommand,),
    {
      "body_pos_relative_w": property(lambda self: ref_pos),
      "robot_body_pos_w": property(lambda self: robot_pos),
      "body_lin_vel_w": property(lambda self: ref_vel),
    },
  )
  command.__class__ = cls
  return SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command), num_envs=n
  )


def _call(env):
  return mdp.bad_motion_body_pos_z_only_lag_aware(
    env,
    command_name="motion",
    threshold=0.25,
    lag_time_s=0.2,
    max_threshold=0.6,
    body_names=BODIES,
  )


def test_still_reference_keeps_the_strict_bound() -> None:
  env = _env(z_error=[[0, 0, 0.3, 0], [0, 0, 0.2, 0]], ref_vz=[[0] * 4] * 2)
  assert _call(env).tolist() == [True, False]


def test_fast_moving_reference_widens_the_bound_by_lag_time_times_speed() -> None:
  # Reference wrist rising at 1.0 m/s: bound = 0.25 + 0.2 * 1.0 = 0.45 m.
  env = _env(
    z_error=[[0, 0, 0.40, 0], [0, 0, 0.50, 0]],
    ref_vz=[[0, 0, 1.0, 0], [0, 0, 1.0, 0]],
  )
  assert _call(env).tolist() == [False, True]


def test_widening_is_per_body_and_uses_the_speed_magnitude() -> None:
  # Left wrist target falling fast (-1.5 m/s) -> bound 0.55; right wrist still.
  env = _env(z_error=[[0, 0, 0.5, 0.3]], ref_vz=[[0, 0, -1.5, 0]])
  assert _call(env).tolist() == [True]  # right wrist 0.3 > 0.25 still trips
  env = _env(z_error=[[0, 0, 0.5, 0.2]], ref_vz=[[0, 0, -1.5, 0]])
  assert _call(env).tolist() == [False]


def test_bound_is_capped() -> None:
  # 5 m/s would allow 1.25 m; capped at 0.6 m.
  env = _env(z_error=[[0, 0, 0.7, 0]], ref_vz=[[0, 0, 5.0, 0]])
  assert _call(env).tolist() == [True]
  env = _env(z_error=[[0, 0, 0.59, 0]], ref_vz=[[0, 0, 5.0, 0]])
  assert _call(env).tolist() == [False]
