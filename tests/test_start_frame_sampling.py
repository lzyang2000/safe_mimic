"""Start-frame sampling for the kinematic replay commands."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from safe_mimic.tasks.env_cfg import (
  unitree_g1_kinematic_reference_lidar_demo_env_cfg,
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
  unitree_g1_lidar_avoidance_tracking_env_cfg,
  unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg,
  unitree_g1_reference_filter_lidar_demo_env_cfg,
  unitree_g1_reference_filter_policy_lidar_demo_env_cfg,
  unitree_g1_sparse_lidar_avoidance_tracking_env_cfg,
)
from safe_mimic.tasks.kinematic_replay_command import (
  KinematicReplayMotionCommand,
  PlanarFilteredReplayMotionCommand,
)

_TOTAL = 500
_BODIES = 3
_JOINTS = 4


def _sampler_stub(mode: str, *, num_envs: int = 64) -> KinematicReplayMotionCommand:
  command = object.__new__(KinematicReplayMotionCommand)
  command.cfg = SimpleNamespace(sampling_mode=mode)
  command.time_steps = torch.full((num_envs,), 7, dtype=torch.long)
  command.motion = SimpleNamespace(time_step_total=_TOTAL)
  command._env = SimpleNamespace(device="cpu")
  command.bin_count = 10
  command.metrics = {
    name: torch.zeros(num_envs)
    for name in ("sampling_entropy", "sampling_top1_prob", "sampling_top1_bin")
  }
  return command


def test_start_mode_zeroes_only_selected_envs() -> None:
  command = _sampler_stub("start")
  env_ids = torch.tensor([0, 5, 63])
  command._sample_start_time_steps(env_ids)
  assert torch.equal(command.time_steps[env_ids], torch.zeros(3, dtype=torch.long))
  untouched = torch.ones(64, dtype=torch.bool)
  untouched[env_ids] = False
  assert torch.all(command.time_steps[untouched] == 7)


def test_uniform_mode_samples_spread_frames_in_range() -> None:
  torch.manual_seed(3)
  command = _sampler_stub("uniform")
  env_ids = torch.arange(64)
  command._sample_start_time_steps(env_ids)
  steps = command.time_steps
  assert torch.all(steps >= 0) and torch.all(steps < _TOTAL)
  assert steps.unique().numel() > 8
  assert command.metrics["sampling_entropy"][0] == 1.0


def test_adaptive_mode_is_refused_explicitly() -> None:
  command = _sampler_stub("adaptive")
  with pytest.raises(NotImplementedError, match="adaptive"):
    command._sample_start_time_steps(torch.arange(4))


def _motion_tensors():
  generator = torch.Generator().manual_seed(11)
  frames = torch.arange(_TOTAL, dtype=torch.float32)
  body_pos = torch.randn(_TOTAL, _BODIES, 3, generator=generator)
  # Encode the frame index in the root x so the sampled frame is recoverable.
  body_pos[:, 0, 0] = frames * 0.01
  body_quat = torch.zeros(_TOTAL, _BODIES, 4)
  body_quat[..., 0] = 1.0
  body_lin_vel = torch.randn(_TOTAL, _BODIES, 3, generator=generator)
  body_ang_vel = torch.randn(_TOTAL, _BODIES, 3, generator=generator)
  joint_pos = torch.randn(_TOTAL, _JOINTS, generator=generator)
  joint_vel = torch.randn(_TOTAL, _JOINTS, generator=generator)
  return body_pos, body_quat, body_lin_vel, body_ang_vel, joint_pos, joint_vel


def _filtered_stub(
  mode: str, *, num_envs: int = 16
) -> PlanarFilteredReplayMotionCommand:
  """Stub exercising ``_resample_command`` end to end on synthetic motion."""
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  body_pos, body_quat, body_lin_vel, body_ang_vel, joint_pos, joint_vel = (
    _motion_tensors()
  )
  command.cfg = SimpleNamespace(sampling_mode=mode)
  command.time_steps = torch.full((num_envs,), 123, dtype=torch.long)
  command.motion = SimpleNamespace(time_step_total=_TOTAL)
  command._env = SimpleNamespace(device="cpu")
  command.bin_count = 10
  command.metrics = {
    name: torch.zeros(num_envs)
    for name in ("sampling_entropy", "sampling_top1_prob", "sampling_top1_bin")
  }
  command._raw_body_pos_w = lambda: body_pos[command.time_steps].clone()
  command._raw_body_quat_w = lambda: body_quat[command.time_steps].clone()
  command._raw_body_lin_vel_w = lambda: body_lin_vel[command.time_steps].clone()
  command._raw_body_ang_vel_w = lambda: body_ang_vel[command.time_steps].clone()
  command._raw_joint_pos = lambda: joint_pos[command.time_steps].clone()
  command._raw_joint_vel = lambda: joint_vel[command.time_steps].clone()
  command._filtered_root_xy_w = torch.full((num_envs, 2), 9.0)
  command._filtered_root_velocity_xy_w = torch.full((num_envs, 2), 9.0)
  command._root_translation_residual_xy_w = torch.full((num_envs, 2), 9.0)
  command._filter_initialized = torch.zeros(num_envs, dtype=torch.bool)
  command._filtered_joint_pos = torch.full((num_envs, _JOINTS), 9.0)
  command._filtered_joint_vel = torch.full((num_envs, _JOINTS), 9.0)
  command._joint_position_residual = torch.full((num_envs, _JOINTS), 9.0)
  command._posture_hold_remaining_s = torch.full((num_envs, 2), 9.0)
  command._joint_filter_initialized = torch.zeros(num_envs, dtype=torch.bool)
  command._obstacle_history_initialized = torch.ones(num_envs, dtype=torch.bool)
  command._propagate_targets = False
  command.calls: list = []
  command._update_reference_alignment = lambda env_ids: command.calls.append(
    ("align", command.time_steps.clone())
  )
  command._write_reference_state_to_sim = lambda env_ids, *state: command.calls.append(
    (
      "write",
      command.time_steps.clone(),
      env_ids.clone(),
      tuple(t.clone() for t in state),
    )
  )
  command._stub_motion = (
    body_pos,
    body_quat,
    body_lin_vel,
    body_ang_vel,
    joint_pos,
    joint_vel,
  )
  return command


def _state_snapshot(
  command: PlanarFilteredReplayMotionCommand,
) -> dict[str, torch.Tensor]:
  return {
    name: getattr(command, name).clone()
    for name in (
      "time_steps",
      "_filtered_root_xy_w",
      "_filtered_root_velocity_xy_w",
      "_root_translation_residual_xy_w",
      "_filter_initialized",
      "_filtered_joint_pos",
      "_filtered_joint_vel",
      "_joint_position_residual",
      "_posture_hold_remaining_s",
      "_joint_filter_initialized",
      "_obstacle_history_initialized",
    )
  }


def _legacy_resample(
  command: PlanarFilteredReplayMotionCommand, env_ids: torch.Tensor
) -> None:
  """The pre-change method, verbatim in effect (frame zero, reset_to_frame)."""
  command.time_steps[env_ids] = 0
  command._update_reference_alignment(env_ids)
  raw_root_pos = command._raw_body_pos_w()[env_ids, 0]
  raw_root_velocity = command._raw_body_lin_vel_w()[env_ids, 0]
  command._filtered_root_xy_w[env_ids] = raw_root_pos[:, :2]
  command._filtered_root_velocity_xy_w[env_ids] = raw_root_velocity[:, :2]
  command._root_translation_residual_xy_w[env_ids] = 0.0
  command._filter_initialized[env_ids] = True
  command._filtered_joint_pos[env_ids] = command._raw_joint_pos()[env_ids]
  command._filtered_joint_vel[env_ids] = command._raw_joint_vel()[env_ids]
  command._joint_position_residual[env_ids] = 0.0
  command._posture_hold_remaining_s[env_ids] = 0.0
  command._joint_filter_initialized[env_ids] = True
  command._obstacle_history_initialized[env_ids] = False
  # reset_to_frame(env_ids, 0): assign the frame, then write the exact state.
  command.time_steps[env_ids] = 0
  command._write_reference_state_to_sim(
    env_ids,
    command.body_pos_w[env_ids, 0],
    command.body_quat_w[env_ids, 0],
    command.body_lin_vel_w[env_ids, 0],
    command.body_ang_vel_w[env_ids, 0],
    command.joint_pos[env_ids],
    command.joint_vel[env_ids],
  )


def test_start_mode_is_bit_identical_to_legacy_frame_zero_reset() -> None:
  env_ids = torch.tensor([0, 3, 9])
  new = _filtered_stub("start")
  new._resample_command(env_ids)
  legacy = _filtered_stub("start")
  _legacy_resample(legacy, env_ids)
  for name, value in _state_snapshot(new).items():
    assert torch.equal(value, _state_snapshot(legacy)[name]), name
  assert torch.all(new.time_steps[env_ids] == 0)
  # Same call sequence and identical written state.
  assert [c[0] for c in new.calls] == [c[0] for c in legacy.calls] == ["align", "write"]
  for got, want in zip(new.calls[1][3], legacy.calls[1][3], strict=True):
    assert torch.equal(got, want)


def test_uniform_mode_initializes_filter_at_the_sampled_frame() -> None:
  torch.manual_seed(5)
  command = _filtered_stub("uniform")
  env_ids = torch.arange(16)
  command._resample_command(env_ids)
  body_pos, _, body_lin_vel, _, joint_pos, joint_vel = command._stub_motion
  steps = command.time_steps
  assert torch.all(steps >= 0) and torch.all(steps < _TOTAL)
  assert steps.unique().numel() > 4
  # Filter state comes from the RAW reference at each env's own frame.
  assert torch.equal(command._filtered_root_xy_w, body_pos[steps, 0, :2])
  assert torch.equal(command._filtered_root_velocity_xy_w, body_lin_vel[steps, 0, :2])
  assert torch.equal(command._filtered_joint_pos, joint_pos[steps])
  assert torch.equal(command._filtered_joint_vel, joint_vel[steps])
  assert torch.all(command._root_translation_residual_xy_w == 0.0)
  assert torch.all(command._joint_position_residual == 0.0)
  assert torch.all(command._posture_hold_remaining_s == 0.0)
  assert torch.all(command._filter_initialized) and torch.all(
    command._joint_filter_initialized
  )
  assert not torch.any(command._obstacle_history_initialized)
  # Alignment ran after sampling (saw the sampled frames), then the write
  # used the same frames and wrote the frame-encoded root x.
  align_steps = command.calls[0][1]
  assert torch.equal(align_steps, steps)
  write_kind, write_steps, write_env_ids, written = command.calls[1]
  assert write_kind == "write"
  assert torch.equal(write_steps, steps) and torch.equal(write_env_ids, env_ids)
  root_pos = written[0]
  assert torch.allclose(root_pos[:, 0], steps.to(torch.float32) * 0.01)
  assert torch.equal(written[4], joint_pos[steps])


def test_adaptive_mode_raises_before_touching_filter_state() -> None:
  command = _filtered_stub("adaptive")
  before = _state_snapshot(command)
  with pytest.raises(NotImplementedError):
    command._resample_command(torch.arange(4))
  for name, value in _state_snapshot(command).items():
    assert torch.equal(value, before[name]), name
  assert command.calls == []


@pytest.mark.parametrize(
  "builder",
  [
    unitree_g1_lidar_avoidance_tracking_env_cfg,
    unitree_g1_lidar_range_rate_avoidance_tracking_env_cfg,
    unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
    unitree_g1_sparse_lidar_avoidance_tracking_env_cfg,
  ],
)
@pytest.mark.parametrize("play", [False, True])
def test_avoidance_family_samples_uniform_start_frames(builder, play) -> None:
  cfg = builder(play=play)
  expected = "start" if play else "uniform"
  assert cfg.commands["motion"].sampling_mode == expected


@pytest.mark.parametrize(
  "builder",
  [
    unitree_g1_kinematic_reference_lidar_demo_env_cfg,
    unitree_g1_reference_filter_lidar_demo_env_cfg,
    unitree_g1_reference_filter_policy_lidar_demo_env_cfg,
  ],
)
def test_demo_cfgs_keep_exact_frame_zero_replay(builder) -> None:
  assert builder().commands["motion"].sampling_mode == "start"
