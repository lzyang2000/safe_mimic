"""Escape-move state machine and blended raw getters inside the replay command."""

from types import SimpleNamespace

import torch

from safe_mimic.tasks.escape_moves import EscapeMoveCfg, EscapeMoveTable
from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
  PlanarFilteredReplayMotionCommandCfg,
)

_N = 2
_J = 2
_B = 2  # body 0 = root, body 1 = anchor
_DT = 0.02
_FRAMES = 30  # three clips of ten frames


def test_cfg_default_is_off() -> None:
  fields = {
    f.name: f.default
    for f in PlanarFilteredReplayMotionCommandCfg.__dataclass_fields__.values()
  }
  assert fields["escape_moves"] is None


def _motion() -> SimpleNamespace:
  frame = torch.arange(_FRAMES, dtype=torch.float32)
  joint_pos = frame[:, None].repeat(1, _J)
  body_pos = torch.zeros(_FRAMES, _B, 3)
  body_pos[..., 0] = frame[:, None]  # travels along x
  body_pos[..., 2] = frame[:, None]  # z encodes the frame for blend checks
  quat = torch.zeros(_FRAMES, _B, 4)
  quat[..., 0] = 1.0
  return SimpleNamespace(
    joint_pos=joint_pos,
    joint_vel=torch.ones(_FRAMES, _J),
    body_pos_w=body_pos,
    body_quat_w=quat,
    body_lin_vel_w=torch.ones(_FRAMES, _B, 3),
    body_ang_vel_w=torch.zeros(_FRAMES, _B, 3),
    time_step_total=_FRAMES,
    clip_start_idx=torch.tensor([0, 10, 20]),
    clip_num_frames=torch.tensor([10, 10, 10]),
  )


def _stub(*, blend_s: float = 0.1, cooldown_s: float = 0.2, source: str = "teacher"):
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command.cfg = SimpleNamespace(
    escape_moves=EscapeMoveCfg(
      index_file="unused.json",
      blend_s=blend_s,
      cooldown_s=cooldown_s,
      trigger_source=source,
    ),
    align_reference_to_robot_each_step=True,
  )

  command.metrics = {}
  command.motion = _motion()
  command.motion_anchor_body_index = 1
  command._env = SimpleNamespace(
    step_dt=_DT,
    num_envs=_N,
    device="cpu",
    scene=SimpleNamespace(env_origins=torch.zeros(_N, 3)),
  )
  command._all_env_ids = torch.arange(_N)
  command.time_steps = torch.tensor([4, 14])
  command._clip_start = torch.tensor([0, 10])
  command._clip_end = torch.tensor([10, 20])
  # Robot: root at (5, 5), heading identity. The stub replaces the live
  # alignment inputs so no scene is needed.
  robot_root = torch.tensor([[5.0, 5.0, 0.8]] * _N)
  robot_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * _N)
  command._alignment_targets = lambda: (robot_root, robot_quat)
  command._reference_alignment_yaw_w = robot_quat.clone()
  command._reference_alignment_root_xy_w = robot_root[:, :2].clone()
  table = EscapeMoveTable(
    entry_frames=torch.tensor([22]),
    exit_frames=torch.tensor([27]),
    clip_starts=torch.tensor([20]),
    clip_ends=torch.tensor([30]),
    direction_b=torch.tensor([[1.0, 0.0]]),
    speed_mps=torch.tensor([1.0]),
    entry_joint_pos=torch.zeros(1, _J),
  )
  command._init_escape_moves(table)
  return command


def _push(command, vec: tuple[float, float], steps: int) -> None:
  for _ in range(steps):
    command._last_intervention_w = torch.tensor([list(vec), [0.0, 0.0]])
    command._update_escape_moves(command._all_env_ids)


def test_sustained_intervention_switches_to_the_aligned_move() -> None:
  command = _stub()
  _push(command, (0.5, 0.0), 2)
  assert command._escape_mode.tolist() == [0, 0]  # two steps are not enough
  _push(command, (0.5, 0.0), 1)
  assert command._escape_mode.tolist() == [1, 0]
  assert command.time_steps.tolist() == [22, 14]
  assert command._clip_start[0] == 20 and command._clip_end[0] == 30
  assert (
    command._escape_saved_start[0],
    command._escape_saved_end[0],
    command._escape_saved_frame[0],
  ) == (0, 10, 4)
  assert command._escape_exit_frame[0] == 27
  assert command._escape_blend_frame[0] == 4
  assert command._escape_blend_steps_left[0] == 5  # 0.1 s at 50 Hz
  assert command.metrics["escape_move_count"].tolist() == [1.0, 0.0]
  assert command.metrics["escape_move_active"].tolist() == [1.0, 0.0]
  # On the switch step the blend is fully the frozen pre-switch pose.
  torch.testing.assert_close(command._raw_joint_pos()[0], torch.tensor([4.0, 4.0]))


def test_weak_or_misaligned_intervention_does_not_switch() -> None:
  command = _stub()
  _push(command, (0.2, 0.0), 5)  # below 0.25 m/s
  assert command._escape_mode.tolist() == [0, 0]
  _push(command, (0.0, -0.5), 3)  # to the right: no candidate
  assert command._escape_mode.tolist() == [0, 0]
  assert command._escape_trigger_count[0] == 0  # reset after a failed selection


def test_exit_resumes_the_saved_clip_and_starts_the_cooldown() -> None:
  command = _stub()
  _push(command, (0.5, 0.0), 3)
  command.time_steps[0] = 27  # the move reached its exit frame
  _push(command, (0.0, 0.0), 1)
  assert command._escape_mode.tolist() == [0, 0]
  assert command.time_steps.tolist() == [4, 14]
  assert command._clip_start[0] == 0 and command._clip_end[0] == 10
  assert command._escape_blend_frame[0] == 26
  assert command._escape_cooldown_steps[0] == 10  # 0.2 s
  assert command.metrics["escape_move_count"][0] == 1.0
  assert command.metrics["escape_move_active"][0] == 0.0
  _push(command, (0.5, 0.0), 5)  # inside the cooldown: no new move
  assert command._escape_mode.tolist() == [0, 0]
  _push(command, (0.5, 0.0), 8)  # cooldown over -> triggers again
  assert command._escape_mode.tolist() == [1, 0]
  assert command.metrics["escape_move_count"][0] == 2.0


def test_blend_interpolates_joints_and_aligned_bodies() -> None:
  command = _stub(blend_s=0.08)  # 4 steps
  command._escape_blend_frame[0] = 4  # frozen pre-switch frame
  command._escape_blend_steps_left[0] = 2  # alpha 0.5
  command.time_steps[0] = 24  # current frame -> midpoint 14
  torch.testing.assert_close(command._raw_joint_pos()[0], torch.tensor([14.0, 14.0]))
  pos = command._raw_body_pos_w()
  # Both frames are glued to the robot root xy; z is the frame midpoint.
  torch.testing.assert_close(pos[0, 0, :2], torch.tensor([5.0, 5.0]))
  torch.testing.assert_close(pos[0, 0, 2], torch.tensor(14.0))
  # Env 1 has no blend and reads its own frame.
  torch.testing.assert_close(command._raw_joint_pos()[1], torch.tensor([14.0, 14.0]))
  torch.testing.assert_close(pos[1, 0, 2], torch.tensor(14.0))
  quat = command._raw_body_quat_w()
  assert torch.allclose(torch.linalg.vector_norm(quat, dim=-1), torch.ones(_N, _B))


def test_blend_counts_down_and_clears() -> None:
  command = _stub(blend_s=0.04)  # 2 steps
  _push(command, (0.5, 0.0), 3)
  assert command._escape_blend_steps_left[0] == 2
  _push(command, (0.0, 0.0), 1)
  assert command._escape_blend_steps_left[0] == 1
  _push(command, (0.0, 0.0), 1)
  assert command._escape_blend_frame[0] == -1  # blend finished
  torch.testing.assert_close(command._raw_joint_pos()[0], torch.tensor([22.0, 22.0]))


def test_actor_source_uses_the_hint_not_the_teacher() -> None:
  command = _stub(source="actor")
  _push(command, (0.9, 0.0), 4)  # strong teacher signal is ignored
  assert command._escape_mode.tolist() == [0, 0]
  for _ in range(3):
    command.set_actor_escape_hint(torch.tensor([[0.5, 0.0], [0.0, 0.0]]))
    command._update_escape_moves(command._all_env_ids)
  assert command._escape_mode.tolist() == [1, 0]


def test_reset_clears_the_planner_state() -> None:
  command = _stub()
  _push(command, (0.5, 0.0), 3)
  command._reset_escape_state(torch.tensor([0]))
  assert command._escape_mode[0] == 0
  assert command._escape_blend_frame[0] == -1
  assert command._escape_trigger_count[0] == 0
  assert command._escape_cooldown_steps[0] == 0
  assert command.metrics["escape_move_count"][0] == 0.0
