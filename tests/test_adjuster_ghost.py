"""Play-only ghost override: draw the adjuster's command instead of the teacher."""

from types import SimpleNamespace

import numpy as np
import torch

from safe_mimic.tasks.kinematic_replay_command import PlanarFilteredReplayMotionCommand


class _FakeVisualizer:
  def __init__(self) -> None:
    self.ghosts: list[tuple[np.ndarray, object, str | None]] = []

  def get_env_indices(self, num_envs: int):
    return range(num_envs)

  def add_ghost_mesh(
    self, qpos, model, mocap_pos=None, mocap_quat=None, alpha=0.5, label=None
  ):
    self.ghosts.append((np.array(qpos, copy=True), model, label))


def _command(num_envs: int = 2, num_joints: int = 3, parent_calls: list | None = None):
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command.cfg = SimpleNamespace(viz=SimpleNamespace(mode="ghost"), entity_name="robot")
  nq = 7 + num_joints
  model = SimpleNamespace(
    nq=nq,
    ngeom=0,
    geom_contype=np.zeros(0),
    geom_conaffinity=np.zeros(0),
    geom_rgba=np.zeros((0, 4)),
  )
  indexing = SimpleNamespace(
    free_joint_q_adr=torch.arange(7),
    joint_q_adr=torch.arange(7, nq),
  )
  command._env = SimpleNamespace(
    num_envs=num_envs,
    sim=SimpleNamespace(mj_model=model),
    scene={"robot": SimpleNamespace(indexing=indexing)},
  )
  raw_pos = torch.tensor([[[1.0, 2.0, 0.8]], [[-1.0, 0.5, 0.7]]])
  raw_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]])
  raw_joint = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
  command._raw_body_pos_w = lambda: raw_pos
  command._raw_body_quat_w = lambda: raw_quat
  command._raw_joint_pos = lambda: raw_joint
  command._adjuster_ghost_model_cache = "adjuster-model"
  parent_calls = [] if parent_calls is None else parent_calls
  return command, raw_pos, raw_quat, raw_joint


def test_no_override_falls_through_to_teacher_ghost(monkeypatch) -> None:
  command, *_ = _command()
  calls = []
  monkeypatch.setattr(
    "mjlab.tasks.tracking.mdp.commands.MotionCommand._debug_vis_impl",
    lambda self, visualizer: calls.append("teacher"),
  )
  viz = _FakeVisualizer()
  command._debug_vis_impl(viz)
  assert calls == ["teacher"]
  assert viz.ghosts == []
  command.set_ghost_override(torch.zeros(2, 3))
  command.set_ghost_override(None)
  command._debug_vis_impl(viz)
  assert calls == ["teacher", "teacher"]


def test_override_poses_ghost_from_raw_root_plus_residual(monkeypatch) -> None:
  command, raw_pos, raw_quat, raw_joint = _command()
  calls = []
  monkeypatch.setattr(
    "mjlab.tasks.tracking.mdp.commands.MotionCommand._debug_vis_impl",
    lambda self, visualizer: calls.append("teacher"),
  )
  residual = torch.tensor([[0.5, 0.0, -0.25], [0.0, 0.75, 0.0]])
  command.set_ghost_override(residual)
  viz = _FakeVisualizer()
  command._debug_vis_impl(viz)
  assert calls == []  # teacher ghost replaced, not drawn
  assert [label for _, _, label in viz.ghosts] == [
    "adjuster_ghost_0",
    "adjuster_ghost_1",
  ]
  for batch, (qpos, model, _) in enumerate(viz.ghosts):
    assert model == "adjuster-model"
    np.testing.assert_allclose(qpos[0:3], raw_pos[batch, 0].numpy())
    np.testing.assert_allclose(qpos[3:7], raw_quat[batch, 0].numpy())
    np.testing.assert_allclose(qpos[7:], (raw_joint[batch] + residual[batch]).numpy())


def test_show_teacher_draws_both(monkeypatch) -> None:
  command, *_ = _command()
  calls = []
  monkeypatch.setattr(
    "mjlab.tasks.tracking.mdp.commands.MotionCommand._debug_vis_impl",
    lambda self, visualizer: calls.append("teacher"),
  )
  command.set_ghost_override(torch.zeros(2, 3), show_teacher=True)
  viz = _FakeVisualizer()
  command._debug_vis_impl(viz)
  assert calls == ["teacher"]
  assert len(viz.ghosts) == 2


def test_frames_mode_ignores_override(monkeypatch) -> None:
  command, *_ = _command()
  command.cfg.viz.mode = "frames"
  calls = []
  monkeypatch.setattr(
    "mjlab.tasks.tracking.mdp.commands.MotionCommand._debug_vis_impl",
    lambda self, visualizer: calls.append("teacher"),
  )
  command.set_ghost_override(torch.zeros(2, 3))
  viz = _FakeVisualizer()
  command._debug_vis_impl(viz)
  assert calls == ["teacher"] and viz.ghosts == []
