"""Closed-loop integration of the privileged filtered root target."""

from types import SimpleNamespace

import torch

import safe_mimic.tasks.kinematic_replay_command as krc
from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
  PlanarFilteredReplayMotionCommandCfg,
  advance_filtered_root_xy,
)


def test_cfg_defaults_off() -> None:
  fields = {
    f.name: f.default
    for f in PlanarFilteredReplayMotionCommandCfg.__dataclass_fields__.values()
  }
  assert fields["closed_loop_root_target"] is False
  assert fields["max_root_lead_m"] is None
  assert fields["planar_filter_at_robot_root"] is False
  assert fields["disable_filters"] is False


def test_cfg_rejects_non_positive_leash() -> None:
  import pytest

  base = dict(
    motion_file="x.npz",
    anchor_body_name="torso_link",
    body_names=("torso_link",),
    obstacle_entity_names=("human",),
    resampling_time_range=(1.0e9, 1.0e9),
    entity_name="robot",
  )
  PlanarFilteredReplayMotionCommandCfg(**base, max_root_lead_m=0.3)
  for bad in (0.0, -0.1):
    with pytest.raises(ValueError, match="max_root_lead_m"):
      PlanarFilteredReplayMotionCommandCfg(**base, max_root_lead_m=bad)


def test_open_loop_matches_legacy_bookkeeping() -> None:
  raw = torch.tensor([[1.0, 2.0]])
  residual = torch.tensor([[0.10, -0.05]])
  v_raw = torch.tensor([[0.5, 0.0]])
  v_filt = torch.tensor([[0.5, 1.0]])
  filtered, new_residual = advance_filtered_root_xy(
    pre_filter_xy_w=raw + residual,
    raw_root_xy_w=raw,
    raw_root_velocity_xy_w=v_raw,
    filtered_velocity_xy_w=v_filt,
    translation_residual_xy_w=residual,
    step_dt=0.02,
    closed_loop=False,
  )
  torch.testing.assert_close(new_residual, residual + 0.02 * (v_filt - v_raw))
  torch.testing.assert_close(filtered, raw + new_residual)


def test_closed_loop_integrates_world_frame_regardless_of_robot() -> None:
  previous_filtered = torch.tensor([[3.0, 4.0]])
  v_filt = torch.tensor([[1.0, -2.0]])
  # Robot (live-aligned raw root) did or did not move: same filtered target.
  for raw in (torch.tensor([[3.0, 4.0]]), torch.tensor([[2.5, 4.4]])):
    filtered, residual = advance_filtered_root_xy(
      pre_filter_xy_w=previous_filtered,
      raw_root_xy_w=raw,
      raw_root_velocity_xy_w=torch.zeros(1, 2),
      filtered_velocity_xy_w=v_filt,
      translation_residual_xy_w=torch.zeros(1, 2),
      step_dt=0.02,
      closed_loop=True,
    )
    torch.testing.assert_close(filtered, previous_filtered + 0.02 * v_filt)
    torch.testing.assert_close(residual, filtered - raw)


def test_closed_loop_target_stops_running_ahead_of_a_compliant_robot() -> None:
  # Robot tracks the filtered velocity exactly: the residual must stay zero in
  # closed loop but grow without bound in open loop when v_raw != v_filt.
  dt = 0.02
  v_raw = torch.tensor([[0.0, 0.0]])
  v_filt = torch.tensor([[1.0, 0.0]])
  raw = torch.zeros(1, 2)
  filtered_closed = torch.zeros(1, 2)
  residual_closed = torch.zeros(1, 2)
  residual_open = torch.zeros(1, 2)
  for _ in range(50):
    raw = raw + dt * v_filt  # live alignment: raw root == robot root
    filtered_closed, residual_closed = advance_filtered_root_xy(
      pre_filter_xy_w=filtered_closed,
      raw_root_xy_w=raw,
      raw_root_velocity_xy_w=v_raw,
      filtered_velocity_xy_w=v_filt,
      translation_residual_xy_w=residual_closed,
      step_dt=dt,
      closed_loop=True,
    )
    _, residual_open = advance_filtered_root_xy(
      pre_filter_xy_w=raw + residual_open,
      raw_root_xy_w=raw,
      raw_root_velocity_xy_w=v_raw,
      filtered_velocity_xy_w=v_filt,
      translation_residual_xy_w=residual_open,
      step_dt=dt,
      closed_loop=False,
    )
  torch.testing.assert_close(residual_closed, torch.zeros(1, 2), atol=1e-5, rtol=0)
  torch.testing.assert_close(
    residual_open, torch.tensor([[1.0, 0.0]]), atol=1e-5, rtol=0
  )


def test_closed_loop_non_compliant_robot_accumulates_error() -> None:
  # Robot stands still while the filter asks for 1 m/s: the closed-loop
  # residual grows like the open-loop one (both integrate dt * v_filt here).
  dt = 0.02
  raw = torch.zeros(1, 2)
  filtered = torch.zeros(1, 2)
  residual = torch.zeros(1, 2)
  for _ in range(25):
    filtered, residual = advance_filtered_root_xy(
      pre_filter_xy_w=filtered,
      raw_root_xy_w=raw,
      raw_root_velocity_xy_w=torch.zeros(1, 2),
      filtered_velocity_xy_w=torch.tensor([[1.0, 0.0]]),
      translation_residual_xy_w=residual,
      step_dt=dt,
      closed_loop=True,
    )
  torch.testing.assert_close(residual, torch.tensor([[0.5, 0.0]]), atol=1e-5, rtol=0)


def test_inputs_are_not_mutated() -> None:
  residual = torch.tensor([[0.1, 0.2]])
  pre = torch.tensor([[1.0, 1.0]])
  residual_copy, pre_copy = residual.clone(), pre.clone()
  for closed_loop in (False, True):
    advance_filtered_root_xy(
      pre_filter_xy_w=pre,
      raw_root_xy_w=torch.zeros(1, 2),
      raw_root_velocity_xy_w=torch.ones(1, 2),
      filtered_velocity_xy_w=torch.zeros(1, 2),
      translation_residual_xy_w=residual,
      step_dt=0.02,
      closed_loop=closed_loop,
    )
  torch.testing.assert_close(residual, residual_copy)
  torch.testing.assert_close(pre, pre_copy)


def test_leash_clamps_lead_along_its_direction_in_closed_loop() -> None:
  raw = torch.tensor([[1.0, 2.0]])
  pre = torch.tensor([[1.6, 2.8]])  # leads raw by (0.6, 0.8) = 1.0 m
  v_filt = torch.zeros(1, 2)
  filtered, residual = advance_filtered_root_xy(
    pre_filter_xy_w=pre,
    raw_root_xy_w=raw,
    raw_root_velocity_xy_w=torch.zeros(1, 2),
    filtered_velocity_xy_w=v_filt,
    translation_residual_xy_w=pre - raw,
    step_dt=0.02,
    closed_loop=True,
    max_lead_m=0.3,
  )
  torch.testing.assert_close(residual, torch.tensor([[0.18, 0.24]]))
  torch.testing.assert_close(filtered, raw + residual)


def test_leash_leaves_a_target_inside_the_leash_untouched() -> None:
  raw = torch.tensor([[1.0, 2.0]])
  pre = torch.tensor([[1.1, 2.1]])
  v_filt = torch.tensor([[0.5, 0.0]])
  kwargs = dict(
    pre_filter_xy_w=pre,
    raw_root_xy_w=raw,
    raw_root_velocity_xy_w=torch.zeros(1, 2),
    filtered_velocity_xy_w=v_filt,
    translation_residual_xy_w=pre - raw,
    step_dt=0.02,
    closed_loop=True,
  )
  plain_f, plain_r = advance_filtered_root_xy(**kwargs)
  leash_f, leash_r = advance_filtered_root_xy(**kwargs, max_lead_m=0.3)
  torch.testing.assert_close(leash_f, plain_f)
  torch.testing.assert_close(leash_r, plain_r)


def test_leash_also_bounds_the_open_loop_residual() -> None:
  raw = torch.tensor([[0.0, 0.0]])
  residual = torch.tensor([[0.5, 0.0]])
  filtered, new_residual = advance_filtered_root_xy(
    pre_filter_xy_w=raw + residual,
    raw_root_xy_w=raw,
    raw_root_velocity_xy_w=torch.zeros(1, 2),
    filtered_velocity_xy_w=torch.tensor([[1.0, 0.0]]),
    translation_residual_xy_w=residual,
    step_dt=0.02,
    closed_loop=False,
    max_lead_m=0.3,
  )
  torch.testing.assert_close(new_residual, torch.tensor([[0.3, 0.0]]))
  torch.testing.assert_close(filtered, torch.tensor([[0.3, 0.0]]))


def test_leash_none_is_the_unbounded_legacy_behaviour() -> None:
  raw = torch.tensor([[0.0, 0.0]])
  pre = torch.tensor([[3.0, 0.0]])
  filtered, residual = advance_filtered_root_xy(
    pre_filter_xy_w=pre,
    raw_root_xy_w=raw,
    raw_root_velocity_xy_w=torch.zeros(1, 2),
    filtered_velocity_xy_w=torch.zeros(1, 2),
    translation_residual_xy_w=pre - raw,
    step_dt=0.02,
    closed_loop=True,
    max_lead_m=None,
  )
  torch.testing.assert_close(residual, torch.tensor([[3.0, 0.0]]))


# --- stub-driven `_update_command` -------------------------------------------

_N = 3  # env 0: steady; env 1: filter uninitialized; env 2: motion wraps
_J = 2
_TOTAL = 100
_DT = 0.02


def _update_stub(
  monkeypatch,
  *,
  closed_loop: bool,
  max_root_lead_m: float | None = None,
  planar_filter_at_robot_root: bool = False,
  disable_filters: bool = False,
) -> tuple[PlanarFilteredReplayMotionCommand, dict]:
  """Minimal fakes for every collaborator of ``_update_command``.

  The planar filter is replaced by a fake that records the position it was
  evaluated at and returns a fixed escape velocity, so the test can pin both
  the pre-filter position selection and the post-filter integration.
  """
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command.cfg = SimpleNamespace(
    closed_loop_root_target=closed_loop,
    max_root_lead_m=max_root_lead_m,
    planar_filter_at_robot_root=planar_filter_at_robot_root,
    disable_filters=disable_filters,
    write_reference_to_sim=False,
    planar_filter=SimpleNamespace(
      robot_radius_m=0.35,
      vertical_gate_m=1.0,
      recovery_gain=0.8,
      max_recovery_speed_mps=0.4,
    ),
    link_filter=SimpleNamespace(
      joint_recovery_gain=3.0, max_joint_velocity_correction_rps=1.5
    ),
  )
  command._all_env_ids = torch.arange(_N)
  command.time_steps = torch.tensor([10, 20, _TOTAL - 1])
  command.motion = SimpleNamespace(time_step_total=_TOTAL)
  command._env = SimpleNamespace(step_dt=_DT)
  command.robot = SimpleNamespace(
    data=SimpleNamespace(soft_joint_pos_limits=torch.tensor([[[-3.0, 3.0]] * _J] * _N))
  )
  command._joint_position_residual_limit_rad = torch.full((_J,), 0.75)
  command._propagate_targets = False
  command.metrics = {
    name: torch.zeros(_N)
    for name in (
      "joint_filter_reference_residual_rad",
      "filter_minimum_clearance_m",
      "filter_intervention_speed_mps",
      "filter_reference_offset_m",
      "filter_cbf_violation_mps",
    )
  }
  raw_root = torch.tensor([[1.0, 2.0, 0.8], [-1.0, 0.5, 0.8], [3.0, -2.0, 0.8]])
  raw_vel = torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.2, 0.0], [-0.1, 0.1, 0.0]])
  cloud_pos = torch.zeros(_N, 2, 3)
  cloud_pos[:, 0] = raw_root
  cloud_vel = torch.zeros(_N, 2, 3)
  cloud_vel[:, 0] = raw_vel
  command._raw_body_pos_w = lambda: cloud_pos.clone()
  command._raw_body_lin_vel_w = lambda: cloud_vel.clone()
  command._raw_joint_pos = lambda: torch.zeros(_N, _J)
  command._raw_joint_vel = lambda: torch.zeros(_N, _J)
  command._update_reference_alignment = lambda env_ids: None
  command.update_relative_body_poses = lambda: None
  # Persistent filter state that differs from raw + residual on EVERY env so a
  # wrong pre-filter selection is visible.
  command._filtered_root_xy_w = torch.tensor([[1.5, 2.5], [-0.4, 0.9], [3.7, -2.6]])
  command._filtered_root_velocity_xy_w = torch.zeros(_N, 2)
  command._root_translation_residual_xy_w = torch.tensor(
    [[0.1, -0.1], [0.2, 0.2], [-0.3, 0.3]]
  )
  command._filter_initialized = torch.tensor([True, False, True])
  command._filtered_joint_pos = torch.zeros(_N, _J)
  command._filtered_joint_vel = torch.zeros(_N, _J)
  command._joint_position_residual = torch.zeros(_N, _J)
  command._posture_hold_remaining_s = torch.zeros(_N, 2)
  command._joint_filter_initialized = torch.tensor([True, False, True])
  command._obstacle_history_initialized = torch.ones(_N, dtype=torch.bool)
  command._obstacle_tensors = lambda: (
    torch.zeros(_N, 1, 3),
    torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * _N),
    torch.zeros(1, 3),
  )
  command._estimate_obstacle_velocities = lambda centers, active: torch.zeros(_N, 1, 3)
  command._select_link_filter_obstacles = lambda *args: (
    torch.zeros(_N, 1, 3),
    torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * _N),
    torch.zeros(1, 3),
    torch.zeros(_N, 1, 3),
  )
  command._joint_velocity_avoidance_correction = lambda *args: (
    torch.zeros(_N, _J),
    torch.zeros(_N, _J),
  )
  escape = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5]])
  seen: dict = {"raw_root": raw_root, "raw_vel": raw_vel, "escape": escape}

  def fake_geometry(robot_position_w, *args, **kwargs):
    seen["geometry_position"] = robot_position_w.clone()
    return (
      torch.zeros(_N, 1, 2),
      torch.full((_N, 1), 10.0),
      torch.zeros(_N, 1, dtype=torch.bool),
    )

  monkeypatch.setattr(krc, "planar_capsule_geometry", fake_geometry)
  monkeypatch.setattr(krc, "gate_joint_recovery_during_posture", lambda r, p: r)

  def fake_filter(cfg, *, robot_position_xy_w, nominal_velocity_xy_w, **kwargs):
    seen["position"] = robot_position_xy_w.clone()
    seen["nominal"] = nominal_velocity_xy_w.clone()
    return SimpleNamespace(
      velocity_w=escape.clone(),
      minimum_clearance_m=torch.full((_N,), 10.0),
      intervention_w=torch.zeros(_N, 2),
      maximum_cbf_violation_mps=torch.zeros(_N),
    )

  monkeypatch.setattr(krc, "filter_planar_velocity", fake_filter)
  seen["stale_filtered"] = command._filtered_root_xy_w.clone()
  seen["residual_before"] = command._root_translation_residual_xy_w.clone()
  return command, seen


def test_update_command_closed_loop_keeps_persistent_target(monkeypatch) -> None:
  command, seen = _update_stub(monkeypatch, closed_loop=True)
  command._update_command()
  raw_xy = seen["raw_root"][:, :2]
  # Env 0 (steady): the filter is evaluated at the PERSISTENT target, not at
  # raw + residual, and the target integrates from there.
  torch.testing.assert_close(seen["position"][0], seen["stale_filtered"][0])
  expected0 = seen["stale_filtered"][0] + _DT * seen["escape"][0]
  torch.testing.assert_close(command._filtered_root_xy_w[0], expected0)
  torch.testing.assert_close(
    command._root_translation_residual_xy_w[0], expected0 - raw_xy[0]
  )
  # Env 1 (first initialization) and env 2 (motion wrap): reset to the raw
  # root BEFORE the pre-filter position is taken, then integrated.
  for env in (1, 2):
    assert not torch.equal(seen["stale_filtered"][env], raw_xy[env])
    torch.testing.assert_close(seen["position"][env], raw_xy[env])
    torch.testing.assert_close(
      command._filtered_root_xy_w[env], raw_xy[env] + _DT * seen["escape"][env]
    )
    torch.testing.assert_close(
      command._root_translation_residual_xy_w[env], _DT * seen["escape"][env]
    )
  assert bool(command._filter_initialized.all())
  assert command.time_steps.tolist() == [11, 21, 0]
  torch.testing.assert_close(command._filtered_root_velocity_xy_w, seen["escape"])
  torch.testing.assert_close(
    command.metrics["filter_reference_offset_m"],
    torch.linalg.vector_norm(command._root_translation_residual_xy_w, dim=-1),
  )


def test_update_command_open_loop_reglues_to_raw_root(monkeypatch) -> None:
  command, seen = _update_stub(monkeypatch, closed_loop=False)
  command._update_command()
  raw_xy = seen["raw_root"][:, :2]
  raw_v = seen["raw_vel"][:, :2]
  residual0 = seen["residual_before"][0]
  # Env 0: pre-filter position is raw + residual (legacy re-glue) ...
  torch.testing.assert_close(seen["position"][0], raw_xy[0] + residual0)
  # ... and the residual integrates the velocity difference.
  expected_residual = residual0 + _DT * (seen["escape"][0] - raw_v[0])
  torch.testing.assert_close(
    command._root_translation_residual_xy_w[0], expected_residual
  )
  torch.testing.assert_close(
    command._filtered_root_xy_w[0], raw_xy[0] + expected_residual
  )
  for env in (1, 2):
    torch.testing.assert_close(seen["position"][env], raw_xy[env])
    torch.testing.assert_close(
      command._root_translation_residual_xy_w[env],
      _DT * (seen["escape"][env] - raw_v[env]),
    )


def test_update_command_modes_differ_by_the_documented_relation(
  monkeypatch,
) -> None:
  closed, seen = _update_stub(monkeypatch, closed_loop=True)
  closed._update_command()
  open_, _ = _update_stub(monkeypatch, closed_loop=False)
  open_._update_command()
  raw_v = seen["raw_vel"][:, :2]
  # Reset rows (1, 2) start from raw in both modes; closed loop integrates the
  # full filtered velocity while open loop integrates only the delta to the
  # raw velocity (the raw root is re-glued to the robot), so they differ by
  # exactly dt * v_raw.
  torch.testing.assert_close(
    closed._filtered_root_xy_w[1:] - open_._filtered_root_xy_w[1:],
    _DT * raw_v[1:],
  )
  # The steady row also carries the persistent-target offset.
  steady_gap = closed._filtered_root_xy_w[0] - open_._filtered_root_xy_w[0]
  assert not torch.allclose(steady_gap, _DT * raw_v[0])


def test_update_command_planar_filter_at_robot_root(monkeypatch) -> None:
  command, seen = _update_stub(
    monkeypatch, closed_loop=True, planar_filter_at_robot_root=True
  )
  command._update_command()
  raw_xy = seen["raw_root"][:, :2]
  # Both the capsule geometry and the CBF are evaluated at the live-aligned
  # raw root (the robot), on EVERY env, even when the persistent target leads.
  torch.testing.assert_close(seen["geometry_position"][:, :2], raw_xy)
  torch.testing.assert_close(seen["position"], raw_xy)
  # The target itself still integrates from the persistent position.
  expected0 = seen["stale_filtered"][0] + _DT * seen["escape"][0]
  torch.testing.assert_close(command._filtered_root_xy_w[0], expected0)


def test_update_command_default_evaluates_filter_at_the_target(monkeypatch) -> None:
  command, seen = _update_stub(monkeypatch, closed_loop=True)
  command._update_command()
  torch.testing.assert_close(
    seen["geometry_position"][0, :2], seen["stale_filtered"][0]
  )


def test_update_command_leash_bounds_the_persistent_target(monkeypatch) -> None:
  command, seen = _update_stub(monkeypatch, closed_loop=True, max_root_lead_m=0.3)
  command._update_command()
  raw_xy = seen["raw_root"][:, :2]
  lead = command._filtered_root_xy_w - raw_xy
  torch.testing.assert_close(lead, command._root_translation_residual_xy_w)
  # Env 0 started 0.707 m ahead: clamped onto the leash along its direction.
  lead0 = seen["stale_filtered"][0] + _DT * seen["escape"][0] - raw_xy[0]
  torch.testing.assert_close(lead[0], 0.3 * lead0 / lead0.norm())
  torch.testing.assert_close(
    command.metrics["filter_reference_offset_m"][0], torch.tensor(0.3)
  )
  # Reset rows integrate one step from raw and stay inside the leash.
  for env in (1, 2):
    torch.testing.assert_close(lead[env], _DT * seen["escape"][env])


def test_update_command_with_filters_disabled_passes_the_raw_reference_through(
  monkeypatch,
) -> None:
  """The no-avoidance baseline: nominal reference, zero teacher correction."""
  command, seen = _update_stub(monkeypatch, closed_loop=True, disable_filters=True)
  command._update_command()
  raw_xy = seen["raw_root"][:, :2]
  raw_v = seen["raw_vel"][:, :2]
  # The planar CBF is never even evaluated.
  assert "position" not in seen
  torch.testing.assert_close(command._filtered_root_xy_w, raw_xy)
  torch.testing.assert_close(command._filtered_root_velocity_xy_w, raw_v)
  torch.testing.assert_close(
    command._root_translation_residual_xy_w, torch.zeros(_N, 2)
  )
  torch.testing.assert_close(command._filtered_joint_pos, torch.zeros(_N, _J))
  torch.testing.assert_close(command._joint_position_residual, torch.zeros(_N, _J))
  assert bool(command._filter_initialized.all())
  assert bool(command._joint_filter_initialized.all())
  torch.testing.assert_close(
    command.metrics["filter_intervention_speed_mps"], torch.zeros(_N)
  )
  torch.testing.assert_close(
    command.metrics["filter_reference_offset_m"], torch.zeros(_N)
  )
  torch.testing.assert_close(
    command.metrics["joint_filter_reference_residual_rad"], torch.zeros(_N)
  )
  # Time still advances.
  assert command.time_steps.tolist() == [11, 21, 0]
