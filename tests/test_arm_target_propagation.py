"""Reference-frame FK propagation of arm corrections into body targets."""

import math

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul

import safe_mimic.tasks.kinematic_replay_command as krc
from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
  hinge_chain_body_positions,
)

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _chain_kwargs(**overrides):
  kwargs = {
    "parent_chain_index": (-1, 0, 1),
    "parent_root_slot": (0, 0, 0),
    "body_pos_l": torch.tensor([[0.0] * 3, [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    "body_quat_l": torch.tensor([_IDENTITY_QUAT] * 3),
    "joint_pos_l": torch.zeros(3, 3),
    "joint_axis_l": torch.tensor([[0.0, 0.0, 1.0]] * 3),
  }
  kwargs.update(overrides)
  return kwargs


def _assert_quat_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
  """Compare unit quaternions up to the q / -q sign ambiguity."""
  dot = (actual * expected).sum(dim=-1).abs()
  torch.testing.assert_close(dot, torch.ones_like(dot), atol=1e-5, rtol=0)


def test_two_link_planar_arm_matches_trig() -> None:
  theta, phi = 0.6, -0.9
  root_pos = torch.zeros(1, 1, 3)
  root_quat = torch.tensor([_IDENTITY_QUAT]).reshape(1, 1, 4)
  angles = torch.tensor([[theta, phi, 0.0]])
  positions, _ = hinge_chain_body_positions(
    root_pos, root_quat, joint_angles=angles, **_chain_kwargs()
  )
  length_one, length_two = 0.4, 0.3
  expected_shoulder = torch.zeros(3)
  expected_elbow = torch.tensor(
    [length_one * math.cos(theta), length_one * math.sin(theta), 0.0]
  )
  expected_wrist = expected_elbow + torch.tensor(
    [
      length_two * math.cos(theta + phi),
      length_two * math.sin(theta + phi),
      0.0,
    ]
  )
  torch.testing.assert_close(positions[0, 0], expected_shoulder, atol=1e-5, rtol=0)
  torch.testing.assert_close(positions[0, 1], expected_elbow, atol=1e-5, rtol=0)
  torch.testing.assert_close(positions[0, 2], expected_wrist, atol=1e-5, rtol=0)


def test_offset_anchor_rotates_body_origin_about_anchor() -> None:
  angle = 0.8
  kwargs = _chain_kwargs(
    parent_chain_index=(-1,),
    parent_root_slot=(0,),
    body_pos_l=torch.tensor([[1.0, 0.0, 0.0]]),
    body_quat_l=torch.tensor([_IDENTITY_QUAT]),
    joint_pos_l=torch.tensor([[-1.0, 0.0, 0.0]]),
    joint_axis_l=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  positions, quaternions = hinge_chain_body_positions(
    torch.zeros(1, 1, 3),
    torch.tensor([_IDENTITY_QUAT]).reshape(1, 1, 4),
    joint_angles=torch.tensor([[angle]]),
    **kwargs,
  )
  expected = torch.tensor([math.cos(angle), math.sin(angle), 0.0])
  torch.testing.assert_close(positions[0, 0], expected, atol=1e-5, rtol=0)
  expected_quat = quat_from_angle_axis(
    torch.tensor([angle]), torch.tensor([[0.0, 0.0, 1.0]])
  )
  _assert_quat_close(quaternions[0, 0], expected_quat[0])


def test_matches_mujoco_kinematics() -> None:
  xml = """
  <mujoco>
    <compiler angle="radian"/>
    <worldbody>
      <body name="base" pos="0.1 0.2 0.3" euler="0.3 -0.2 0.5">
        <body name="upper" pos="0.05 -0.02 0.11" euler="0.1 0.2 -0.3">
          <joint name="j1" type="hinge" axis="0 1 0" pos="0.02 0.01 -0.03"/>
          <geom type="sphere" size="0.02" mass="0.1"/>
          <body name="lower" pos="0.31 0.0 -0.02" euler="-0.2 0.1 0.15">
            <joint name="j2" type="hinge" axis="0 0 1" pos="0.0 0.015 0.0"/>
            <geom type="sphere" size="0.02" mass="0.1"/>
            <body name="tip" pos="0.24 0.02 0.0">
              <joint name="j3" type="hinge" axis="1 0 0" pos="0.01 0 0"/>
              <geom type="sphere" size="0.02" mass="0.1"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  chain_names = ("upper", "lower", "tip")
  body_ids = [model.body(name).id for name in chain_names]
  joint_ids = [model.joint(name).id for name in ("j1", "j2", "j3")]
  base_id = model.body("base").id
  # A raw and a "filtered" pose: both must match mj_kinematics exactly.
  for qpos in (np.array([0.7, -0.4, 0.3]), np.array([1.2, 0.35, -0.8])):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    positions, quaternions = hinge_chain_body_positions(
      torch.tensor(data.xpos[base_id], dtype=torch.float32).reshape(1, 1, 3),
      torch.tensor(data.xquat[base_id], dtype=torch.float32).reshape(1, 1, 4),
      parent_chain_index=(-1, 0, 1),
      parent_root_slot=(0, 0, 0),
      body_pos_l=torch.tensor(model.body_pos[body_ids], dtype=torch.float32),
      body_quat_l=torch.tensor(model.body_quat[body_ids], dtype=torch.float32),
      joint_pos_l=torch.tensor(model.jnt_pos[joint_ids], dtype=torch.float32),
      joint_axis_l=torch.tensor(model.jnt_axis[joint_ids], dtype=torch.float32),
      joint_angles=torch.tensor(qpos, dtype=torch.float32).reshape(1, 3),
    )
    expected_pos = torch.tensor(data.xpos[body_ids], dtype=torch.float32)
    expected_quat = torch.tensor(data.xquat[body_ids], dtype=torch.float32)
    torch.testing.assert_close(positions[0], expected_pos, atol=1e-5, rtol=0)
    _assert_quat_close(quaternions[0], expected_quat)


def test_zero_delta_gives_zero_displacement() -> None:
  angles = torch.tensor([[0.3, -0.2, 0.9], [0.0, 1.1, -0.4]])
  root_pos = torch.randn(2, 1, 3)
  root_quat = torch.tensor([_IDENTITY_QUAT]).reshape(1, 1, 4).expand(2, 1, 4)
  kwargs = _chain_kwargs()
  first_pos, first_quat = hinge_chain_body_positions(
    root_pos, root_quat, joint_angles=angles, **kwargs
  )
  second_pos, second_quat = hinge_chain_body_positions(
    root_pos, root_quat, joint_angles=angles.clone(), **kwargs
  )
  assert torch.equal(first_pos, second_pos)
  assert torch.equal(first_quat, second_quat)


def _property_stub(*, enabled: bool) -> PlanarFilteredReplayMotionCommand:
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  generator = torch.Generator().manual_seed(7)
  raw = torch.randn(2, 5, 3, generator=generator)
  raw_quat = torch.nn.functional.normalize(
    torch.randn(2, 5, 4, generator=generator), dim=-1
  )
  command._raw_body_pos_w = lambda: raw.clone()
  command._raw_body_quat_w = lambda: raw_quat.clone()
  command._filter_initialized = torch.tensor([True, False])
  command._filtered_root_xy_w = torch.tensor([[0.5, -0.25], [1.0, 2.0]])
  raw_vel = torch.randn(2, 5, 3, generator=generator)
  command._raw_body_lin_vel_w = lambda: raw_vel.clone()
  command._filtered_root_velocity_xy_w = torch.tensor([[0.3, 0.1], [-0.2, 0.4]])
  command._propagate_targets = enabled
  if enabled:
    command._arm_body_target_offset_w = torch.zeros(2, 5, 3)
    command._arm_body_target_quat_delta_w = torch.zeros(2, 5, 4)
    command._arm_body_target_quat_delta_w[..., 0] = 1.0
    command._arm_body_target_lin_vel_w = torch.zeros(2, 5, 3)
    command._arm_prop_cloud_ids = torch.tensor([1, 3], dtype=torch.long)
  command._stub_raw = raw
  command._stub_raw_quat = raw_quat
  command._stub_raw_vel = raw_vel
  return command


def _expected_planar_only(command: PlanarFilteredReplayMotionCommand) -> torch.Tensor:
  raw = command._stub_raw.clone()
  offset_xy = command._filtered_root_xy_w - raw[:, 0, :2]
  offset_xy = torch.where(command._filter_initialized[:, None], offset_xy, 0.0)
  expected = raw.clone()
  expected[..., :2] += offset_xy[:, None, :]
  return expected


def _expected_planar_only_velocity(
  command: PlanarFilteredReplayMotionCommand,
) -> torch.Tensor:
  raw = command._stub_raw_vel.clone()
  delta_xy = command._filtered_root_velocity_xy_w - raw[:, 0, :2]
  delta_xy = torch.where(command._filter_initialized[:, None], delta_xy, 0.0)
  expected = raw.clone()
  expected[..., :2] += delta_xy[:, None, :]
  return expected


def test_body_lin_vel_default_off_bit_identical() -> None:
  command = _property_stub(enabled=False)
  assert torch.equal(command.body_lin_vel_w, _expected_planar_only_velocity(command))


def test_body_lin_vel_enabled_touches_only_arm_cloud_bodies() -> None:
  command = _property_stub(enabled=True)
  correction = torch.zeros(2, 5, 3)
  correction[0, 1] = torch.tensor([0.5, -0.2, 0.1])
  correction[1, 3] = torch.tensor([-0.3, 0.0, 0.7])
  command._arm_body_target_lin_vel_w = correction
  expected = _expected_planar_only_velocity(command)
  actual = command.body_lin_vel_w
  non_arm = torch.tensor([0, 2, 4])
  assert torch.equal(actual[:, non_arm], expected[:, non_arm])
  arm = torch.tensor([1, 3])
  torch.testing.assert_close(actual[:, arm], expected[:, arm] + correction[:, arm])
  # The cache is applied, never recomputed, and the raw accessor is untouched.
  assert torch.equal(command._raw_body_lin_vel_w(), command._stub_raw_vel)


def test_body_pose_default_off_bit_identical() -> None:
  command = _property_stub(enabled=False)
  assert torch.equal(command.body_pos_w, _expected_planar_only(command))
  assert torch.equal(command.body_quat_w, command._stub_raw_quat)


def test_body_pose_enabled_touches_only_arm_cloud_bodies() -> None:
  command = _property_stub(enabled=True)
  expected = _expected_planar_only(command)
  assert torch.equal(command.body_pos_w, expected)
  # Identity deltas: non-arm rows bit-exact, arm rows equal up to quat_mul fp.
  baseline_quat = command.body_quat_w
  for untouched in (0, 2, 4):
    assert torch.equal(
      baseline_quat[:, untouched], command._stub_raw_quat[:, untouched]
    )
  for touched in (1, 3):
    _assert_quat_close(baseline_quat[:, touched], command._stub_raw_quat[:, touched])

  offsets = torch.zeros(2, 5, 3)
  offsets[:, 1] = torch.tensor([0.1, -0.2, 0.3])
  offsets[:, 3] = torch.tensor([-0.4, 0.5, -0.6])
  command._arm_body_target_offset_w = offsets
  quat_delta = command._arm_body_target_quat_delta_w
  quat_delta[:, 1] = quat_from_angle_axis(
    torch.tensor([0.4, -0.7]), torch.tensor([[0.0, 0.0, 1.0]] * 2)
  )
  quat_delta[:, 3] = quat_from_angle_axis(
    torch.tensor([-0.2, 0.9]), torch.tensor([[0.0, 1.0, 0.0]] * 2)
  )
  adjusted_pos = command.body_pos_w
  adjusted_quat = command.body_quat_w
  for untouched in (0, 2, 4):
    assert torch.equal(adjusted_pos[:, untouched], expected[:, untouched])
    assert torch.equal(
      adjusted_quat[:, untouched], command._stub_raw_quat[:, untouched]
    )
  for touched in (1, 3):
    assert torch.equal(
      adjusted_pos[:, touched], expected[:, touched] + offsets[:, touched]
    )
    assert torch.equal(
      adjusted_quat[:, touched],
      quat_mul(quat_delta[:, touched], command._stub_raw_quat[:, touched]),
    )


def _update_stub(alignment_yaw_rad: float = 0.0) -> PlanarFilteredReplayMotionCommand:
  """Stub with distinct aligned (_raw_body_*) and unaligned (_motion_body_*)
  accessors so a regression reading the unaligned cloud is caught."""
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command._propagate_targets = True
  command._joint_filter_initialized = torch.tensor([True, True])
  command._arm_prop_chain_joint_ids = torch.tensor([0], dtype=torch.long)
  command._arm_prop_parent_chain_index = (-1,)
  command._arm_prop_parent_root_slot = (0,)
  command._arm_prop_body_pos_l = torch.tensor([[1.0, 0.0, 0.0]])
  command._arm_prop_body_quat_l = torch.tensor([_IDENTITY_QUAT])
  command._arm_prop_joint_pos_l = torch.tensor([[-1.0, 0.0, 0.0]])
  command._arm_prop_joint_axis_l = torch.tensor([[0.0, 1.0, 0.0]])
  command._arm_prop_root_cloud_ids = torch.tensor([0], dtype=torch.long)
  command._arm_prop_cloud_ids = torch.tensor([2], dtype=torch.long)
  command._arm_prop_cloud_chain_ids = torch.tensor([0], dtype=torch.long)
  command._arm_body_target_offset_w = torch.zeros(2, 4, 3)
  command._arm_body_target_quat_delta_w = torch.zeros(2, 4, 4)
  command._arm_body_target_quat_delta_w[..., 0] = 1.0
  command._arm_body_target_lin_vel_w = torch.zeros(2, 4, 3)
  command._arm_prop_velocity_hold = torch.zeros(2, dtype=torch.bool)
  command._arm_prop_step_dt = 0.02

  yaw = torch.tensor([alignment_yaw_rad])
  aligned_root_quat = quat_from_angle_axis(yaw, torch.tensor([[0.0, 0.0, 1.0]]))
  aligned_quat = torch.zeros(2, 4, 4)
  aligned_quat[:] = aligned_root_quat[0]
  unaligned_quat = torch.zeros(2, 4, 4)
  unaligned_quat[..., 0] = 1.0
  cloud_pos = torch.zeros(2, 4, 3)
  command._raw_body_pos_w = lambda: cloud_pos.clone()
  command._raw_body_quat_w = lambda: aligned_quat.clone()
  command._motion_body_pos_w = lambda: cloud_pos.clone()
  command._motion_body_quat_w = lambda: unaligned_quat.clone()
  command._stub_aligned_root_quat = aligned_root_quat[0]
  return command


def test_update_computes_displacement_and_property_never_recomputes(
  monkeypatch,
) -> None:
  command = _update_stub()
  raw_joint_pos = torch.tensor([[0.2, 0.0], [0.0, 0.0]])
  delta = torch.tensor([[0.5, 0.0], [0.0, 0.0]])
  command._filtered_joint_pos = raw_joint_pos + delta

  calls = {"count": 0}
  original = krc.hinge_chain_body_positions

  def counting(*args, **kwargs):
    calls["count"] += 1
    return original(*args, **kwargs)

  monkeypatch.setattr(krc, "hinge_chain_body_positions", counting)
  command._update_arm_body_target_offsets(raw_joint_pos)
  assert calls["count"] == 2

  offsets = command._arm_body_target_offset_w
  assert not torch.equal(offsets[0, 2], torch.zeros(3))
  assert torch.equal(offsets[1], torch.zeros(4, 3))
  assert torch.equal(offsets[0, torch.tensor([0, 1, 3])], torch.zeros(3, 3))
  quat_delta = command._arm_body_target_quat_delta_w
  identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
  assert torch.equal(quat_delta[1], identity.expand(4, 4))
  _assert_quat_close(
    quat_delta[0, 2],
    quat_from_angle_axis(
      torch.tensor([0.5]),
      quat_apply(
        command._stub_aligned_root_quat.reshape(1, 4),
        torch.tensor([[0.0, 1.0, 0.0]]),
      ),
    )[0],
  )

  command._filter_initialized = torch.tensor([True, True])
  command._filtered_root_xy_w = torch.zeros(2, 2)
  _ = command.body_pos_w
  _ = command.body_pos_w
  _ = command.body_quat_w
  assert calls["count"] == 2


def test_update_uses_aligned_reference_cloud() -> None:
  """A regression swapping _raw_body_* for _motion_body_* must fail here."""
  yaw = 1.1
  command = _update_stub(alignment_yaw_rad=yaw)
  raw_joint_pos = torch.tensor([[0.3, 0.0], [0.3, 0.0]])
  delta = torch.tensor([[0.6, 0.0], [0.6, 0.0]])
  command._filtered_joint_pos = raw_joint_pos + delta
  command._update_arm_body_target_offsets(raw_joint_pos)

  root_pos = torch.zeros(1, 1, 3)
  kwargs = {
    "parent_chain_index": command._arm_prop_parent_chain_index,
    "parent_root_slot": command._arm_prop_parent_root_slot,
    "body_pos_l": command._arm_prop_body_pos_l,
    "body_quat_l": command._arm_prop_body_quat_l,
    "joint_pos_l": command._arm_prop_joint_pos_l,
    "joint_axis_l": command._arm_prop_joint_axis_l,
  }
  aligned_root_quat = command._stub_aligned_root_quat.reshape(1, 1, 4)
  raw_pos, raw_quat = hinge_chain_body_positions(
    root_pos, aligned_root_quat, joint_angles=torch.tensor([[0.3]]), **kwargs
  )
  adj_pos, adj_quat = hinge_chain_body_positions(
    root_pos, aligned_root_quat, joint_angles=torch.tensor([[0.9]]), **kwargs
  )
  expected_offset = (adj_pos - raw_pos)[0, 0]
  expected_delta = quat_mul(adj_quat, krc.quat_inv(raw_quat))[0, 0]
  torch.testing.assert_close(
    command._arm_body_target_offset_w[0, 2], expected_offset, atol=1e-6, rtol=0
  )
  _assert_quat_close(command._arm_body_target_quat_delta_w[0, 2], expected_delta)

  # The unaligned cloud would produce a measurably different displacement.
  identity_root = torch.tensor([_IDENTITY_QUAT]).reshape(1, 1, 4)
  unaligned_raw, _ = hinge_chain_body_positions(
    root_pos, identity_root, joint_angles=torch.tensor([[0.3]]), **kwargs
  )
  unaligned_adj, _ = hinge_chain_body_positions(
    root_pos, identity_root, joint_angles=torch.tensor([[0.9]]), **kwargs
  )
  unaligned_offset = (unaligned_adj - unaligned_raw)[0, 0]
  assert (expected_offset - unaligned_offset).abs().max() > 1e-2


def test_update_skips_fk_below_threshold(monkeypatch) -> None:
  command = _update_stub()
  raw_joint_pos = torch.tensor([[0.2, 0.0], [0.0, 0.0]])
  command._filtered_joint_pos = raw_joint_pos + 5e-5
  command._arm_body_target_offset_w[0, 2] = 1.0
  command._arm_body_target_quat_delta_w[0, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0])

  def forbidden(*args, **kwargs):
    raise AssertionError("FK must not run for negligible residuals")

  monkeypatch.setattr(krc, "hinge_chain_body_positions", forbidden)
  command._update_arm_body_target_offsets(raw_joint_pos)
  assert torch.equal(command._arm_body_target_offset_w, torch.zeros(2, 4, 3))
  identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
  assert torch.equal(
    command._arm_body_target_quat_delta_w, identity.expand(2, 4, 4).clone()
  )


def _run_update(command: PlanarFilteredReplayMotionCommand, delta: torch.Tensor):
  raw_joint_pos = torch.tensor([[0.2, 0.0], [0.0, 0.0]])
  command._filtered_joint_pos = raw_joint_pos + delta
  command._update_arm_body_target_offsets(raw_joint_pos)
  return command._arm_body_target_offset_w.clone()


def test_velocity_is_zero_for_constant_displacement() -> None:
  command = _update_stub()
  delta = torch.tensor([[0.5, 0.0], [0.3, 0.0]])
  _run_update(command, delta)
  _run_update(command, delta)
  assert torch.equal(command._arm_body_target_lin_vel_w, torch.zeros(2, 4, 3))


def test_velocity_matches_finite_difference_on_arm_bodies_only() -> None:
  command = _update_stub()
  first = _run_update(command, torch.tensor([[0.2, 0.0], [0.4, 0.0]]))
  second = _run_update(command, torch.tensor([[0.5, 0.0], [0.1, 0.0]]))
  assert not torch.equal(first[:, 2], second[:, 2])
  expected = (second - first) / 0.02
  velocity = command._arm_body_target_lin_vel_w
  torch.testing.assert_close(velocity[:, 2], expected[:, 2])
  assert torch.equal(velocity[:, torch.tensor([0, 1, 3])], torch.zeros(2, 3, 3))
  assert not torch.equal(velocity[0, 2], torch.zeros(3))


def test_velocity_hold_suppresses_one_step_after_reset() -> None:
  """The hold flag (set by reset, wrap, and first initialization) must make the
  following update report zero velocity even though the displacement jumps."""
  command = _update_stub()
  _run_update(command, torch.tensor([[0.5, 0.0], [0.5, 0.0]]))
  command._arm_prop_velocity_hold[0] = True
  _run_update(command, torch.tensor([[0.0, 0.0], [0.1, 0.0]]))
  velocity = command._arm_body_target_lin_vel_w
  assert torch.equal(velocity[0], torch.zeros(4, 3))
  assert not torch.equal(velocity[1, 2], torch.zeros(3))
  # The hold is consumed: the next step differences normally again.
  assert not bool(command._arm_prop_velocity_hold.any())
  _run_update(command, torch.tensor([[0.3, 0.0], [0.1, 0.0]]))
  assert not torch.equal(command._arm_body_target_lin_vel_w[0, 2], torch.zeros(3))
  assert torch.equal(command._arm_body_target_lin_vel_w[1], torch.zeros(4, 3))


def test_velocity_differences_through_the_inactive_skip_path() -> None:
  """When the residual drops below the FK threshold the displacement snaps to
  zero; the velocity must reflect that real finite difference, not stale data."""
  command = _update_stub()
  first = _run_update(command, torch.tensor([[0.5, 0.0], [0.0, 0.0]]))
  _run_update(command, torch.tensor([[5e-5, 0.0], [0.0, 0.0]]))
  torch.testing.assert_close(
    command._arm_body_target_lin_vel_w[0, 2], -first[0, 2] / 0.02
  )
  assert torch.equal(command._arm_body_target_lin_vel_w[1], torch.zeros(4, 3))
