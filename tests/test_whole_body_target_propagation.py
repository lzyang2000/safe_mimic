"""Whole-body reference-frame FK propagation of joint corrections."""

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
  PlanarFilteredReplayMotionCommandCfg,
  hinge_chain_body_positions,
)

# Same three-hinge idea as tests/test_arm_target_propagation.py but with two
# branches off the base so one root serves two chains (legs + waist).
_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0.1 0.2 0.3" euler="0.3 -0.2 0.5">
      <body name="waist" pos="0.0 0.0 0.2" euler="0.05 0.0 0.0">
        <joint name="jw" type="hinge" axis="0 0 1" pos="0 0 -0.02"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
        <body name="torso" pos="0.0 0.0 0.15">
          <joint name="jt" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.02" mass="0.1"/>
        </body>
      </body>
      <body name="thigh" pos="0.0 -0.1 -0.05" euler="0.1 0.2 -0.3">
        <joint name="jh" type="hinge" axis="0 1 0" pos="0.02 0.01 -0.03"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
        <body name="shin" pos="0.0 0.0 -0.3" euler="-0.2 0.1 0.15">
          <joint name="jk" type="hinge" axis="0 1 0" pos="0.0 0.015 0.0"/>
          <geom type="sphere" size="0.02" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _assert_quat_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
  dot = (actual * expected).sum(dim=-1).abs()
  torch.testing.assert_close(dot, torch.ones_like(dot), atol=1e-5, rtol=0)


def test_branched_chain_from_one_root_matches_mujoco_kinematics() -> None:
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)
  chain_names = ("waist", "thigh", "torso", "shin")  # topological order
  body_ids = [model.body(name).id for name in chain_names]
  joint_ids = [model.joint(name).id for name in ("jw", "jh", "jt", "jk")]
  base_id = model.body("base").id
  for qpos in (np.array([0.4, -0.3, 0.7, 0.2]), np.array([-0.9, 0.5, -0.2, 1.1])):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    positions, quaternions = hinge_chain_body_positions(
      torch.tensor(data.xpos[base_id], dtype=torch.float32).reshape(1, 1, 3),
      torch.tensor(data.xquat[base_id], dtype=torch.float32).reshape(1, 1, 4),
      parent_chain_index=(-1, -1, 0, 1),
      parent_root_slot=(0, 0, 0, 0),
      body_pos_l=torch.tensor(model.body_pos[body_ids], dtype=torch.float32),
      body_quat_l=torch.tensor(model.body_quat[body_ids], dtype=torch.float32),
      joint_pos_l=torch.tensor(model.jnt_pos[joint_ids], dtype=torch.float32),
      joint_axis_l=torch.tensor(model.jnt_axis[joint_ids], dtype=torch.float32),
      # qpos is in MuJoCo joint-id order (XML depth-first); reorder to the
      # chain order used above.
      joint_angles=torch.tensor(
        qpos[model.jnt_qposadr[joint_ids]], dtype=torch.float32
      ).reshape(1, 4),
    )
    torch.testing.assert_close(
      positions[0],
      torch.tensor(data.xpos[body_ids], dtype=torch.float32),
      atol=1e-5,
      rtol=0,
    )
    _assert_quat_close(
      quaternions[0], torch.tensor(data.xquat[body_ids], dtype=torch.float32)
    )


def test_cfg_default_off_and_flag_guard() -> None:
  from safe_mimic.tasks.kinematic_replay_command import _validate_propagation_flags

  defaults = {
    f.name: f.default
    for f in PlanarFilteredReplayMotionCommandCfg.__dataclass_fields__.values()
  }
  assert defaults["propagate_joint_corrections_to_body_targets"] is False
  assert defaults["propagate_arm_corrections_to_body_targets"] is False
  with pytest.raises(ValueError, match="not both"):
    _validate_propagation_flags(arm=True, whole_body=True)
  assert _validate_propagation_flags(arm=False, whole_body=True) == (True, True)
  assert _validate_propagation_flags(arm=True, whole_body=False) == (True, False)
  assert _validate_propagation_flags(arm=False, whole_body=False) == (False, False)


def _bare_command(
  num_envs: int, num_bodies: int, anchor: int
) -> PlanarFilteredReplayMotionCommand:
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command.motion_anchor_body_index = anchor
  command._arm_body_target_offset_w = torch.zeros(num_envs, num_bodies, 3)
  command._arm_body_target_quat_delta_w = torch.zeros(num_envs, num_bodies, 4)
  command._arm_body_target_quat_delta_w[..., 0] = 1.0
  command._arm_body_target_lin_vel_w = torch.zeros(num_envs, num_bodies, 3)
  return command


def test_anchor_targets_follow_correction_only_in_whole_body_mode(
  monkeypatch,
) -> None:
  torch.manual_seed(0)
  raw_pos = torch.randn(2, 4, 3)
  raw_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4] * 2)
  raw_vel = torch.randn(2, 4, 3)
  command = _bare_command(2, 4, anchor=2)
  # Planar filter state present but inactive: isolates the FK correction.
  command._filter_initialized = torch.tensor([False, False])
  command._filtered_root_xy_w = raw_pos[:, 0, :2].clone()
  command._filtered_root_velocity_xy_w = raw_vel[:, 0, :2].clone()
  monkeypatch.setattr(command, "_raw_body_pos_w", lambda: raw_pos)
  monkeypatch.setattr(command, "_raw_body_quat_w", lambda: raw_quat)
  monkeypatch.setattr(command, "_raw_body_lin_vel_w", lambda: raw_vel)
  offset = torch.tensor([0.1, -0.2, 0.3])
  lin_vel = torch.tensor([1.0, 2.0, 3.0])
  command._arm_body_target_offset_w[:, 2] = offset
  command._arm_body_target_lin_vel_w[:, 2] = lin_vel
  yaw90 = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])
  command._arm_body_target_quat_delta_w[:, 2] = yaw90

  command._propagate_targets = True
  command._anchor_target_corrected = False  # arm-only mode
  torch.testing.assert_close(command.anchor_pos_w, raw_pos[:, 2])
  torch.testing.assert_close(command.anchor_quat_w, raw_quat[:, 2])
  torch.testing.assert_close(command.anchor_lin_vel_w, raw_vel[:, 2])

  command._anchor_target_corrected = True  # whole-body mode
  torch.testing.assert_close(command.anchor_pos_w, raw_pos[:, 2] + offset)
  torch.testing.assert_close(command.anchor_lin_vel_w, raw_vel[:, 2] + lin_vel)
  torch.testing.assert_close(
    command.anchor_quat_w, quat_mul(yaw90.expand(2, 4), raw_quat[:, 2])
  )


def test_anchor_correction_stacks_on_planar_offset(monkeypatch) -> None:
  raw_pos = torch.zeros(1, 3, 3)
  raw_pos[0, 1] = torch.tensor([1.0, 1.0, 1.0])
  raw_vel = torch.zeros(1, 3, 3)
  command = _bare_command(1, 3, anchor=1)
  command._filter_initialized = torch.tensor([True])
  command._filtered_root_xy_w = torch.tensor([[0.5, -0.5]])
  command._filtered_root_velocity_xy_w = torch.tensor([[2.0, 0.0]])
  monkeypatch.setattr(command, "_raw_body_pos_w", lambda: raw_pos)
  monkeypatch.setattr(command, "_raw_body_lin_vel_w", lambda: raw_vel)
  command._arm_body_target_offset_w[:, 1] = torch.tensor([0.0, 0.0, -0.25])
  command._arm_body_target_lin_vel_w[:, 1] = torch.tensor([0.0, 0.0, 0.75])
  command._propagate_targets = True
  command._anchor_target_corrected = True
  torch.testing.assert_close(
    command.anchor_pos_w, torch.tensor([[1.5, 0.5, 0.75]])
  )
  torch.testing.assert_close(
    command.anchor_lin_vel_w, torch.tensor([[2.0, 0.0, 0.75]])
  )
