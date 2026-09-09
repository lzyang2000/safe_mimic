"""End-effector height termination that loosens only where the link filter acts."""

from types import SimpleNamespace

import pytest
import torch

from safe_mimic.tasks import mdp
from safe_mimic.tasks.kinematic_replay_command import PlanarFilteredReplayMotionCommand

JOINTS = (
  "left_hip_pitch_joint",
  "left_knee_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_knee_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
)
BODIES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def test_limb_joint_mask_maps_a_wrist_to_its_own_arm() -> None:
  mask = mdp.limb_joint_mask(JOINTS, "left_wrist_yaw_link")
  assert [JOINTS[i] for i in mask.nonzero(as_tuple=True)[0].tolist()] == [
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
  ]


def test_limb_joint_mask_maps_an_ankle_to_its_own_leg() -> None:
  mask = mdp.limb_joint_mask(JOINTS, "right_ankle_roll_link")
  assert [JOINTS[i] for i in mask.nonzero(as_tuple=True)[0].tolist()] == [
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
  ]


def test_limb_joint_mask_rejects_unknown_bodies() -> None:
  with pytest.raises(ValueError, match="torso_link"):
    mdp.limb_joint_mask(JOINTS, "torso_link")


def _stub(*, z_error, residual):
  """Two envs, four bodies; ``z_error`` is (env, body), ``residual`` (env, joint)."""
  z_error = torch.tensor(z_error, dtype=torch.float32)
  robot_pos = torch.zeros(z_error.shape[0], 4, 3)
  ref_pos = robot_pos.clone()
  ref_pos[..., 2] = z_error
  command = PlanarFilteredReplayMotionCommand.__new__(PlanarFilteredReplayMotionCommand)
  command.cfg = SimpleNamespace(body_names=BODIES)
  command.robot = SimpleNamespace(joint_names=JOINTS)
  command._filtered_joint_pos = torch.tensor(residual, dtype=torch.float32)
  command._raw_joint_pos = lambda: torch.zeros(len(residual), len(JOINTS))
  # Stub the properties the termination reads.
  type(command).__dict__  # noqa: B018 (documenting: properties come from the class)
  command.__dict__["_stub_ref"] = ref_pos
  command.__dict__["_stub_robot"] = robot_pos
  cls = type(
    "StubCommand",
    (PlanarFilteredReplayMotionCommand,),
    {
      "body_pos_relative_w": property(lambda self: self.__dict__["_stub_ref"]),
      "robot_body_pos_w": property(lambda self: self.__dict__["_stub_robot"]),
      "filtered_joint_pos": property(lambda self: self._filtered_joint_pos),
    },
  )
  command.__class__ = cls
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command), num_envs=2
  )
  return env


def _call(env):
  return mdp.bad_motion_body_pos_z_only_filter_gated(
    env,
    command_name="motion",
    threshold=0.25,
    loosened_threshold=0.5,
    activation_threshold_rad=0.05,
    body_names=BODIES,
  )


def test_strict_threshold_applies_while_the_filter_is_idle() -> None:
  env = _stub(
    z_error=[[0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.2, 0.0]], residual=[[0.0] * 13] * 2
  )
  assert _call(env).tolist() == [True, False]


def test_active_limb_uses_the_loosened_threshold() -> None:
  active_left_arm = [0.0] * 13
  active_left_arm[8] = 0.3  # left elbow residual
  env = _stub(
    z_error=[[0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.6, 0.0]],
    residual=[active_left_arm, active_left_arm],
  )
  # 0.3 m survives under the 0.5 m loosened bound; 0.6 m still terminates.
  assert _call(env).tolist() == [False, True]


def test_loosening_is_per_limb_not_global() -> None:
  active_left_arm = [0.0] * 13
  active_left_arm[8] = 0.3
  # Left arm active, but the RIGHT wrist is the one that is 0.3 m off.
  env = _stub(z_error=[[0.0, 0.0, 0.0, 0.3]], residual=[active_left_arm])
  assert _call(env).tolist() == [True]


def test_residual_below_activation_does_not_loosen() -> None:
  weak = [0.0] * 13
  weak[8] = 0.04
  env = _stub(z_error=[[0.0, 0.0, 0.3, 0.0]], residual=[weak])
  assert _call(env).tolist() == [True]


def test_leg_corrections_loosen_the_ankle_check() -> None:
  active_right_leg = [0.0] * 13
  active_right_leg[4] = -0.2  # right knee residual
  env = _stub(z_error=[[0.0, 0.3, 0.0, 0.0]], residual=[active_right_leg])
  assert _call(env).tolist() == [False]
