from types import SimpleNamespace

import pytest
import torch

from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
)
from safe_mimic.tasks.reference_filter import (
  LinkCbfReferenceFilterCfg,
  PlanarCbfReferenceFilterCfg,
  arm_posture_velocity_candidates,
  filter_link_velocities,
  filter_planar_velocity,
  gate_joint_recovery_during_posture,
  joint_position_residual_limits,
  planar_capsule_geometry,
  project_preferred_joint_velocity_to_cbf,
  safe_standing_joint_velocity,
  select_lookahead_arm_posture_velocity,
  update_posture_hold_time,
)


def test_link_filter_broad_phase_keeps_nearest_groups_and_full_primary() -> None:
  command = SimpleNamespace(
    _obstacle_geom_counts=(6, 2),
    _link_filter_capsules_per_group=(2, None),
    _link_filter_nearest_groups=(2, None),
    num_envs=1,
    device=torch.device("cpu"),
  )
  x = torch.arange(8, dtype=torch.float32)
  centers = torch.zeros(1, 8, 3)
  centers[0, :, 0] = x
  quaternions = torch.zeros(1, 8, 4)
  quaternions[..., 0] = 1.0
  sizes = torch.ones(1, 8, 3)
  velocities = centers + 10.0
  clearance = torch.tensor([[10.0, 9.0, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0]])
  active = torch.ones(1, 8, dtype=torch.bool)

  selected = PlanarFilteredReplayMotionCommand._select_link_filter_obstacles(
    command,
    centers,
    quaternions,
    sizes,
    velocities,
    clearance,
    active,
  )

  selected_centers, selected_quaternions, selected_sizes, selected_velocities = (
    selected
  )
  assert set(selected_centers[0, :4, 0].tolist()) == {2.0, 3.0, 4.0, 5.0}
  assert selected_centers[0, 4:, 0].tolist() == [6.0, 7.0]
  torch.testing.assert_close(selected_quaternions[..., 0], torch.ones(1, 6))
  torch.testing.assert_close(selected_sizes, torch.ones(1, 6, 3))
  torch.testing.assert_close(selected_velocities, selected_centers + 10.0)


def test_joint_residual_limits_widen_only_arm_joints() -> None:
  cfg = LinkCbfReferenceFilterCfg(body_names=("test_link",))

  limits = joint_position_residual_limits(
    cfg,
    (
      "left_shoulder_pitch_joint",
      "right_elbow_joint",
      "left_wrist_yaw_joint",
      "waist_yaw_joint",
      "left_hip_pitch_joint",
      "right_ankle_roll_joint",
    ),
    "cpu",
  )
  torch.testing.assert_close(
    limits,
    torch.tensor([1.5, 1.5, 1.5, 0.75, 0.75, 0.75]),
  )


def test_lookahead_selects_arm_down_when_it_improves_future_clearance() -> None:
  cfg = LinkCbfReferenceFilterCfg(
    body_names=("left_wrist_yaw_link",),
    standing_posture_gain=1.0,
    max_standing_velocity_correction_rps=1.0,
    posture_lookahead_s=0.5,
  )
  candidates = arm_posture_velocity_candidates(
    cfg,
    joint_names=("left_shoulder_roll_joint",),
    joint_pos=torch.zeros(1, 1),
    standing_joint_pos=-torch.ones(1, 1),
  )
  candidate_links = torch.tensor(
    [[[[0.0, 0.0, 0.0]], [[0.0, 0.0, -0.5]], [[0.0, 0.0, -0.5]], [[0.0, 0.0, 0.0]]]]
  )
  common = {
    "cfg": cfg,
    "link_names": cfg.body_names,
    "candidate_joint_velocities": candidates,
    "candidate_link_positions_w": candidate_links,
    "nearest_obstacle_ids": torch.zeros(1, 1, dtype=torch.long),
    "capsule_centers_w": torch.tensor([[[0.5, 0.0, 0.0]]]),
    "capsule_quaternions_w": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    "capsule_sizes": torch.tensor([[[0.1, 0.0, 0.0]]]),
    "obstacle_velocities_w": torch.zeros(1, 1, 3),
  }

  selected = select_lookahead_arm_posture_velocity(
    link_active=torch.tensor([[True]]), **common
  )
  safe = select_lookahead_arm_posture_velocity(
    **{
      **common,
      "link_active": torch.tensor([[False]]),
      "capsule_centers_w": torch.tensor([[[2.0, 0.0, 0.0]]]),
    }
  )

  assert selected.item() == pytest.approx(-1.0)
  assert safe.item() == pytest.approx(0.0)


def test_arm_posture_candidates_keep_left_and_right_choices_independent() -> None:
  cfg = LinkCbfReferenceFilterCfg(
    body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    standing_posture_gain=1.0,
    max_standing_velocity_correction_rps=1.0,
  )
  candidates = arm_posture_velocity_candidates(
    cfg,
    joint_names=(
      "left_shoulder_roll_joint",
      "right_shoulder_roll_joint",
      "left_shoulder_pitch_joint",
      "left_elbow_joint",
      "right_wrist_yaw_joint",
      "waist_yaw_joint",
    ),
    joint_pos=torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0, 1.0]]),
    standing_joint_pos=torch.zeros(1, 6),
  )

  torch.testing.assert_close(
    candidates,
    torch.tensor(
      [
        [
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [-1.0, 1.0, -1.0, -1.0, 1.0, 0.0],
          [-1.0, 0.0, -1.0, -1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]
      ]
    ),
  )


def test_cached_posture_masks_preserve_candidate_outputs() -> None:
  cfg = LinkCbfReferenceFilterCfg(
    body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    standing_posture_gain=1.0,
    max_standing_velocity_correction_rps=1.0,
  )
  joint_names = (
    "left_shoulder_roll_joint",
    "right_elbow_joint",
    "waist_yaw_joint",
  )
  joint_pos = torch.tensor([[0.7, -0.4, 0.2], [-0.2, 0.9, -0.5]])
  standing_joint_pos = torch.zeros_like(joint_pos)
  derived = arm_posture_velocity_candidates(
    cfg,
    joint_names=joint_names,
    joint_pos=joint_pos,
    standing_joint_pos=standing_joint_pos,
  )
  posture_mask = torch.tensor([True, True, False])
  cached = arm_posture_velocity_candidates(
    cfg,
    joint_names=joint_names,
    joint_pos=joint_pos,
    standing_joint_pos=standing_joint_pos,
    posture_joint_mask=posture_mask,
    left_joint_mask=torch.tensor([True, False, False]),
    right_joint_mask=torch.tensor([False, True, False]),
  )

  torch.testing.assert_close(cached, derived, rtol=0.0, atol=0.0)


def test_cached_arm_link_mask_preserves_lookahead_output() -> None:
  cfg = LinkCbfReferenceFilterCfg(
    body_names=("left_wrist_yaw_link", "torso_link"),
    standing_posture_gain=1.0,
    max_standing_velocity_correction_rps=1.0,
  )
  candidates = arm_posture_velocity_candidates(
    cfg,
    joint_names=("left_shoulder_roll_joint",),
    joint_pos=torch.zeros(1, 1),
    standing_joint_pos=-torch.ones(1, 1),
  )
  common = {
    "cfg": cfg,
    "link_names": cfg.body_names,
    "candidate_joint_velocities": candidates,
    "candidate_link_positions_w": torch.tensor(
      [
        [
          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[0.0, 0.0, -0.5], [0.0, 0.0, 0.0]],
          [[0.0, 0.0, -0.5], [0.0, 0.0, 0.0]],
          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
      ]
    ),
    "nearest_obstacle_ids": torch.zeros(1, 2, dtype=torch.long),
    "link_active": torch.tensor([[True, False]]),
    "capsule_centers_w": torch.tensor([[[0.5, 0.0, 0.0]]]),
    "capsule_quaternions_w": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    "capsule_sizes": torch.tensor([[[0.1, 0.0, 0.0]]]),
    "obstacle_velocities_w": torch.zeros(1, 1, 3),
  }
  derived = select_lookahead_arm_posture_velocity(**common)
  cached = select_lookahead_arm_posture_velocity(
    **common,
    arm_link_mask=torch.tensor([True, False]),
  )

  torch.testing.assert_close(cached, derived, rtol=0.0, atol=0.0)


def test_default_link_and_posture_activation_share_proactive_margin() -> None:
  cfg = LinkCbfReferenceFilterCfg(body_names=("left_wrist_yaw_link",))

  assert cfg.activation_clearance_m == pytest.approx(0.8)
  assert cfg.posture_activation_clearance_m == pytest.approx(0.8)


def test_active_posture_is_not_cancelled_by_reference_recovery() -> None:
  recovery = torch.tensor([[0.8, -0.6, 0.4]])
  preferred = torch.tensor([[-1.0, 0.0, 0.5]])

  gated = gate_joint_recovery_during_posture(recovery, preferred)

  torch.testing.assert_close(gated, torch.tensor([[0.0, -0.6, 0.0]]))


def test_posture_hold_is_independent_and_batched_per_side() -> None:
  remaining = torch.tensor([[0.8, 0.0], [0.1, 0.4]])
  selected = torch.tensor([[False, True], [False, False]])

  updated = update_posture_hold_time(
    remaining,
    selected,
    step_dt=0.1,
    hold_s=1.0,
  )

  torch.testing.assert_close(updated, torch.tensor([[0.7, 1.0], [0.0, 0.3]]))


def test_lookahead_wrist_improvement_is_not_masked_by_unchanged_shoulder() -> None:
  cfg = LinkCbfReferenceFilterCfg(
    body_names=("left_shoulder_roll_link", "left_wrist_yaw_link"),
    standing_posture_gain=1.0,
    max_standing_velocity_correction_rps=1.0,
  )
  candidates = arm_posture_velocity_candidates(
    cfg,
    joint_names=("left_shoulder_roll_joint",),
    joint_pos=torch.zeros(1, 1),
    standing_joint_pos=-torch.ones(1, 1),
  )
  candidate_links = torch.tensor(
    [
      [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, -0.5]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, -0.5]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
      ]
    ]
  )
  selected = select_lookahead_arm_posture_velocity(
    cfg,
    link_names=cfg.body_names,
    candidate_joint_velocities=candidates,
    candidate_link_positions_w=candidate_links,
    nearest_obstacle_ids=torch.tensor([[0, 1]]),
    link_active=torch.tensor([[True, True]]),
    capsule_centers_w=torch.tensor([[[0.2, 0.0, 0.0], [1.3, 0.0, 0.0]]]),
    capsule_quaternions_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
    capsule_sizes=torch.tensor([[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
    obstacle_velocities_w=torch.zeros(1, 2, 3),
  )

  assert selected.item() == pytest.approx(-1.0)


def test_cbf_projection_retains_downward_preference_and_adds_outward_motion() -> None:
  cfg = LinkCbfReferenceFilterCfg(body_names=("left_wrist_yaw_link",))
  projected = project_preferred_joint_velocity_to_cbf(
    cfg,
    preferred_joint_velocity=torch.tensor([[0.0, -0.5]]),
    link_linear_jacobian=torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]]),
    link_normals_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
    required_joint_outward_speed_mps=torch.tensor([[0.2]]),
    link_active=torch.tensor([[True]]),
  )

  assert projected[0, 0].item() == pytest.approx(0.2, abs=1e-3)
  assert projected[0, 1].item() == pytest.approx(-0.5)


def _filter_cfg() -> PlanarCbfReferenceFilterCfg:
  return PlanarCbfReferenceFilterCfg(
    safe_clearance_m=0.65,
    cbf_alpha=2.0,
    max_planar_speed_mps=3.0,
    max_intervention_speed_mps=3.0,
    nearest_obstacles=1,
    projection_iterations=1,
  )


def test_planar_capsule_geometry_masks_inactive_below_scene() -> None:
  robot = torch.tensor([[0.0, 0.0, 0.0]])
  centers = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, -100.0]]])
  quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
  sizes = torch.tensor([[[0.2, 0.5, 0.0], [0.2, 0.5, 0.0]]])

  closest, clearance, active = planar_capsule_geometry(
    robot,
    centers,
    quaternions,
    sizes,
    robot_radius_m=0.3,
    vertical_gate_m=0.1,
  )

  assert closest[0, 0].tolist() == pytest.approx([1.0, 0.0])
  assert clearance[0, 0].item() == pytest.approx(0.5)
  assert active.tolist() == [[True, False]]


def test_planar_filter_leaves_safe_nominal_velocity_unchanged() -> None:
  result = filter_planar_velocity(
    _filter_cfg(),
    robot_position_xy_w=torch.tensor([[0.0, 0.0]]),
    nominal_velocity_xy_w=torch.tensor([[0.5, 0.0]]),
    closest_points_xy_w=torch.tensor([[[1.5, 0.0]]]),
    surface_clearances_m=torch.tensor([[1.0]]),
    obstacle_velocities_xy_w=torch.zeros(1, 1, 2),
    active=torch.tensor([[True]]),
  )

  assert result.velocity_w[0].tolist() == pytest.approx([0.5, 0.0])
  assert result.intervention_w[0].tolist() == pytest.approx([0.0, 0.0])
  assert result.maximum_cbf_violation_mps.item() == pytest.approx(0.0)


def test_planar_filter_adds_outward_velocity_inside_margin() -> None:
  result = filter_planar_velocity(
    _filter_cfg(),
    robot_position_xy_w=torch.tensor([[0.0, 0.0]]),
    nominal_velocity_xy_w=torch.tensor([[1.0, 0.0]]),
    closest_points_xy_w=torch.tensor([[[1.0, 0.0]]]),
    surface_clearances_m=torch.tensor([[0.2]]),
    obstacle_velocities_xy_w=torch.zeros(1, 1, 2),
    active=torch.tensor([[True]]),
  )

  assert result.velocity_w[0].tolist() == pytest.approx([-0.9, 0.0])
  assert result.intervention_w[0].tolist() == pytest.approx([-1.9, 0.0])
  assert result.maximum_cbf_violation_mps.item() == pytest.approx(0.0)


def test_planar_filter_accounts_for_obstacle_velocity() -> None:
  result = filter_planar_velocity(
    _filter_cfg(),
    robot_position_xy_w=torch.tensor([[0.0, 0.0]]),
    nominal_velocity_xy_w=torch.zeros(1, 2),
    closest_points_xy_w=torch.tensor([[[1.0, 0.0]]]),
    surface_clearances_m=torch.tensor([[0.65]]),
    obstacle_velocities_xy_w=torch.tensor([[[-0.5, 0.0]]]),
    active=torch.tensor([[True]]),
  )

  assert result.velocity_w[0].tolist() == pytest.approx([-0.5, 0.0])
  assert result.maximum_cbf_violation_mps.item() == pytest.approx(0.0)


def test_link_filter_adds_minimum_outward_velocity_correction() -> None:
  cfg = LinkCbfReferenceFilterCfg(
    body_names=("test_link",),
    safe_clearance_m=0.2,
    link_radius_m=0.1,
    activation_clearance_m=0.6,
  )
  result = filter_link_velocities(
    cfg,
    link_positions_w=torch.tensor([[[0.0, 0.0, 0.0]]]),
    link_velocities_w=torch.tensor([[[0.5, 0.0, 0.0]]]),
    capsule_centers_w=torch.tensor([[[0.4, 0.0, 0.0]]]),
    capsule_quaternions_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    capsule_sizes=torch.tensor([[[0.1, 0.5, 0.0]]]),
    obstacle_velocities_w=torch.zeros(1, 1, 3),
  )

  assert result.minimum_clearance_m.item() == pytest.approx(0.2)
  assert result.active.item()
  assert result.link_velocity_correction_w[0, 0].tolist() == pytest.approx(
    [-0.5, 0.0, 0.0]
  )


def test_link_filter_accounts_for_approaching_capsule_velocity() -> None:
  cfg = LinkCbfReferenceFilterCfg(
    body_names=("test_link",),
    safe_clearance_m=0.2,
    link_radius_m=0.1,
  )
  result = filter_link_velocities(
    cfg,
    link_positions_w=torch.tensor([[[0.0, 0.0, 0.0]]]),
    link_velocities_w=torch.zeros(1, 1, 3),
    capsule_centers_w=torch.tensor([[[0.4, 0.0, 0.0]]]),
    capsule_quaternions_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    capsule_sizes=torch.tensor([[[0.1, 0.5, 0.0]]]),
    obstacle_velocities_w=torch.tensor([[[-0.2, 0.0, 0.0]]]),
  )

  assert result.link_velocity_correction_w[0, 0].tolist() == pytest.approx(
    [-0.2, 0.0, 0.0]
  )


def test_standing_pull_is_threat_gated_and_cannot_move_link_inward() -> None:
  cfg = LinkCbfReferenceFilterCfg(body_names=("test_link",))
  common = {
    "cfg": cfg,
    "standing_joint_pos": torch.zeros(1, 1),
    "link_joint_ancestry": torch.ones(1, 1),
    "link_linear_jacobian": torch.tensor([[[[1.0, 0.0, 0.0]]]]),
    "link_normals_w": torch.tensor([[[1.0, 0.0, 0.0]]]),
    "link_clearance_m": torch.tensor([[0.1]]),
  }

  inactive = safe_standing_joint_velocity(
    joint_pos=torch.tensor([[1.0]]),
    link_active=torch.tensor([[False]]),
    **common,
  )
  inward = safe_standing_joint_velocity(
    joint_pos=torch.tensor([[1.0]]),
    link_active=torch.tensor([[True]]),
    **common,
  )
  outward = safe_standing_joint_velocity(
    joint_pos=torch.tensor([[-1.0]]),
    link_active=torch.tensor([[True]]),
    **common,
  )

  assert inactive.item() == pytest.approx(0.0)
  assert inward.item() == pytest.approx(0.0)
  assert outward.item() == pytest.approx(1.5)
