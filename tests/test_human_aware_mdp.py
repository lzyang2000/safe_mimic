from types import SimpleNamespace

import torch

from safe_mimic.assets.soma_capsules import HUMAN_INACTIVE_HEIGHT_M
from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommand,
)
from safe_mimic.tasks.mdp import (
  active_correction_joint_tracking_exp,
  active_correction_tracking,
  avoidance_conditioning_noise,
  avoidance_teacher_corrections,
  capsule_link_surface_clearances,
  human_capsule_collision,
  human_capsule_proximity_penalty,
  human_capsule_vectors_b,
  reference_filter_clearance_penalty,
  safe_planar_progress_reward,
  urgency_weighted_outward_progress,
  urgent_escape_progress_reward,
)


def test_avoidance_conditioning_noise_is_bounded_and_per_environment() -> None:
  noise = avoidance_conditioning_noise(
    SimpleNamespace(num_envs=8, device="cpu"),
    size=31,
  )

  assert noise.shape == (8, 31)
  assert torch.all(noise >= -1.0)
  assert torch.all(noise <= 1.0)


def test_avoidance_teacher_contains_planar_and_limb_filter_residuals() -> None:
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command._filtered_root_velocity_xy_w = torch.tensor([[0.0, 1.0]])
  command._raw_body_lin_vel_w = lambda: torch.tensor(  # type: ignore[method-assign]
    [[[1.0, 0.0, 0.0]]]
  )
  command._raw_joint_pos = lambda: torch.zeros(1, 3)  # type: ignore[method-assign]
  command._filtered_joint_pos = torch.tensor([[0.2, -0.3, 0.4]])
  command._joint_filter_initialized = torch.ones(1, dtype=torch.bool)
  command.robot_anchor_body_index = 0
  command.robot = SimpleNamespace(
    data=SimpleNamespace(body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))

  target = avoidance_teacher_corrections(env, "motion")

  torch.testing.assert_close(
    target,
    torch.tensor([[-1.0, 1.0, 0.2, -0.3, 0.4]]),
  )


def test_active_correction_tracking_zero_when_no_weights_active() -> None:
  joint_error = torch.tensor([[1.0, 2.0, 3.0]])
  weights = torch.zeros_like(joint_error)

  reward = active_correction_tracking(joint_error, weights, std=0.2)

  torch.testing.assert_close(reward, torch.tensor([0.0]))


def test_active_correction_tracking_is_one_at_zero_error() -> None:
  joint_error = torch.zeros((1, 3))
  weights = torch.tensor([[1.0, 0.0, 1.0]])

  reward = active_correction_tracking(joint_error, weights, std=0.2)

  torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_active_correction_tracking_falls_monotonically_with_error() -> None:
  weights = torch.ones((3, 2))
  joint_error = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.3, 0.3]])

  reward = active_correction_tracking(joint_error, weights, std=0.2)

  assert reward[0] > reward[1] > reward[2] > 0.0


def test_active_correction_tracking_ignores_inactive_joints() -> None:
  weights = torch.tensor([[1.0, 0.0]])
  small_error = torch.tensor([[0.05, 0.0]])
  # Same active-joint error, but the inactive joint carries a huge residual
  # that must not leak into the sum or the normalizing weight count.
  huge_inactive_error = torch.tensor([[0.05, 100.0]])

  reward_small = active_correction_tracking(small_error, weights, std=0.2)
  reward_huge = active_correction_tracking(huge_inactive_error, weights, std=0.2)

  torch.testing.assert_close(reward_small, reward_huge)


def test_active_correction_joint_tracking_exp_gates_by_teacher_residual() -> None:
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command._raw_joint_pos = lambda: torch.tensor([[0.0, 0.0]])  # type: ignore[method-assign]
  command._filtered_joint_pos = torch.tensor([[0.2, 0.01]])
  command._joint_filter_initialized = torch.ones(1, dtype=torch.bool)
  command.robot = SimpleNamespace(
    data=SimpleNamespace(joint_pos=torch.tensor([[0.2, 5.0]]))
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))

  reward = active_correction_joint_tracking_exp(
    env, "motion", std=0.2, activation_threshold_rad=0.05
  )

  # Joint 0's teacher residual (0.2) clears the 0.05 threshold and the robot
  # sits exactly at the filtered target -> zero weighted error. Joint 1's
  # residual (0.01) is inactive, so its 4.99 rad tracking error must not
  # leak into the reward.
  torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_active_correction_joint_tracking_exp_zero_without_active_joints() -> None:
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command._raw_joint_pos = lambda: torch.tensor([[0.0, 0.0]])  # type: ignore[method-assign]
  command._filtered_joint_pos = torch.tensor([[0.01, -0.01]])
  command._joint_filter_initialized = torch.ones(1, dtype=torch.bool)
  command.robot = SimpleNamespace(
    data=SimpleNamespace(joint_pos=torch.tensor([[5.0, -5.0]]))
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))

  reward = active_correction_joint_tracking_exp(
    env, "motion", std=0.2, activation_threshold_rad=0.05
  )

  torch.testing.assert_close(reward, torch.tensor([0.0]))


def test_capsule_link_clearance_detects_non_root_arm_overlap() -> None:
  link_positions = torch.tensor([[[0.0, 0.0, 0.8], [1.0, 0.0, 1.2]]])
  capsule_centers = torch.tensor([[[1.0, 0.0, 1.2]]])
  capsule_quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
  capsule_sizes = torch.tensor([[[0.15, 0.3, 0.0]]])

  clearance = capsule_link_surface_clearances(
    link_positions,
    capsule_centers,
    capsule_quaternions,
    capsule_sizes,
    link_radius_m=0.1,
  )

  assert clearance.shape == (1, 2, 1)
  assert clearance[0, 0, 0] > 0.5
  torch.testing.assert_close(clearance[0, 1, 0], torch.tensor(-0.25))


def _fake_env(
  centers_w: torch.Tensor,
  robot_positions_w: torch.Tensor,
  *,
  sizes: torch.Tensor | None = None,
  quaternions_wxyz: torch.Tensor | None = None,
) -> SimpleNamespace:
  env_count, capsule_count, _ = centers_w.shape
  if sizes is None:
    sizes = torch.zeros((env_count, capsule_count, 3))
  if quaternions_wxyz is None:
    quaternions_wxyz = torch.zeros((env_count, capsule_count, 4))
    quaternions_wxyz[..., 0] = 1.0
  robot_quaternions = torch.zeros((env_count, 4))
  robot_quaternions[:, 0] = 1.0
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=robot_positions_w,
      root_link_quat_w=robot_quaternions,
    )
  )
  human = SimpleNamespace(
    data=SimpleNamespace(
      geom_pos_w=centers_w,
      geom_quat_w=quaternions_wxyz,
    ),
    indexing=SimpleNamespace(geom_ids=torch.arange(capsule_count)),
  )
  return SimpleNamespace(
    scene={"robot": robot, "human": human},
    sim=SimpleNamespace(model=SimpleNamespace(geom_size=sizes)),
  )


def test_privileged_vectors_keep_all_capsules_of_nearest_people() -> None:
  centers = torch.tensor(
    [
      [
        [3.0, 0.0, 0.0],
        [3.0, 0.1, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.1, 0.0],
      ]
    ]
  )
  env = _fake_env(centers, torch.zeros((1, 3)))

  vectors = human_capsule_vectors_b(
    env,
    "robot",
    "human",
    max_distance=1.0,
    capsules_per_group=2,
    nearest_groups=2,
  )

  assert vectors.shape == (1, 12)
  assert torch.equal(vectors.reshape(1, 4, 3), centers[:, [2, 3, 4, 5]])


def test_nearest_eight_matches_exhaustive_proximity_in_dense_ring() -> None:
  people = 58
  capsules_per_person = 5
  angles = torch.arange(people) * (2.0 * torch.pi / people)
  radii = 3.0 + 3.0 * (torch.arange(people) % 7) / 6.0
  anchors = torch.stack(
    (radii * torch.cos(angles), radii * torch.sin(angles), torch.ones(people)),
    dim=-1,
  )
  local_offsets = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.35, 0.1],
      [0.0, -0.35, 0.1],
      [0.0, 0.12, -0.6],
      [0.0, -0.12, -0.6],
    ]
  )
  centers = (anchors[:, None] + local_offsets[None]).reshape(-1, 3)
  robot_positions = torch.tensor(
    [[0.0, 0.0, 0.8], [1.0, 0.0, 0.8], [-1.0, 1.0, 0.8], [2.5, 0.0, 0.8]]
  )
  centers = centers[None].expand(len(robot_positions), -1, -1).clone()
  sizes = torch.zeros((len(robot_positions), people * capsules_per_person, 3))
  sizes[..., 0] = 0.17
  sizes[..., 1] = 0.35
  env = _fake_env(centers, robot_positions, sizes=sizes)

  exhaustive = human_capsule_proximity_penalty(
    env,
    "robot",
    "human",
    safe_clearance=0.65,
    robot_radius=0.35,
  )
  nearest = human_capsule_proximity_penalty(
    env,
    "robot",
    "human",
    safe_clearance=0.65,
    robot_radius=0.35,
    capsules_per_group=capsules_per_person,
    nearest_groups=8,
  )

  assert torch.equal(nearest, exhaustive)


def test_capsule_collision_detects_overlap_but_ignores_inactive_height() -> None:
  centers = torch.tensor(
    [
      [[0.40, 0.0, 0.8]],
      [[1.00, 0.0, 0.8]],
      [[0.00, 0.0, -1000.0]],
    ]
  )
  sizes = torch.zeros((3, 1, 3))
  sizes[..., 0] = 0.10
  robot_positions = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [0.0, 0.0, 0.8]])
  env = _fake_env(centers, robot_positions, sizes=sizes)

  collision = human_capsule_collision(
    env,
    "robot",
    "human",
    robot_radius=0.35,
  )
  proximity = human_capsule_proximity_penalty(
    env,
    "robot",
    "human",
    safe_clearance=0.8,
    robot_radius=0.35,
  )

  assert collision.tolist() == [True, False, False]
  assert proximity[0] > 0
  assert proximity[1] > 0
  assert proximity[2] == 0


def test_safe_planar_progress_rewards_motion_in_escape_direction() -> None:
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command._filtered_root_velocity_xy_w = torch.tensor(
    [[0.5, 0.0], [0.5, 0.0], [0.05, 0.0]]
  )
  command.robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_lin_vel_w=torch.tensor(
        [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
      )
    )
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))

  reward = safe_planar_progress_reward(
    env,
    "motion",
    minimum_target_speed=0.1,
    normalization_speed=0.5,
  )

  torch.testing.assert_close(reward, torch.tensor([1.0, -1.0, 0.0]))


def test_urgency_weighted_outward_progress_gates_and_clamps() -> None:
  clearance = torch.tensor([2.0, 0.0, 10.0, 1.0, 0.5, -0.2])
  closing_speed = torch.tensor([1.0, 1.0, 1.0, 0.2, 1.0, 1.0])
  outward_speed = torch.tensor([0.5, 0.5, 0.5, 0.5, -1.0, 0.25])

  reward = urgency_weighted_outward_progress(
    clearance,
    closing_speed,
    outward_speed,
    ttc_horizon_s=2.5,
    normalization_speed=0.5,
    min_closing_speed_mps=0.3,
  )

  # Contact pace (zero or negative clearance) reaches full urgency; a slow or
  # receding obstacle is exactly zero; the horizon zeroes distant threats; the
  # outward-progress factor is clamped to [-1, 1] and carries the sign.
  torch.testing.assert_close(
    reward,
    torch.tensor([0.2, 1.0, 0.0, 0.0, -0.8, 0.5]),
  )


def _urgent_escape_env(
  *,
  primary_centers_w: torch.Tensor,
  command_obstacle_velocity_w: torch.Tensor,
  primary_slice: slice,
  robot_velocity_w: torch.Tensor,
) -> SimpleNamespace:
  """Stub a two-entity filter command: crowd rows precede the primary block."""
  env_count, capsule_count, _ = primary_centers_w.shape
  command = object.__new__(PlanarFilteredReplayMotionCommand)
  command._obstacle_entity_slices = {
    "human": slice(0, primary_slice.start),
    "primary_human": primary_slice,
  }
  command._obstacle_velocity_w = command_obstacle_velocity_w
  command.cfg = SimpleNamespace(
    planar_filter=SimpleNamespace(robot_radius_m=0.35, vertical_gate_m=1.0)
  )
  robot_positions = torch.zeros((env_count, 3))
  robot_positions[:, 2] = 0.8
  command.robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=robot_positions,
      root_link_lin_vel_w=robot_velocity_w,
    )
  )
  quaternions = torch.zeros((env_count, capsule_count, 4))
  quaternions[..., 0] = 1.0
  sizes = torch.zeros((env_count, capsule_count, 3))
  sizes[..., 0] = 0.1
  human = SimpleNamespace(
    data=SimpleNamespace(
      geom_pos_w=primary_centers_w, geom_quat_w=quaternions
    ),
    indexing=SimpleNamespace(geom_ids=torch.arange(capsule_count)),
  )
  return SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command),
    scene={"primary_human": human},
    sim=SimpleNamespace(model=SimpleNamespace(geom_size=sizes)),
  )


def test_urgent_escape_reward_pays_fleeing_from_closing_human_only() -> None:
  # Env 0: human 2 m ahead closing at 1.5 m/s while the robot flees at 0.5.
  # Env 1: same geometry but the human walks away.
  # Env 2: human parked at the real inactive height.
  primary_centers = torch.tensor(
    [
      [[2.0, 0.0, 0.8], [4.0, 0.0, 0.8]],
      [[2.0, 0.0, 0.8], [4.0, 0.0, 0.8]],
      [
        [0.5, 0.0, HUMAN_INACTIVE_HEIGHT_M],
        [0.6, 0.0, HUMAN_INACTIVE_HEIGHT_M],
      ],
    ]
  )
  # The command's velocity tensor covers crowd rows 0-1 plus the primary block
  # at rows 2-3. Crowd rows and the far primary capsule carry receding junk
  # velocities, so a wrong entity offset or a wrong nearest-capsule gather
  # reads a receding row and zeroes the env-0 reward.
  command_velocity = torch.tensor(
    [
      [
        [9.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
        [-1.5, 0.0, 0.0],
        [9.0, 0.0, 0.0],
      ],
      [
        [-9.0, 0.0, 0.0],
        [-9.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
      ],
      [
        [-9.0, 0.0, 0.0],
        [-9.0, 0.0, 0.0],
        [-1.5, 0.0, 0.0],
        [-1.5, 0.0, 0.0],
      ],
    ]
  )
  robot_velocity = torch.tensor(
    [[-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]
  )
  env = _urgent_escape_env(
    primary_centers_w=primary_centers,
    command_obstacle_velocity_w=command_velocity,
    primary_slice=slice(2, 4),
    robot_velocity_w=robot_velocity,
  )

  reward = urgent_escape_progress_reward(
    env,
    "motion",
    "primary_human",
    ttc_horizon_s=2.5,
    normalization_speed=0.5,
    min_closing_speed_mps=0.3,
  )

  # Env 0 selects the nearer capsule: clearance 2.0 - 0.1 - 0.35 = 1.55 m
  # (robot radius from the stub's planar-filter cfg), closing 1.0 m/s ->
  # urgency 1 - 1.55/2.5 = 0.38, outward 0.5/0.5 clamps to 1.0.
  torch.testing.assert_close(reward[0], torch.tensor(0.38))
  torch.testing.assert_close(reward[1], torch.tensor(0.0))
  torch.testing.assert_close(reward[2], torch.tensor(0.0))


def test_link_clearance_penalty_is_dense_bounded_and_ignores_infinity() -> None:
  command = SimpleNamespace(
    metrics={
      "link_filter_minimum_clearance_m": torch.tensor([torch.inf, 0.8, 0.4, -0.2])
    }
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))

  penalty = reference_filter_clearance_penalty(
    env,
    "motion",
    safe_clearance_m=0.8,
  )

  torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0, 0.25, 1.0]))
