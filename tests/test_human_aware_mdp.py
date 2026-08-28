from types import SimpleNamespace

import torch

from safe_mimic.tasks.mdp import (
  human_capsule_proximity_penalty,
  human_capsule_vectors_b,
)


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
