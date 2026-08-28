"""Fit a compact, MJLab-compatible capsule proxy to SOMA joint positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CapsuleSpec:
  """Definition of one proxy geom."""

  name: str
  start_joint: str
  end_joint: str | None
  radius_m: float


@dataclass(frozen=True)
class CapsuleFit:
  """Per-frame mocap poses and static MuJoCo geom sizes."""

  names: tuple[str, ...]
  centers_m: np.ndarray
  quaternions_wxyz: np.ndarray
  radii_m: np.ndarray
  half_lengths_m: np.ndarray
  root_path_m: np.ndarray


@dataclass(frozen=True)
class CapsuleDomainRandomization:
  """Per-human shape and collision-envelope randomization."""

  body_scale_xyz: np.ndarray
  radius_scale: float
  radius_margin_m: float


SOMA_CAPSULE_SPECS = (
  CapsuleSpec("pelvis", "LeftLeg", "RightLeg", 0.095),
  CapsuleSpec("torso_lower", "Hips", "Spine2", 0.125),
  CapsuleSpec("torso_upper", "Spine2", "Chest", 0.145),
  CapsuleSpec("shoulders", "LeftArm", "RightArm", 0.080),
  CapsuleSpec("neck", "Chest", "Head", 0.070),
  CapsuleSpec("head", "Head", None, 0.120),
  CapsuleSpec("left_upper_arm", "LeftArm", "LeftForeArm", 0.055),
  CapsuleSpec("left_forearm", "LeftForeArm", "LeftHand", 0.045),
  CapsuleSpec("left_hand", "LeftHand", None, 0.065),
  CapsuleSpec("right_upper_arm", "RightArm", "RightForeArm", 0.055),
  CapsuleSpec("right_forearm", "RightForeArm", "RightHand", 0.045),
  CapsuleSpec("right_hand", "RightHand", None, 0.065),
  CapsuleSpec("left_thigh", "LeftLeg", "LeftShin", 0.080),
  CapsuleSpec("left_shin", "LeftShin", "LeftFoot", 0.065),
  CapsuleSpec("left_foot", "LeftFoot", "LeftToeBase", 0.060),
  CapsuleSpec("right_thigh", "RightLeg", "RightShin", 0.080),
  CapsuleSpec("right_shin", "RightShin", "RightFoot", 0.065),
  CapsuleSpec("right_foot", "RightFoot", "RightToeBase", 0.060),
)

# Crowd members are distant, stationary obstacles rather than the primary
# interacting human. These five inflated capsules retain the gross animated
# silhouette without paying for all 18 limb segments in every crowd slot.
SOMA_CROWD_PROXY_SPECS = (
  CapsuleSpec("body_head", "Hips", "Head", 0.165),
  CapsuleSpec("left_arm", "LeftArm", "LeftHand", 0.075),
  CapsuleSpec("right_arm", "RightArm", "RightHand", 0.075),
  CapsuleSpec("left_leg", "LeftLeg", "LeftFoot", 0.085),
  CapsuleSpec("right_leg", "RightLeg", "RightFoot", 0.085),
)

# Measured directly from the bundled SOMA bind mesh. Runtime height targets use
# this physical reference instead of an arbitrary unit-scale assumption.
SOMA_BASE_MESH_HEIGHT_M = 1.76051475
DEFAULT_MIN_HUMAN_HEIGHT_M = 1.3
DEFAULT_MAX_HUMAN_HEIGHT_M = 1.9
MIN_CROSS_SECTION_PROPORTION = 0.88
MAX_CROSS_SECTION_PROPORTION = 1.12


def sample_body_scales_for_height(
  count: int,
  device: str | torch.device,
  *,
  min_height_m: float = DEFAULT_MIN_HUMAN_HEIGHT_M,
  max_height_m: float = DEFAULT_MAX_HUMAN_HEIGHT_M,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """Sample SOMA XYZ scales with an explicit standing-height range."""

  if count < 0:
    raise ValueError("body-scale sample count must be non-negative")
  if not 0.0 < min_height_m <= max_height_m:
    raise ValueError("human height range must be positive and ordered")
  torch_device = torch.device(device)
  target_height = torch.empty(count, device=torch_device).uniform_(
    min_height_m, max_height_m, generator=generator
  )
  global_scale = target_height / SOMA_BASE_MESH_HEIGHT_M
  body_scale = global_scale[:, None].expand(-1, 3).clone()
  cross_section_proportion = torch.empty((count, 2), device=torch_device).uniform_(
    MIN_CROSS_SECTION_PROPORTION,
    MAX_CROSS_SECTION_PROPORTION,
    generator=generator,
  )
  body_scale[:, :2] *= cross_section_proportion
  return body_scale


def _bvh_to_mujoco(points: np.ndarray) -> np.ndarray:
  """Convert BVH X-right/Y-up/Z-forward to MuJoCo X/Y/Z-up."""

  return np.stack((points[..., 0], points[..., 2], points[..., 1]), axis=-1)


def _quaternion_from_z(vectors: np.ndarray) -> np.ndarray:
  lengths = np.linalg.norm(vectors, axis=-1)
  unit = vectors / np.maximum(lengths[:, None], 1e-12)
  dot = np.clip(unit[:, 2], -1.0, 1.0)
  quaternions = np.column_stack(
    (
      1.0 + dot,
      -unit[:, 1],
      unit[:, 0],
      np.zeros(len(unit)),
    )
  )
  opposite = dot < -0.999999
  quaternions[opposite] = (0.0, 1.0, 0.0, 0.0)
  quaternions /= np.maximum(np.linalg.norm(quaternions, axis=-1, keepdims=True), 1e-12)
  return quaternions


def fit_soma_capsules(
  joint_names: tuple[str, ...],
  positions_m: np.ndarray,
  randomization: CapsuleDomainRandomization | None = None,
) -> CapsuleFit:
  """Fit the 18-geom human proxy to global SOMA joint positions."""

  indices = {name: index for index, name in enumerate(joint_names)}
  points = _bvh_to_mujoco(positions_m)
  if randomization is not None:
    points = points * randomization.body_scale_xyz
  frame_count = len(points)
  geom_count = len(SOMA_CAPSULE_SPECS)
  centers = np.empty((frame_count, geom_count, 3), dtype=np.float64)
  quaternions = np.zeros((frame_count, geom_count, 4), dtype=np.float64)
  quaternions[..., 0] = 1.0
  radii = np.asarray([spec.radius_m for spec in SOMA_CAPSULE_SPECS], dtype=np.float64)
  if randomization is not None:
    cross_section_scale = float(np.mean(randomization.body_scale_xyz[:2]))
    radii = (
      radii * cross_section_scale * randomization.radius_scale
      + randomization.radius_margin_m
    )
  half_lengths = np.zeros(geom_count, dtype=np.float64)

  for geom_index, spec in enumerate(SOMA_CAPSULE_SPECS):
    start = points[:, indices[spec.start_joint]]
    if spec.end_joint is None:
      centers[:, geom_index] = start
      continue
    end = points[:, indices[spec.end_joint]]
    vectors = end - start
    centers[:, geom_index] = 0.5 * (start + end)
    quaternions[:, geom_index] = _quaternion_from_z(vectors)
    half_lengths[geom_index] = 0.5 * np.median(np.linalg.norm(vectors, axis=-1))

  root_path = points[:, indices["Hips"]].copy()
  return CapsuleFit(
    names=tuple(spec.name for spec in SOMA_CAPSULE_SPECS),
    centers_m=centers,
    quaternions_wxyz=quaternions,
    radii_m=radii,
    half_lengths_m=half_lengths,
    root_path_m=root_path,
  )


def sample_capsule_domain_randomization(
  rng: np.random.Generator,
  *,
  min_height_m: float = DEFAULT_MIN_HUMAN_HEIGHT_M,
  max_height_m: float = DEFAULT_MAX_HUMAN_HEIGHT_M,
) -> CapsuleDomainRandomization:
  """Sample conservative human-proxy domain randomization for training/debug."""

  if not 0.0 < min_height_m <= max_height_m:
    raise ValueError("human height range must be positive and ordered")
  global_scale = rng.uniform(min_height_m, max_height_m) / SOMA_BASE_MESH_HEIGHT_M

  return CapsuleDomainRandomization(
    body_scale_xyz=np.asarray(
      [
        global_scale
        * rng.uniform(MIN_CROSS_SECTION_PROPORTION, MAX_CROSS_SECTION_PROPORTION),
        global_scale
        * rng.uniform(MIN_CROSS_SECTION_PROPORTION, MAX_CROSS_SECTION_PROPORTION),
        global_scale,
      ],
      dtype=np.float64,
    ),
    radius_scale=float(rng.uniform(0.92, 1.08)),
    radius_margin_m=float(rng.uniform(0.0, 0.025)),
  )
