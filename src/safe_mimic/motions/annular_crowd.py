"""GPU-vectorized placement sampling for stationary annular crowds."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch


@dataclass(frozen=True)
class AnnularCrowdPlacement:
  """One fixed world-space slot per possible crowd member."""

  offsets_xy_m: torch.Tensor
  radii_m: torch.Tensor
  azimuth_rad: torch.Tensor
  facing_yaw_rad: torch.Tensor
  active: torch.Tensor
  counts: torch.Tensor


def sample_annular_crowd(
  num_envs: int,
  capacity: int,
  device: str | torch.device,
  *,
  min_count: int = 4,
  max_count: int | None = None,
  min_radius_m: float = 3.0,
  max_radius_m: float = 6.0,
  angular_jitter_fraction: float = 0.25,
  inward_facing_probability: float = 0.7,
  inward_facing_jitter_rad: float = pi / 4.0,
  target_arc_spacing_m: float | None = None,
  randomize_density: bool = False,
  radial_jitter_m: float = 0.0,
  min_shape_exponent: float = 2.0,
  max_shape_exponent: float = 2.0,
  generator: torch.Generator | None = None,
) -> AnnularCrowdPlacement:
  """Sample roughly even crowd slots on a randomized annular boundary.

  Counts vary independently per environment. Active members use evenly spaced
  angular strata with bounded jitter, avoiding the severe clumping produced by
  independent azimuth samples. With a target arc spacing, the boundary's packed
  capacity is derived from its perimeter. ``randomize_density`` then samples an
  integer occupancy from ``min_count`` through that packed capacity, inclusive.
  A configurable fraction faces approximately inward; the rest receive an
  unconstrained random heading.
  """

  max_count = capacity if max_count is None else max_count
  if num_envs < 1 or capacity < 1:
    raise ValueError("num_envs and capacity must be positive")
  if not 0 <= min_count <= max_count <= capacity:
    raise ValueError("crowd count range must lie within capacity")
  if not 0.0 < min_radius_m < max_radius_m:
    raise ValueError("annular radii must be positive and increasing")
  if not 0.0 <= angular_jitter_fraction < 0.5:
    raise ValueError("angular jitter fraction must be in [0, 0.5)")
  if not 0.0 <= inward_facing_probability <= 1.0:
    raise ValueError("inward-facing probability must be in [0, 1]")
  if not 0.0 <= inward_facing_jitter_rad <= pi:
    raise ValueError("inward-facing jitter must be in [0, pi]")
  if target_arc_spacing_m is not None and target_arc_spacing_m <= 0.0:
    raise ValueError("target arc spacing must be positive")
  if not 0.0 <= radial_jitter_m < 0.5 * (max_radius_m - min_radius_m):
    raise ValueError("radial jitter is too large for the configured ring")
  if not 1.0 <= min_shape_exponent <= max_shape_exponent:
    raise ValueError("shape exponents must be ordered and at least one")

  torch_device = torch.device(device)
  ring_radius: torch.Tensor | None = None
  shaped_offsets: torch.Tensor | None = None
  shaped_radii: torch.Tensor | None = None
  shaped_azimuth: torch.Tensor | None = None
  if target_arc_spacing_m is None:
    counts = torch.randint(
      min_count,
      max_count + 1,
      (num_envs,),
      device=torch_device,
      generator=generator,
    )
  else:
    ring_radius = torch.empty(num_envs, device=torch_device).uniform_(
      min_radius_m,
      max_radius_m,
      generator=generator,
    )
    spacing_radius = (ring_radius - radial_jitter_m).clamp_min(min_radius_m)
    use_shaped_boundary = min_shape_exponent != 2.0 or max_shape_exponent != 2.0
    if use_shaped_boundary:
      exponent = torch.empty(num_envs, device=torch_device).uniform_(
        min_shape_exponent, max_shape_exponent, generator=generator
      )
      sample_count = 256
      theta = torch.arange(sample_count, device=torch_device) * (
        2.0 * pi / sample_count
      )
      cosine = torch.cos(theta)[None]
      sine = torch.sin(theta)[None]
      power = 2.0 / exponent[:, None]
      unit_points = torch.stack(
        (
          torch.sign(cosine) * torch.abs(cosine).pow(power),
          torch.sign(sine) * torch.abs(sine).pow(power),
        ),
        dim=-1,
      )
      unit_segments = torch.roll(unit_points, shifts=-1, dims=1) - unit_points
      unit_lengths = torch.linalg.vector_norm(unit_segments, dim=-1)
      unit_perimeter = unit_lengths.sum(dim=1)
      packed_counts = torch.floor(
        unit_perimeter * spacing_radius / target_arc_spacing_m
      ).long()
      packed_counts.clamp_(min=min_count, max=max_count)
      if randomize_density:
        counts = (
          torch.floor(
            torch.rand(num_envs, device=torch_device, generator=generator)
            * (packed_counts - min_count + 1)
          ).long()
          + min_count
        )
      else:
        counts = packed_counts

      points = unit_points * ring_radius[:, None, None]
      shape_rotation = torch.rand(
        (num_envs, 1), device=torch_device, generator=generator
      ) * (2.0 * pi)
      rotation_cos = torch.cos(shape_rotation)
      rotation_sin = torch.sin(shape_rotation)
      rotated = points.clone()
      rotated[..., 0] = rotation_cos * points[..., 0] - rotation_sin * points[..., 1]
      rotated[..., 1] = rotation_sin * points[..., 0] + rotation_cos * points[..., 1]
      points = rotated
      segments = torch.roll(points, shifts=-1, dims=1) - points
      segment_lengths = torch.linalg.vector_norm(segments, dim=-1)
      cumulative = torch.cat(
        (
          torch.zeros((num_envs, 1), device=torch_device),
          torch.cumsum(segment_lengths, dim=1),
        ),
        dim=1,
      )
      perimeter = cumulative[:, -1]
      slots_float = torch.arange(capacity, device=torch_device).float()[None]
      phase = torch.rand(num_envs, device=torch_device, generator=generator)
      target_distance = (
        phase[:, None] + slots_float / counts[:, None].clamp_min(1)
      ) * perimeter[:, None]
      if angular_jitter_fraction:
        arc_jitter = (
          torch.rand(
            (num_envs, capacity),
            device=torch_device,
            generator=generator,
          )
          * 2.0
          - 1.0
        ) * angular_jitter_fraction
        target_distance += (
          arc_jitter * perimeter[:, None] / counts[:, None].clamp_min(1)
        )
      target_distance = torch.remainder(target_distance, perimeter[:, None])
      segment_index = (
        torch.searchsorted(
          cumulative.contiguous(), target_distance.contiguous(), right=True
        )
        - 1
      )
      segment_index.clamp_(min=0, max=sample_count - 1)
      gather_xy = segment_index[..., None].expand(-1, -1, 2)
      segment_start = torch.gather(points, 1, gather_xy)
      segment_vector = torch.gather(segments, 1, gather_xy)
      start_distance = torch.gather(cumulative[:, :-1], 1, segment_index)
      selected_length = torch.gather(segment_lengths, 1, segment_index)
      blend = (target_distance - start_distance) / selected_length.clamp_min(1e-8)
      shaped_offsets = segment_start + blend[..., None] * segment_vector
      if radial_jitter_m:
        radial_jitter = (
          torch.rand(
            (num_envs, capacity),
            device=torch_device,
            generator=generator,
          )
          * 2.0
          - 1.0
        ) * radial_jitter_m
        radial_direction = shaped_offsets / torch.linalg.vector_norm(
          shaped_offsets, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        shaped_offsets += radial_jitter[..., None] * radial_direction
      shaped_radii = torch.linalg.vector_norm(shaped_offsets, dim=-1)
      shaped_azimuth = torch.atan2(shaped_offsets[..., 1], shaped_offsets[..., 0])
    else:
      packed_counts = torch.floor(
        (2.0 * pi / target_arc_spacing_m) * spacing_radius
      ).long()
      packed_counts.clamp_(min=min_count, max=max_count)
      if randomize_density:
        counts = (
          torch.floor(
            torch.rand(num_envs, device=torch_device, generator=generator)
            * (packed_counts - min_count + 1)
          ).long()
          + min_count
        )
      else:
        counts = packed_counts
    counts.clamp_(min=min_count, max=max_count)
  slots = torch.arange(capacity, device=torch_device)[None]
  active = slots < counts[:, None]

  if shaped_offsets is not None:
    assert shaped_radii is not None and shaped_azimuth is not None
    offsets = shaped_offsets
    radii = shaped_radii
    azimuth = shaped_azimuth
  else:
    base_rotation = torch.rand(
      (num_envs, 1), device=torch_device, generator=generator
    ) * (2.0 * pi)
    jitter = (
      torch.rand((num_envs, capacity), device=torch_device, generator=generator) * 2.0
      - 1.0
    ) * angular_jitter_fraction
    azimuth = base_rotation + (2.0 * pi) * (slots + jitter) / counts[:, None].clamp_min(
      1
    )
    azimuth = torch.remainder(azimuth, 2.0 * pi)

    if ring_radius is None:
      radius_squared = torch.empty((num_envs, capacity), device=torch_device).uniform_(
        min_radius_m**2,
        max_radius_m**2,
        generator=generator,
      )
      radii = torch.sqrt(radius_squared)
    else:
      radius_jitter = (
        torch.rand((num_envs, capacity), device=torch_device, generator=generator) * 2.0
        - 1.0
      ) * radial_jitter_m
      radii = (ring_radius[:, None] + radius_jitter).clamp(min_radius_m, max_radius_m)
    offsets = torch.stack(
      (radii * torch.cos(azimuth), radii * torch.sin(azimuth)), dim=-1
    )

  inward_yaw = torch.atan2(-offsets[..., 1], -offsets[..., 0])
  inward_yaw += (
    torch.rand((num_envs, capacity), device=torch_device, generator=generator) * 2.0
    - 1.0
  ) * inward_facing_jitter_rad
  random_yaw = (
    torch.rand((num_envs, capacity), device=torch_device, generator=generator) * 2.0
    - 1.0
  ) * pi
  face_inward = (
    torch.rand((num_envs, capacity), device=torch_device, generator=generator)
    < inward_facing_probability
  )
  facing_yaw = torch.where(face_inward, inward_yaw, random_yaw)
  facing_yaw = torch.atan2(torch.sin(facing_yaw), torch.cos(facing_yaw))

  return AnnularCrowdPlacement(
    offsets_xy_m=offsets,
    radii_m=radii,
    azimuth_rad=azimuth,
    facing_yaw_rad=facing_yaw,
    active=active,
    counts=counts,
  )
