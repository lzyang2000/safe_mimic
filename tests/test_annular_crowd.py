from math import pi

import pytest
import torch

from safe_mimic.motions.annular_crowd import sample_annular_crowd


def test_annular_crowd_varies_density_and_respects_radii() -> None:
  generator = torch.Generator().manual_seed(23)
  placement = sample_annular_crowd(
    64,
    12,
    "cpu",
    min_count=4,
    max_count=12,
    generator=generator,
  )

  assert placement.offsets_xy_m.shape == (64, 12, 2)
  assert placement.active.shape == (64, 12)
  assert placement.counts.min() >= 4
  assert placement.counts.max() <= 12
  assert len(torch.unique(placement.counts)) > 1
  assert torch.equal(placement.active.sum(dim=1), placement.counts)
  measured = torch.linalg.vector_norm(placement.offsets_xy_m, dim=-1)
  assert torch.allclose(measured, placement.radii_m, atol=1e-6)
  assert torch.all(measured >= 3.0)
  assert torch.all(measured <= 6.0)


def test_annular_crowd_can_face_every_member_toward_center() -> None:
  generator = torch.Generator().manual_seed(9)
  placement = sample_annular_crowd(
    3,
    8,
    "cpu",
    min_count=8,
    inward_facing_probability=1.0,
    inward_facing_jitter_rad=0.0,
    generator=generator,
  )

  expected = torch.atan2(
    -placement.offsets_xy_m[..., 1], -placement.offsets_xy_m[..., 0]
  )
  error = torch.atan2(
    torch.sin(placement.facing_yaw_rad - expected),
    torch.cos(placement.facing_yaw_rad - expected),
  )
  assert torch.allclose(error, torch.zeros_like(error), atol=1e-6)


def test_dense_ring_uses_center_to_center_arc_spacing() -> None:
  generator = torch.Generator().manual_seed(41)
  placement = sample_annular_crowd(
    128,
    64,
    "cpu",
    min_count=30,
    max_count=64,
    min_radius_m=3.0,
    max_radius_m=6.0,
    target_arc_spacing_m=0.62,
    radial_jitter_m=0.25,
    angular_jitter_fraction=0.0,
    generator=generator,
  )

  assert placement.counts.min() >= 30
  assert placement.counts.max() <= 64
  mean_radius = torch.stack(
    [
      placement.radii_m[index, : placement.counts[index]].mean()
      for index in range(len(placement.counts))
    ]
  )
  implied_spacing = 2.0 * pi * mean_radius / placement.counts
  assert torch.all(implied_spacing >= 0.58)
  assert torch.all(implied_spacing <= 0.70)
  for env_index, count in enumerate(placement.counts.tolist()):
    points = placement.offsets_xy_m[env_index, :count]
    neighbor_distance = torch.linalg.vector_norm(
      points - torch.roll(points, shifts=-1, dims=0), dim=-1
    )
    assert neighbor_distance.min() >= 0.60


def test_dense_ring_randomizes_from_empty_to_packed_capacity() -> None:
  kwargs = {
    "min_count": 0,
    "max_count": 58,
    "min_radius_m": 2.0,
    "max_radius_m": 4.0,
    "target_arc_spacing_m": 0.62,
    "radial_jitter_m": 0.25,
    "min_shape_exponent": 2.0,
    "max_shape_exponent": 8.0,
    "angular_jitter_fraction": 0.0,
  }
  placement = sample_annular_crowd(
    4096,
    58,
    "cpu",
    **kwargs,
    randomize_density=True,
    generator=torch.Generator().manual_seed(83),
  )
  packed = sample_annular_crowd(
    4096,
    58,
    "cpu",
    **kwargs,
    randomize_density=False,
    generator=torch.Generator().manual_seed(83),
  )

  assert placement.counts.min() == 0
  assert placement.counts.max() >= 40
  assert len(torch.unique(placement.counts)) >= 35
  assert torch.equal(placement.active.sum(dim=1), placement.counts)
  assert torch.all(placement.counts <= packed.counts)
  assert torch.any(placement.counts == packed.counts)


def test_superellipse_crowd_preserves_spacing_and_varies_radius() -> None:
  placement = sample_annular_crowd(
    8,
    58,
    "cpu",
    min_count=30,
    max_count=58,
    min_radius_m=3.0,
    max_radius_m=3.001,
    target_arc_spacing_m=0.62,
    radial_jitter_m=0.0,
    angular_jitter_fraction=0.0,
    min_shape_exponent=8.0,
    max_shape_exponent=8.0,
    generator=torch.Generator().manual_seed(67),
  )

  for env_index, count in enumerate(placement.counts.tolist()):
    points = placement.offsets_xy_m[env_index, :count]
    radii = torch.linalg.vector_norm(points, dim=-1)
    neighbor_distance = torch.linalg.vector_norm(
      points - torch.roll(points, shifts=-1, dims=0), dim=-1
    )
    assert radii.max() - radii.min() > 0.7
    assert neighbor_distance.min() >= 0.58
    assert neighbor_distance.max() <= 0.65


@pytest.mark.parametrize(
  "kwargs",
  (
    {"min_count": 5, "max_count": 4},
    {"min_radius_m": 6.0, "max_radius_m": 3.0},
    {"angular_jitter_fraction": 0.5},
    {"inward_facing_probability": 1.1},
    {"inward_facing_jitter_rad": pi + 0.1},
    {"target_arc_spacing_m": 0.0},
    {"radial_jitter_m": 2.0},
    {"min_shape_exponent": 0.9},
    {"min_shape_exponent": 4.0, "max_shape_exponent": 2.0},
  ),
)
def test_annular_crowd_rejects_invalid_configuration(
  kwargs: dict[str, float],
) -> None:
  with pytest.raises(ValueError):
    sample_annular_crowd(2, 4, "cpu", **kwargs)
