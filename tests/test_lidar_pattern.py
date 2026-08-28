import math

import pytest

from safe_mimic.sensing.held_scan import InterleavedSphericalLidarPatternCfg
from safe_mimic.sensing.lidar import SphericalLidarPatternCfg


def test_pattern_shape_order_and_unit_length() -> None:
  pattern = SphericalLidarPatternCfg(
    azimuth_samples=4,
    elevation_angles_deg=(0.0, -45.0),
  )

  directions = pattern.local_directions()

  assert pattern.num_rays == 8
  assert directions[0] == pytest.approx((1.0, 0.0, 0.0))
  assert directions[1] == pytest.approx((0.0, 1.0, 0.0))
  assert directions[4][2] == pytest.approx(-math.sqrt(0.5))
  for direction in directions:
    assert math.sqrt(sum(component**2 for component in direction)) == pytest.approx(1.0)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"azimuth_samples": 3},
    {"elevation_angles_deg": ()},
    {"elevation_angles_deg": (-90.0,)},
  ],
)
def test_pattern_rejects_invalid_geometry(kwargs: dict[str, object]) -> None:
  with pytest.raises(ValueError):
    SphericalLidarPatternCfg(**kwargs)


def test_interleaved_pattern_phases_partition_full_scan() -> None:
  pattern = InterleavedSphericalLidarPatternCfg(
    azimuth_samples=10,
    elevation_angles_deg=(0.0, -20.0),
    phases=5,
  )

  phase_indices = [
    pattern.full_indices_for_phase(phase) for phase in range(pattern.phases)
  ]

  assert pattern.num_rays == 20
  assert pattern.rays_per_phase == 4
  assert phase_indices[0] == (0, 5, 10, 15)
  assert phase_indices[4] == (4, 9, 14, 19)
  assert sorted(index for phase in phase_indices for index in phase) == list(range(20))


def test_interleaved_pattern_pads_uneven_phases_without_losing_rays() -> None:
  pattern = InterleavedSphericalLidarPatternCfg(
    azimuth_samples=185,
    elevation_angles_deg=tuple(float(-index) for index in range(27)),
    phases=10,
  )

  assert pattern.num_rays == 4995
  assert pattern.rays_per_phase == 513
  all_indices = [
    index
    for phase in range(pattern.phases)
    for index in pattern.full_indices_for_phase(phase)
  ]
  assert sorted(all_indices) == list(range(pattern.num_rays))
  assert len(pattern.padded_full_indices_for_phase(5)) == 513
  assert len(pattern.valid_source_indices_for_phase(5)) == 486
