"""LiDAR observation adapters and sim-to-real corruptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.sensor import RayCastSensor
from mjlab.utils.noise import NoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def normalized_lidar_ranges(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_distance: float = 0.05,
) -> torch.Tensor:
  """Return ranges in ``[0, 1]``, using 1 for a ray miss.

  The output remains flat because mjlab's default MLP policy concatenates
  one-dimensional observation terms. The ordering is documented by
  :class:`SphericalLidarPatternCfg` and can later be reshaped for a CNN or
  transformer encoder.
  """
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, RayCastSensor):
    raise TypeError(f"Sensor '{sensor_name}' must be a RayCastSensor")

  distances = sensor.data.distances
  max_distance = sensor.cfg.max_distance
  distances = torch.where(
    distances < 0,
    torch.full_like(distances, max_distance),
    distances,
  )
  return distances.clamp(min=min_distance, max=max_distance) / max_distance


@dataclass(kw_only=True)
class LidarNoiseCfg(NoiseCfg):
  """Range noise, isolated-ray dropout, and contiguous sector dropout.

  Noise is specified in normalized range units. Sector dropout is the LiDAR
  analogue of RPL's random-side masking and teaches the policy not to rely on
  one permanently visible direction.
  """

  gaussian_std: float = 0.005
  ray_dropout_prob: float = 0.015
  sector_dropout_prob: float = 0.10
  sector_width_samples: int = 45
  azimuth_samples: int = 180
  miss_value: float = 1.0

  def __post_init__(self) -> None:
    if self.gaussian_std < 0:
      raise ValueError("gaussian_std must be non-negative")
    for name, value in (
      ("ray_dropout_prob", self.ray_dropout_prob),
      ("sector_dropout_prob", self.sector_dropout_prob),
    ):
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    if not 0 < self.sector_width_samples <= self.azimuth_samples:
      raise ValueError("sector_width_samples must be in [1, azimuth_samples]")

  def apply(self, data: torch.Tensor) -> torch.Tensor:
    if data.ndim != 2 or data.shape[1] % self.azimuth_samples != 0:
      raise ValueError(
        "LiDAR data must have shape [batch, elevations * azimuth_samples]"
      )

    output = data.clone()
    if self.gaussian_std > 0:
      output.add_(torch.randn_like(output) * self.gaussian_std)

    if self.ray_dropout_prob > 0:
      ray_dropout = torch.rand_like(output) < self.ray_dropout_prob
      output.masked_fill_(ray_dropout, self.miss_value)

    if self.sector_dropout_prob > 0:
      batch_size, num_rays = output.shape
      num_elevations = num_rays // self.azimuth_samples
      starts = torch.randint(
        self.azimuth_samples,
        (batch_size, 1),
        device=output.device,
      )
      azimuth = torch.arange(self.azimuth_samples, device=output.device).view(1, -1)
      sector = (azimuth - starts).remainder(self.azimuth_samples)
      sector = sector < self.sector_width_samples
      active = (
        torch.rand((batch_size, 1), device=output.device)
        < self.sector_dropout_prob
      )
      sector = (sector & active).view(batch_size, 1, self.azimuth_samples)
      sector = sector.expand(batch_size, num_elevations, self.azimuth_samples)
      output.view(batch_size, num_elevations, self.azimuth_samples).masked_fill_(
        sector, self.miss_value
      )

    return output
