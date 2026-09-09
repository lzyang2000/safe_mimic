"""LiDAR observation adapters and sim-to-real corruptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.sensor import RayCastSensor
from mjlab.utils.noise import NoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg

from safe_mimic.sensing.held_scan import HeldScanRayCastSensor


def _normalize_ranges(
  distances: torch.Tensor,
  *,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  distances = torch.where(
    distances < 0,
    torch.full_like(distances, max_distance),
    distances,
  )
  return distances.clamp(min=min_distance, max=max_distance) / max_distance


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

  return _normalize_ranges(
    sensor.data.distances,
    min_distance=min_distance,
    max_distance=sensor.cfg.max_distance,
  )


def directional_lidar_ranges(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  azimuth_samples: int,
  elevation_samples: int,
  azimuth_bins: int = 24,
  elevation_bins: int = 3,
  min_distance: float = 0.05,
) -> torch.Tensor:
  """Summarize every dense ray as directional minimum ranges for the critic.

  Adaptive minimum pooling keeps the closest return in each body-frame
  azimuth/elevation cell. The actor still receives both complete dense scans;
  this compact, noiseless value-function view avoids a very wide critic GEMM
  during every rollout step and PPO minibatch.
  """
  if azimuth_samples < azimuth_bins or elevation_samples < elevation_bins:
    raise ValueError("LiDAR pooling bins cannot exceed the source grid")
  ranges = normalized_lidar_ranges(
    env,
    sensor_name,
    min_distance=min_distance,
  )
  expected_rays = azimuth_samples * elevation_samples
  if ranges.shape[-1] != expected_rays:
    raise ValueError(f"LiDAR has {ranges.shape[-1]} rays; expected {expected_rays}")
  grid = ranges.reshape(-1, 1, elevation_samples, azimuth_samples)
  pooled = -F.adaptive_max_pool2d(
    -grid,
    output_size=(elevation_bins, azimuth_bins),
  )
  return pooled.flatten(start_dim=1)


class CachedDirectionalLidarRanges(ManagerTermBase):
  """Cache the critic's directional view for one complete scan period."""

  def __init__(
    self,
    cfg: ObservationTermCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    super().__init__(env)
    params = cfg.params
    self.sensor_name = str(params["sensor_name"])
    self.azimuth_samples = int(params["azimuth_samples"])
    self.elevation_samples = int(params["elevation_samples"])
    self.azimuth_bins = int(params.get("azimuth_bins", 24))
    self.elevation_bins = int(params.get("elevation_bins", 3))
    self.min_distance = float(params.get("min_distance", 0.05))
    sensor = env.scene[self.sensor_name]
    if not isinstance(sensor, HeldScanRayCastSensor):
      raise TypeError(
        f"Sensor '{self.sensor_name}' must be a HeldScanRayCastSensor"
      )
    self._sensor = sensor
    self._cached: torch.Tensor | None = None
    self._cached_publication_count = -1

  def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
    if (
      self._cached is None
      or self._cached_publication_count != self._sensor.global_publication_count
    ):
      self._cached = directional_lidar_ranges(
        env,
        self.sensor_name,
        azimuth_samples=self.azimuth_samples,
        elevation_samples=self.elevation_samples,
        azimuth_bins=self.azimuth_bins,
        elevation_bins=self.elevation_bins,
        min_distance=self.min_distance,
      )
      self._cached_publication_count = self._sensor.global_publication_count
    return self._cached

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if self._cached is None:
      return
    ids = slice(None) if env_ids is None else env_ids
    self._cached[ids] = 1.0


def normalized_held_lidar_scan_pair(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_distance: float = 0.05,
) -> torch.Tensor:
  """Return current and previous completed scans in normalized range units."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, HeldScanRayCastSensor):
    raise TypeError(f"Sensor '{sensor_name}' must be a HeldScanRayCastSensor")
  current = _normalize_ranges(
    sensor.data.distances,
    min_distance=min_distance,
    max_distance=sensor.cfg.max_distance,
  )
  previous = _normalize_ranges(
    sensor.previous_distances,
    min_distance=min_distance,
    max_distance=sensor.cfg.max_distance,
  )
  return torch.cat((current, previous), dim=-1)


def _directional_minimum_pool(
  ranges: torch.Tensor,
  *,
  scan_count: int,
  elevation_samples: int,
  azimuth_samples: int,
  elevation_bins: int,
  azimuth_bins: int,
) -> torch.Tensor:
  """Pool normalized scans into closest-return angular cells."""
  if azimuth_samples < azimuth_bins or elevation_samples < elevation_bins:
    raise ValueError("LiDAR pooling bins cannot exceed the source grid")
  expected_values = scan_count * elevation_samples * azimuth_samples
  if ranges.shape[-1] != expected_values:
    raise ValueError(
      f"LiDAR scan stack has {ranges.shape[-1]} values; expected {expected_values}"
    )
  grid = ranges.reshape(
    -1,
    scan_count,
    elevation_samples,
    azimuth_samples,
  )
  pooled = -F.adaptive_max_pool2d(
    -grid,
    output_size=(elevation_bins, azimuth_bins),
  )
  return pooled.flatten(start_dim=1)


def held_lidar_scan_age(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Return complete-scan age as a clipped fraction of one scan period."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, HeldScanRayCastSensor):
    raise TypeError(f"Sensor '{sensor_name}' must be a HeldScanRayCastSensor")
  phases = float(sensor.cfg.pattern.phases)
  age = (sensor.published_scan_age_steps.float() / phases).clamp(0.0, 1.0)
  return age.unsqueeze(-1)


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
        torch.rand((batch_size, 1), device=output.device) < self.sector_dropout_prob
      )
      sector = (sector & active).view(batch_size, 1, self.azimuth_samples)
      sector = sector.expand(batch_size, num_elevations, self.azimuth_samples)
      output.view(batch_size, num_elevations, self.azimuth_samples).masked_fill_(
        sector, self.miss_value
      )

    return output


def directional_held_lidar_scan_pair(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  azimuth_samples: int,
  elevation_samples: int,
  azimuth_bins: int = 24,
  elevation_bins: int = 3,
  min_distance: float = 0.05,
  noise_cfg: LidarNoiseCfg | None = None,
) -> torch.Tensor:
  """Return compact directional features from both complete held scans.

  Dense ray casting and point-level corruption happen before minimum pooling.
  Keeping only the directional minima in the observation prevents PPO rollout
  storage from retaining every raw ray for every transition.
  """
  ranges = normalized_held_lidar_scan_pair(
    env,
    sensor_name,
    min_distance=min_distance,
  )
  if noise_cfg is not None:
    ranges = noise_cfg.apply(ranges)
  return _directional_minimum_pool(
    ranges,
    scan_count=2,
    elevation_samples=elevation_samples,
    azimuth_samples=azimuth_samples,
    elevation_bins=elevation_bins,
    azimuth_bins=azimuth_bins,
  )


def directional_held_lidar_range_rate(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  azimuth_samples: int,
  elevation_samples: int,
  azimuth_bins: int = 24,
  elevation_bins: int = 3,
  min_distance: float = 0.05,
  max_abs_range_rate_mps: float = 5.0,
  noise_cfg: LidarNoiseCfg | None = None,
) -> torch.Tensor:
  """Return current directional range and explicit normalized closing speed.

  Positive range rate means the closest return in that angular cell moved
  toward the sensor between the two completed scans. The rate is expressed in
  physical metres per second before being clipped and normalized to ``[-1, 1]``.
  """
  if max_abs_range_rate_mps <= 0.0:
    raise ValueError("max_abs_range_rate_mps must be positive")
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, HeldScanRayCastSensor):
    raise TypeError(f"Sensor '{sensor_name}' must be a HeldScanRayCastSensor")
  pooled_pair = directional_held_lidar_scan_pair(
    env,
    sensor_name,
    azimuth_samples=azimuth_samples,
    elevation_samples=elevation_samples,
    azimuth_bins=azimuth_bins,
    elevation_bins=elevation_bins,
    min_distance=min_distance,
    noise_cfg=noise_cfg,
  )
  cell_count = azimuth_bins * elevation_bins
  current = pooled_pair[:, :cell_count]
  previous = pooled_pair[:, cell_count:]
  closing_speed_mps = (
    (previous - current) * sensor.cfg.max_distance / sensor.cfg.scan_period
  )
  normalized_rate = torch.clamp(
    closing_speed_mps / max_abs_range_rate_mps,
    min=-1.0,
    max=1.0,
  )
  return torch.cat((current, normalized_rate), dim=-1)


class CachedDirectionalHeldLidarScanPair(ManagerTermBase):
  """Hold one corrupted, pooled observation until the next 10 Hz scan."""

  def __init__(
    self,
    cfg: ObservationTermCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    super().__init__(env)
    params = cfg.params
    self.sensor_name = str(params["sensor_name"])
    self.azimuth_samples = int(params["azimuth_samples"])
    self.elevation_samples = int(params["elevation_samples"])
    self.azimuth_bins = int(params.get("azimuth_bins", 24))
    self.elevation_bins = int(params.get("elevation_bins", 3))
    self.min_distance = float(params.get("min_distance", 0.05))
    self.noise_cfg = params.get("noise_cfg")
    if self.noise_cfg is not None and not isinstance(self.noise_cfg, LidarNoiseCfg):
      raise TypeError("noise_cfg must be a LidarNoiseCfg or None")
    sensor = env.scene[self.sensor_name]
    if not isinstance(sensor, HeldScanRayCastSensor):
      raise TypeError(
        f"Sensor '{self.sensor_name}' must be a HeldScanRayCastSensor"
      )
    self._sensor = sensor
    self._cached: torch.Tensor | None = None
    self._cached_publication_count = -1

  def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
    if (
      self._cached is None
      or self._cached_publication_count != self._sensor.global_publication_count
    ):
      self._cached = directional_held_lidar_scan_pair(
        env,
        self.sensor_name,
        azimuth_samples=self.azimuth_samples,
        elevation_samples=self.elevation_samples,
        azimuth_bins=self.azimuth_bins,
        elevation_bins=self.elevation_bins,
        min_distance=self.min_distance,
        noise_cfg=self.noise_cfg,
      )
      self._cached_publication_count = self._sensor.global_publication_count
    return self._cached

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if self._cached is None:
      return
    ids = slice(None) if env_ids is None else env_ids
    self._cached[ids] = 1.0


class CachedDirectionalHeldLidarRangeRate(CachedDirectionalHeldLidarScanPair):
  """Hold current directional ranges and range rate until the next scan."""

  def __init__(
    self,
    cfg: ObservationTermCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    super().__init__(cfg, env)
    self.max_abs_range_rate_mps = float(
      cfg.params.get("max_abs_range_rate_mps", 5.0)
    )
    if self.max_abs_range_rate_mps <= 0.0:
      raise ValueError("max_abs_range_rate_mps must be positive")

  def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
    if (
      self._cached is None
      or self._cached_publication_count != self._sensor.global_publication_count
    ):
      self._cached = directional_held_lidar_range_rate(
        env,
        self.sensor_name,
        azimuth_samples=self.azimuth_samples,
        elevation_samples=self.elevation_samples,
        azimuth_bins=self.azimuth_bins,
        elevation_bins=self.elevation_bins,
        min_distance=self.min_distance,
        max_abs_range_rate_mps=self.max_abs_range_rate_mps,
        noise_cfg=self.noise_cfg,
      )
      self._cached_publication_count = self._sensor.global_publication_count
    return self._cached

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if self._cached is None:
      return
    ids = slice(None) if env_ids is None else env_ids
    cell_count = self.azimuth_bins * self.elevation_bins
    self._cached[ids, :cell_count] = 1.0
    self._cached[ids, cell_count:] = 0.0


class BlindDirectionalHeldLidarRangeRate(CachedDirectionalHeldLidarRangeRate):
  """The actor's LiDAR term with every ray reading 'no return', forever.

  Drop-in for :class:`CachedDirectionalHeldLidarRangeRate` (same params, same
  output layout: ``cells`` normalized ranges then ``cells`` range rates) that
  returns ranges at maximum and rates at zero, i.e. exactly what the avoidance
  benchmarks feed a policy in their ``blind`` mode. Training with it yields the
  no-perception baseline: the same rewards, filtered reference, and privileged
  critic, but an actor that cannot see humans.
  """

  def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
    cells = self.azimuth_bins * self.elevation_bins
    out = torch.zeros((env.num_envs, 2 * cells), device=env.device)
    out[:, :cells] = 1.0
    return out
