"""A compact 360-degree, multi-elevation LiDAR pattern for mjlab ray casting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import mujoco
  import torch


@dataclass(frozen=True)
class SphericalLidarPatternCfg:
  """Generate a range-image-ordered spherical ray pattern.

  Rays are elevation-major and azimuth-minor. Reshape a flat sensor result to
  ``[batch, len(elevation_angles_deg), azimuth_samples]`` to recover the range
  image. Azimuth zero points along the attached frame's +X axis and increases
  toward +Y.

  This intentionally follows mjlab's lightweight pattern protocol: a pattern
  only needs to implement ``generate_rays(mj_model, device)``.
  """

  azimuth_samples: int = 180
  elevation_angles_deg: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
  azimuth_offset_deg: float = 0.0
  origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.40)

  def __post_init__(self) -> None:
    if self.azimuth_samples < 4:
      raise ValueError("azimuth_samples must be at least 4")
    if not self.elevation_angles_deg:
      raise ValueError("at least one elevation angle is required")
    if any(not -90.0 < angle < 90.0 for angle in self.elevation_angles_deg):
      raise ValueError("elevation angles must be strictly between -90 and 90 degrees")

  @property
  def num_rays(self) -> int:
    return self.azimuth_samples * len(self.elevation_angles_deg)

  def local_directions(self) -> tuple[tuple[float, float, float], ...]:
    """Return unit ray directions without requiring a Torch installation."""
    directions: list[tuple[float, float, float]] = []
    azimuth_offset = math.radians(self.azimuth_offset_deg)
    for elevation_deg in self.elevation_angles_deg:
      elevation = math.radians(elevation_deg)
      cos_elevation = math.cos(elevation)
      for index in range(self.azimuth_samples):
        azimuth = 2.0 * math.pi * index / self.azimuth_samples + azimuth_offset
        directions.append(
          (
            cos_elevation * math.cos(azimuth),
            cos_elevation * math.sin(azimuth),
            math.sin(elevation),
          )
        )
    return tuple(directions)

  def generate_rays(
    self,
    mj_model: mujoco.MjModel | None,
    device: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local origins and directions in mjlab's pattern format."""
    del mj_model
    import torch

    directions = torch.tensor(
      self.local_directions(), device=device, dtype=torch.float32
    )
    offsets = torch.tensor(
      self.origin_offset, device=device, dtype=torch.float32
    ).expand(self.num_rays, 3)
    return offsets.clone(), directions
