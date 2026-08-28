"""Interleaved ray casting with a held, low-rate LiDAR scan output."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.sensor.raycast_sensor import (
  RayCastData,
  RayCastSensor,
  RayCastSensorCfg,
)

from safe_mimic.sensing.lidar import SphericalLidarPatternCfg


def _apply_range_limits_(
  distances: torch.Tensor,
  normals_w: torch.Tensor,
  hit_pos_w: torch.Tensor,
  origins_w: torch.Tensor,
  *,
  min_distance: float,
  max_distance: float,
) -> None:
  """Turn near-field and over-range returns into ordinary ray misses in-place."""
  invalid = (distances < min_distance) | (distances > max_distance)
  distances.masked_fill_(invalid, -1.0)
  normals_w.masked_fill_(invalid.unsqueeze(-1), 0.0)
  hit_pos_w.copy_(torch.where(invalid.unsqueeze(-1), origins_w, hit_pos_w))


@dataclass(frozen=True)
class InterleavedSphericalLidarPatternCfg(SphericalLidarPatternCfg):
  """Partition a full spherical scan into interleaved azimuth phases."""

  phases: int = 5

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.phases < 1:
      raise ValueError("phases must be positive")
    if self.phases > self.azimuth_samples:
      raise ValueError("phases cannot exceed azimuth_samples")

  @property
  def azimuths_per_phase(self) -> int:
    """Fixed phase width, including at most one padded azimuth."""
    return math.ceil(self.azimuth_samples / self.phases)

  @property
  def rays_per_phase(self) -> int:
    return self.azimuths_per_phase * len(self.elevation_angles_deg)

  def full_indices_for_phase(self, phase: int) -> tuple[int, ...]:
    if not 0 <= phase < self.phases:
      raise ValueError(f"phase must be in [0, {self.phases})")
    indices: list[int] = []
    for elevation_index in range(len(self.elevation_angles_deg)):
      elevation_offset = elevation_index * self.azimuth_samples
      indices.extend(
        elevation_offset + azimuth_index
        for azimuth_index in range(phase, self.azimuth_samples, self.phases)
      )
    return tuple(indices)

  def padded_full_indices_for_phase(self, phase: int) -> tuple[int, ...]:
    """Return fixed-width phase indices, repeating the last ray as padding."""
    indices: list[int] = []
    for elevation_index in range(len(self.elevation_angles_deg)):
      elevation_offset = elevation_index * self.azimuth_samples
      elevation_indices = [
        elevation_offset + azimuth_index
        for azimuth_index in range(phase, self.azimuth_samples, self.phases)
      ]
      indices.extend(elevation_indices)
      indices.extend(
        [elevation_indices[-1]]
        * (self.azimuths_per_phase - len(elevation_indices))
      )
    return tuple(indices)

  def valid_source_indices_for_phase(self, phase: int) -> tuple[int, ...]:
    """Select non-padding rays from a fixed-width phase output."""
    if not 0 <= phase < self.phases:
      raise ValueError(f"phase must be in [0, {self.phases})")
    valid: list[int] = []
    actual_azimuths = len(range(phase, self.azimuth_samples, self.phases))
    for elevation_index in range(len(self.elevation_angles_deg)):
      source_offset = elevation_index * self.azimuths_per_phase
      valid.extend(source_offset + index for index in range(actual_azimuths))
    return tuple(valid)

  def generate_phase_rays(
    self,
    mj_model: mujoco.MjModel | None,
    device: str,
    phase: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del mj_model
    full_directions = self.local_directions()
    indices = self.padded_full_indices_for_phase(phase)
    directions = torch.tensor(
      [full_directions[index] for index in indices],
      device=device,
      dtype=torch.float32,
    )
    offsets = torch.tensor(
      self.origin_offset,
      device=device,
      dtype=torch.float32,
    ).expand(self.rays_per_phase, 3)
    return offsets.clone(), directions

  def generate_rays(
    self,
    mj_model: mujoco.MjModel | None,
    device: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate phase zero; the sensor rotates phases on later updates."""
    return self.generate_phase_rays(mj_model, device, phase=0)


@dataclass
class HeldScanRayCastSensorCfg(RayCastSensorCfg):
  """Ray sensor that publishes one accumulated scan per ``scan_period``."""

  pattern: InterleavedSphericalLidarPatternCfg = field(
    default_factory=InterleavedSphericalLidarPatternCfg
  )
  scan_period: float = 0.1
  min_distance: float = 0.0
  """Blind-zone radius around the LiDAR origin. Nearer returns report -1."""

  def __post_init__(self) -> None:
    if self.scan_period <= 0:
      raise ValueError("scan_period must be positive")
    if not 0.0 <= self.min_distance < self.max_distance:
      raise ValueError("min_distance must be in [0, max_distance)")

  def build(self) -> HeldScanRayCastSensor:
    return HeldScanRayCastSensor(self)


class HeldScanRayCastSensor(RayCastSensor):
  """Cast one scan phase per control step and hold completed scans."""

  def __init__(self, cfg: HeldScanRayCastSensorCfg) -> None:
    super().__init__(cfg)
    self.cfg = cfg
    self._phase = 0
    self._phase_elapsed = 0.0
    self._phase_directions: list[torch.Tensor] = []
    self._phase_indices: list[torch.Tensor] = []
    self._phase_source_indices: list[torch.Tensor] = []

    self._accum_distances: torch.Tensor | None = None
    self._accum_normals_w: torch.Tensor | None = None
    self._accum_hit_pos_w: torch.Tensor | None = None
    self._published_distances: torch.Tensor | None = None
    self._published_normals_w: torch.Tensor | None = None
    self._published_hit_pos_w: torch.Tensor | None = None
    self._published_pos_w: torch.Tensor | None = None
    self._published_quat_w: torch.Tensor | None = None
    self._published_frame_pos_w: torch.Tensor | None = None
    self._published_frame_quat_w: torch.Tensor | None = None

  @property
  def output_num_rays(self) -> int:
    """Number of rays in each published, fully accumulated scan."""
    return self.cfg.pattern.num_rays

  @property
  def rays_per_update(self) -> int:
    """Number of rays evaluated on each 50 Hz environment step."""
    return self.cfg.pattern.rays_per_phase

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    pattern = self.cfg.pattern
    self._phase_directions = []
    self._phase_indices = []
    self._phase_source_indices = []
    for phase in range(pattern.phases):
      _, directions = pattern.generate_phase_rays(mj_model, device, phase)
      self._phase_directions.append(directions)
      self._phase_indices.append(
        torch.tensor(
          pattern.full_indices_for_phase(phase),
          device=device,
          dtype=torch.long,
        )
      )
      self._phase_source_indices.append(
        torch.tensor(
          pattern.valid_source_indices_for_phase(phase),
          device=device,
          dtype=torch.long,
        )
      )

    batch_size = data.nworld
    full_rays = pattern.num_rays
    frames = self._num_frames
    self._accum_distances = torch.full(
      (batch_size, full_rays), -1.0, device=device
    )
    self._accum_normals_w = torch.zeros(
      batch_size, full_rays, 3, device=device
    )
    self._accum_hit_pos_w = torch.zeros(
      batch_size, full_rays, 3, device=device
    )
    self._published_distances = self._accum_distances.clone()
    self._published_normals_w = self._accum_normals_w.clone()
    self._published_hit_pos_w = self._accum_hit_pos_w.clone()
    self._published_pos_w = torch.zeros(batch_size, 3, device=device)
    self._published_quat_w = torch.zeros(batch_size, 4, device=device)
    self._published_quat_w[:, 0] = 1.0
    self._published_frame_pos_w = torch.zeros(
      batch_size, frames, 3, device=device
    )
    self._published_frame_quat_w = torch.zeros(
      batch_size, frames, 4, device=device
    )
    self._published_frame_quat_w[..., 0] = 1.0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if self._published_distances is None:
      return
    ids = slice(None) if env_ids is None else env_ids
    assert self._accum_distances is not None
    assert self._accum_normals_w is not None and self._accum_hit_pos_w is not None
    assert self._published_normals_w is not None
    assert self._published_hit_pos_w is not None
    assert self._published_pos_w is not None and self._published_quat_w is not None
    assert self._published_frame_pos_w is not None
    assert self._published_frame_quat_w is not None
    self._accum_distances[ids] = -1.0
    self._accum_normals_w[ids] = 0.0
    self._accum_hit_pos_w[ids] = 0.0
    self._published_distances[ids] = -1.0
    self._published_normals_w[ids] = 0.0
    self._published_hit_pos_w[ids] = 0.0
    self._published_pos_w[ids] = 0.0
    self._published_quat_w[ids] = 0.0
    self._published_quat_w[ids, 0] = 1.0
    self._published_frame_pos_w[ids] = 0.0
    self._published_frame_quat_w[ids] = 0.0
    self._published_frame_quat_w[ids, :, 0] = 1.0

  def update(self, dt: float) -> None:
    # Deliberately do not invalidate the published cache each physics substep:
    # the policy sees one immutable scan until all five new phases are complete.
    phase_period = self.cfg.scan_period / self.cfg.pattern.phases
    self._phase_elapsed += dt
    while self._phase_elapsed + 1.0e-9 >= phase_period:
      self._phase_elapsed -= phase_period
      self._phase = (self._phase + 1) % self.cfg.pattern.phases

  def prepare_rays(self) -> None:
    self._local_directions = self._phase_directions[self._phase]
    super().prepare_rays()

  def postprocess_rays(self) -> None:
    super().postprocess_rays()
    assert self._distances is not None and self._normals_w is not None
    assert self._hit_pos_w is not None
    assert self._cached_world_origins is not None
    _apply_range_limits_(
      self._distances,
      self._normals_w,
      self._hit_pos_w,
      self._cached_world_origins,
      min_distance=self.cfg.min_distance,
      max_distance=self.cfg.max_distance,
    )
    assert self._accum_distances is not None
    assert self._accum_normals_w is not None and self._accum_hit_pos_w is not None
    indices = self._phase_indices[self._phase]
    source_indices = self._phase_source_indices[self._phase]
    self._accum_distances.index_copy_(
      1, indices, self._distances.index_select(1, source_indices)
    )
    self._accum_normals_w.index_copy_(
      1, indices, self._normals_w.index_select(1, source_indices)
    )
    self._accum_hit_pos_w.index_copy_(
      1, indices, self._hit_pos_w.index_select(1, source_indices)
    )

    if self._phase == self.cfg.pattern.phases - 1:
      assert self._published_distances is not None
      assert self._published_normals_w is not None
      assert self._published_hit_pos_w is not None
      assert self._published_pos_w is not None and self._published_quat_w is not None
      assert self._published_frame_pos_w is not None
      assert self._published_frame_quat_w is not None
      assert self._pos_w is not None and self._quat_w is not None
      assert self._frame_pos_w is not None and self._frame_quat_w is not None
      self._published_distances.copy_(self._accum_distances)
      self._published_normals_w.copy_(self._accum_normals_w)
      self._published_hit_pos_w.copy_(self._accum_hit_pos_w)
      self._published_pos_w.copy_(self._pos_w)
      self._published_quat_w.copy_(self._quat_w)
      self._published_frame_pos_w.copy_(self._frame_pos_w)
      self._published_frame_quat_w.copy_(self._frame_quat_w)
      self._invalidate_cache()

  def _compute_data(self) -> RayCastData:
    if self._published_distances is None:
      return super()._compute_data()
    assert self._published_normals_w is not None
    assert self._published_hit_pos_w is not None
    assert self._published_pos_w is not None and self._published_quat_w is not None
    assert self._published_frame_pos_w is not None
    assert self._published_frame_quat_w is not None
    return RayCastData(
      distances=self._published_distances,
      normals_w=self._published_normals_w,
      hit_pos_w=self._published_hit_pos_w,
      pos_w=self._published_pos_w,
      quat_w=self._published_quat_w,
      frame_pos_w=self._published_frame_pos_w,
      frame_quat_w=self._published_frame_quat_w,
    )

  def debug_vis(self, visualizer) -> None:
    """Draw held hit points; ray arrows stay disabled for this sensor."""
    if not self.cfg.debug_vis or not self._debug_vis_enabled:
      return
    data = self.data
    sphere_radius = self.cfg.viz.hit_sphere_radius * visualizer.meansize
    for env_index in visualizer.get_env_indices(data.distances.shape[0]):
      hit_indices = (data.distances[env_index] >= 0).nonzero().flatten()
      hit_positions = data.hit_pos_w[env_index, hit_indices].cpu().numpy()
      indexed_positions = zip(
        hit_indices.cpu().tolist(), hit_positions, strict=True
      )
      for index, position in indexed_positions:
        visualizer.add_sphere(
          center=position,
          radius=sphere_radius,
          color=self.cfg.viz.hit_sphere_color,
          label=f"{self.cfg.name}_hit_{index}",
        )
