import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ObjRef
from mjlab.sensor.raycast_sensor import RayCastData

from safe_mimic.sensing.held_scan import (
  HeldScanRayCastSensorCfg,
  InterleavedSphericalLidarPatternCfg,
  _apply_range_limits_,
)
from safe_mimic.sensing.observations import (
  CachedDirectionalHeldLidarRangeRate,
  CachedDirectionalHeldLidarScanPair,
  CachedDirectionalLidarRanges,
  directional_held_lidar_range_rate,
  directional_held_lidar_scan_pair,
  directional_lidar_ranges,
  held_lidar_scan_age,
  normalized_held_lidar_scan_pair,
)


def test_range_limits_turn_near_and_far_returns_into_misses() -> None:
  distances = torch.tensor([[-1.0, 0.29, 0.3, 5.0, 5.01]])
  normals = torch.ones(1, 5, 3)
  origins = torch.tensor([[[1.0, 2.0, 3.0]]]).expand(1, 5, 3)
  hits = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)

  _apply_range_limits_(
    distances,
    normals,
    hits,
    origins,
    min_distance=0.3,
    max_distance=5.0,
  )

  torch.testing.assert_close(distances, torch.tensor([[-1.0, -1.0, 0.3, 5.0, -1.0]]))
  assert torch.equal(normals[0, 2:4], torch.ones(2, 3))
  assert torch.count_nonzero(normals[0, (0, 1, 4)]) == 0
  assert torch.equal(hits[0, (0, 1, 4)], origins[0, (0, 1, 4)])


def test_completed_scan_pair_and_age_use_sensor_publication_state() -> None:
  cfg = HeldScanRayCastSensorCfg(
    name="lidar",
    frame=ObjRef(type="body", name="robot"),
    pattern=InterleavedSphericalLidarPatternCfg(
      azimuth_samples=4,
      elevation_angles_deg=(0.0, -20.0),
      phases=2,
    ),
    max_distance=5.0,
  )
  sensor = cfg.build()
  current = torch.tensor([[1.0, -1.0, 2.5, 5.0, 0.5, 4.0, -1.0, 3.0]])
  previous = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, -1.0, 0.5, 2.5]])
  zeros3 = torch.zeros(1, 8, 3)
  sensor._cached_data = RayCastData(
    distances=current,
    normals_w=zeros3,
    hit_pos_w=zeros3,
    pos_w=torch.zeros(1, 3),
    quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    frame_pos_w=torch.zeros(1, 1, 3),
    frame_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
  )
  sensor._cache_valid = True
  sensor._previous_published_distances = previous
  sensor._published_scan_age_steps = torch.tensor([1])
  env = type("FakeEnv", (), {"scene": {"lidar": sensor}})()

  pair = normalized_held_lidar_scan_pair(env, "lidar")
  age = held_lidar_scan_age(env, "lidar")

  assert pair.shape == (1, 16)
  torch.testing.assert_close(pair[:, :8], current.where(current >= 0, 5.0) / 5.0)
  torch.testing.assert_close(pair[:, 8:], previous.where(previous >= 0, 5.0) / 5.0)
  torch.testing.assert_close(age, torch.tensor([[0.5]]))

  directional_pair = directional_held_lidar_scan_pair(
    env,
    "lidar",
    azimuth_samples=4,
    elevation_samples=2,
    azimuth_bins=2,
    elevation_bins=2,
  )
  assert directional_pair.shape == (1, 8)
  torch.testing.assert_close(
    directional_pair,
    torch.tensor([[0.2, 0.5, 0.1, 0.6, 0.8, 0.4, 0.2, 0.1]]),
  )
  directional_range_rate = directional_held_lidar_range_rate(
    env,
    "lidar",
    azimuth_samples=4,
    elevation_samples=2,
    azimuth_bins=2,
    elevation_bins=2,
    max_abs_range_rate_mps=50.0,
  )
  torch.testing.assert_close(
    directional_range_rate,
    torch.tensor([[0.2, 0.5, 0.1, 0.6, 0.6, -0.1, 0.1, -0.5]]),
  )

  actor_cfg = ObservationTermCfg(
    func=CachedDirectionalHeldLidarScanPair,
    params={
      "sensor_name": "lidar",
      "azimuth_samples": 4,
      "elevation_samples": 2,
      "azimuth_bins": 2,
      "elevation_bins": 2,
      "noise_cfg": None,
    },
  )
  critic_cfg = ObservationTermCfg(
    func=CachedDirectionalLidarRanges,
    params={
      "sensor_name": "lidar",
      "azimuth_samples": 4,
      "elevation_samples": 2,
      "azimuth_bins": 2,
      "elevation_bins": 2,
    },
  )
  actor_term = CachedDirectionalHeldLidarScanPair(actor_cfg, env)
  rate_cfg = ObservationTermCfg(
    func=CachedDirectionalHeldLidarRangeRate,
    params={
      **actor_cfg.params,
      "max_abs_range_rate_mps": 50.0,
    },
  )
  rate_term = CachedDirectionalHeldLidarRangeRate(rate_cfg, env)
  critic_term = CachedDirectionalLidarRanges(critic_cfg, env)
  actor_first = actor_term(env).clone()
  rate_first = rate_term(env).clone()
  critic_first = critic_term(env).clone()

  current.fill_(5.0)
  previous.fill_(5.0)
  torch.testing.assert_close(actor_term(env), actor_first)
  torch.testing.assert_close(rate_term(env), rate_first)
  torch.testing.assert_close(critic_term(env), critic_first)

  sensor._global_publication_count += 1
  torch.testing.assert_close(actor_term(env), torch.ones_like(actor_first))
  torch.testing.assert_close(
    rate_term(env),
    torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
  )
  torch.testing.assert_close(critic_term(env), torch.ones_like(critic_first))


def test_directional_critic_ranges_keep_closest_return_per_cell() -> None:
  cfg = HeldScanRayCastSensorCfg(
    name="lidar",
    frame=ObjRef(type="body", name="robot"),
    pattern=InterleavedSphericalLidarPatternCfg(
      azimuth_samples=4,
      elevation_angles_deg=(0.0, -20.0),
      phases=2,
    ),
    max_distance=5.0,
  )
  sensor = cfg.build()
  distances = torch.tensor([[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]])
  zeros3 = torch.zeros(1, 8, 3)
  sensor._cached_data = RayCastData(
    distances=distances,
    normals_w=zeros3,
    hit_pos_w=zeros3,
    pos_w=torch.zeros(1, 3),
    quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    frame_pos_w=torch.zeros(1, 1, 3),
    frame_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
  )
  sensor._cache_valid = True
  env = type("FakeEnv", (), {"scene": {"lidar": sensor}})()

  pooled = directional_lidar_ranges(
    env,
    "lidar",
    azimuth_samples=4,
    elevation_samples=2,
    azimuth_bins=2,
    elevation_bins=2,
  )

  torch.testing.assert_close(pooled, torch.tensor([[0.2, 0.6, 0.6, 0.2]]))
