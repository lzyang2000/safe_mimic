"""Unit tests for curriculum/failure-adaptive encounter sampling."""

import pytest
import torch

from safe_mimic.tasks.encounter_sampling import (
  EncounterSampler,
  encounter_sampler_overrides,
)

_NUM_ENVS = 8


def _sampler(**overrides) -> EncounterSampler:
  kwargs = {"spawn_radius_clamp_m": (0.75, 4.0)}
  kwargs.update(overrides)
  return EncounterSampler(_NUM_ENVS, "cpu", **kwargs)


def _no_collisions(count: int) -> torch.Tensor:
  return torch.zeros(count, dtype=torch.bool)


def test_bin_weights_normalize_and_respect_floor() -> None:
  sampler = _sampler(bin_base_weight=0.0)
  sampler.bin_failure_ema.zero_()
  weights = sampler.bin_weights(0.0)
  assert torch.isclose(weights.sum(), torch.tensor(1.0))
  # Raw weights are all zero, so every bin is lifted by the 0.05 floor and
  # normalization yields the uniform distribution.
  assert torch.allclose(weights, torch.full_like(weights, 1.0 / weights.numel()))


def test_floor_applies_after_the_hard_bin_multiplier() -> None:
  # With base+EMA = 0.2 the two orders diverge: multiplier-then-floor gives
  # hard bins max(0.05, 0.02) = 0.05, floor-then-multiplier would give 0.02.
  sampler = _sampler(bin_base_weight=0.0)
  sampler.bin_failure_ema.fill_(0.2)
  weights = sampler.bin_weights(0.0)
  hard = sampler.hard_bin_mask
  hard_count = int(hard.sum())
  easy_count = int((~hard).sum())
  total = 0.05 * hard_count + 0.2 * easy_count
  assert torch.allclose(weights[hard], torch.full((hard_count,), 0.05 / total))
  assert torch.allclose(weights[~hard], torch.full((easy_count,), 0.2 / total))


def test_hard_bin_ramp_endpoints() -> None:
  sampler = _sampler(curriculum_ramp_s=3000.0)
  assert sampler.hard_bin_ramp(0.0) == pytest.approx(0.1)
  assert sampler.hard_bin_ramp(3000.0) == pytest.approx(1.0)
  assert sampler.hard_bin_ramp(9000.0) == pytest.approx(1.0)
  assert 0.1 < sampler.hard_bin_ramp(1500.0) < 1.0


def test_hard_bins_are_downweighted_early_and_uniform_late() -> None:
  sampler = _sampler()
  hard = sampler.hard_bin_mask
  assert hard.any() and (~hard).any()
  early = sampler.bin_weights(0.0)
  late = sampler.bin_weights(1.0e9)
  assert torch.all(early[hard] < early[~hard].min())
  assert torch.allclose(late, torch.full_like(late, 1.0 / late.numel()))


def test_default_hard_bin_mask_matches_edge_rule() -> None:
  sampler = _sampler()
  ttc_lows = sampler.ttc_bin_edges_s[:-1].repeat_interleave(sampler.speed_bin_count)
  speed_lows = sampler.speed_bin_edges_mps[:-1].repeat(sampler.ttc_bin_count)
  expected = (ttc_lows < 1.5) | (speed_lows >= 1.5)
  assert torch.equal(sampler.hard_bin_mask, expected)


def test_collision_flags_without_assignment_do_not_update_ema() -> None:
  sampler = _sampler()
  before = sampler.bin_failure_ema.clone()
  sampler.sample(torch.arange(_NUM_ENVS), 0.0, torch.ones(_NUM_ENVS, dtype=torch.bool))
  assert torch.equal(sampler.bin_failure_ema, before)


def test_failure_ema_moves_toward_observed_flags() -> None:
  torch.manual_seed(0)
  sampler = _sampler()
  env_ids = torch.arange(_NUM_ENVS)
  sampler.sample(env_ids, 1.0e9, _no_collisions(_NUM_ENVS))
  touched = torch.unique(sampler.assigned_bins)
  untouched = torch.ones_like(sampler.bin_failure_ema, dtype=torch.bool)
  untouched[touched] = False
  before = sampler.bin_failure_ema.clone()

  sampler.sample(env_ids, 1.0e9, torch.ones(_NUM_ENVS, dtype=torch.bool))
  after_collisions = sampler.bin_failure_ema.clone()
  assert torch.all(after_collisions[touched] > before[touched])
  assert torch.equal(after_collisions[untouched], before[untouched])

  touched_second = torch.unique(sampler.assigned_bins)
  sampler.sample(env_ids, 1.0e9, _no_collisions(_NUM_ENVS))
  after_survivals = sampler.bin_failure_ema
  assert torch.all(after_survivals[touched_second] < after_collisions[touched_second])


def test_attribution_updates_exactly_the_realized_bin_once() -> None:
  torch.manual_seed(3)
  alpha = 0.25
  sampler = _sampler(bin_failure_ema_alpha=alpha)
  sampler.sample(torch.tensor([0]), 1.0e9, _no_collisions(1))
  assigned = int(sampler.assigned_bins[0])
  before = sampler.bin_failure_ema.clone()
  # Env 1 carries no assignment; its True flag must be ignored entirely.
  sampler.sample(torch.tensor([0, 1]), 1.0e9, torch.ones(2, dtype=torch.bool))
  after = sampler.bin_failure_ema
  expected = (1.0 - alpha) * float(before[assigned]) + alpha * 1.0
  assert float(after[assigned]) == pytest.approx(expected)
  others = torch.ones_like(after, dtype=torch.bool)
  others[assigned] = False
  assert torch.equal(after[others], before[others])
  assert torch.all(sampler.assigned_bins[:2] >= 0)


def test_observe_terminal_attributes_and_clears_without_rescheduling() -> None:
  torch.manual_seed(4)
  alpha = 0.25
  sampler = _sampler(bin_failure_ema_alpha=alpha)
  sampler.sample(torch.tensor([2]), 1.0e9, _no_collisions(1))
  assigned = int(sampler.assigned_bins[2])
  before = float(sampler.bin_failure_ema[assigned])
  sampler.observe_terminal(torch.tensor([2]), torch.ones(1, dtype=torch.bool))
  assert float(sampler.bin_failure_ema[assigned]) == pytest.approx(
    (1.0 - alpha) * before + alpha
  )
  assert int(sampler.assigned_bins[2]) == -1
  # The outcome was consumed: a later schedule must not attribute it again.
  ema_after_observe = sampler.bin_failure_ema.clone()
  sampler.sample(torch.tensor([2]), 1.0e9, _no_collisions(1))
  assert torch.equal(sampler.bin_failure_ema, ema_after_observe)


def test_clear_assignments_drops_pending_attribution() -> None:
  torch.manual_seed(5)
  sampler = _sampler()
  sampler.sample(torch.tensor([3]), 1.0e9, _no_collisions(1))
  sampler.clear_assignments(torch.tensor([3]))
  assert int(sampler.assigned_bins[3]) == -1
  before = sampler.bin_failure_ema.clone()
  sampler.sample(torch.tensor([3]), 1.0e9, torch.ones(1, dtype=torch.bool))
  assert torch.equal(sampler.bin_failure_ema, before)


def test_radius_clamp_and_effective_speed() -> None:
  torch.manual_seed(1)
  sampler = _sampler(spawn_radius_clamp_m=(1.0, 2.0))
  ttc, speed, radius = sampler.sample(
    torch.arange(_NUM_ENVS), 1.0e9, _no_collisions(_NUM_ENVS)
  )
  assert torch.all(radius >= 1.0 - 1.0e-6)
  assert torch.all(radius <= 2.0 + 1.0e-6)
  assert torch.allclose(speed, radius / ttc)


def test_samples_lie_inside_assigned_bins() -> None:
  torch.manual_seed(2)
  # A wide clamp keeps the effective speed equal to the sampled speed, so the
  # realized bins coincide with the sampled ones.
  sampler = _sampler(spawn_radius_clamp_m=(0.3, 12.0))
  ttc, speed, _ = sampler.sample(
    torch.arange(_NUM_ENVS), 1.0e9, _no_collisions(_NUM_ENVS)
  )
  bins = sampler.assigned_bins
  assert torch.all(bins >= 0)
  ttc_bin = torch.div(bins, sampler.speed_bin_count, rounding_mode="floor")
  speed_bin = torch.remainder(bins, sampler.speed_bin_count)
  assert torch.all(ttc >= sampler.ttc_bin_edges_s[ttc_bin])
  assert torch.all(ttc <= sampler.ttc_bin_edges_s[ttc_bin + 1])
  assert torch.all(speed >= sampler.speed_bin_edges_mps[speed_bin])
  assert torch.all(speed <= sampler.speed_bin_edges_mps[speed_bin + 1])


def test_assignment_stores_the_realized_bin_under_clamping() -> None:
  torch.manual_seed(6)
  # Long TTC and fast sampled speeds force the radius clamp: any draw from the
  # fast speed bin realizes below 1.5 m/s and must be re-bucketed.
  count = 64
  sampler = EncounterSampler(
    count,
    "cpu",
    ttc_bin_edges_s=(3.0, 4.0),
    speed_bin_edges_mps=(0.75, 1.5, 3.0),
    spawn_radius_clamp_m=(0.75, 4.0),
  )
  ttc, speed, radius = sampler.sample(torch.arange(count), 1.0e9, _no_collisions(count))
  assert torch.any(radius == 4.0)
  expected_speed_bin = (
    torch.bucketize(speed, sampler.speed_bin_edges_mps, right=True) - 1
  ).clamp(0, sampler.speed_bin_count - 1)
  expected_ttc_bin = (
    torch.bucketize(ttc, sampler.ttc_bin_edges_s, right=True) - 1
  ).clamp(0, sampler.ttc_bin_count - 1)
  expected = expected_ttc_bin * sampler.speed_bin_count + expected_speed_bin
  assert torch.equal(sampler.assigned_bins, expected)
  clamped = radius == 4.0
  assert torch.all(speed[clamped] < 1.5)
  assert torch.all(expected_speed_bin[clamped] == 0)


def test_sample_with_no_env_ids_is_a_no_op() -> None:
  sampler = _sampler()
  ttc, speed, radius = sampler.sample(
    torch.empty(0, dtype=torch.long), 0.0, _no_collisions(0)
  )
  assert ttc.numel() == speed.numel() == radius.numel() == 0
  assert torch.all(sampler.assigned_bins == -1)


def test_sample_rejects_misaligned_collision_flags() -> None:
  sampler = _sampler()
  with pytest.raises(ValueError, match="one-to-one"):
    sampler.sample(torch.arange(4), 0.0, _no_collisions(3))


def test_constructor_rejects_invalid_edges() -> None:
  with pytest.raises(ValueError, match="strictly increasing"):
    _sampler(ttc_bin_edges_s=(0.5, 0.5, 4.0))
  with pytest.raises(ValueError, match="positive"):
    _sampler(speed_bin_edges_mps=(0.0, 1.5))
  with pytest.raises(ValueError, match="min, max"):
    _sampler(spawn_radius_clamp_m=(2.0, 1.0))


def test_encounter_sampler_overrides_filters_known_keys() -> None:
  params = {
    "encounter_sampling": "ttc",
    "curriculum_ramp_s": 10.0,
    "bin_base_weight": 0.2,
    "unrelated": 5,
  }
  assert encounter_sampler_overrides(params) == {
    "curriculum_ramp_s": 10.0,
    "bin_base_weight": 0.2,
  }
