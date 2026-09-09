"""Curriculum-ramped, failure-adaptive sampling of primary-human encounters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

ENCOUNTER_SAMPLER_PARAM_NAMES = (
  "ttc_bin_edges_s",
  "speed_bin_edges_mps",
  "hard_ttc_below_s",
  "hard_speed_above_mps",
  "curriculum_ramp_s",
  "bin_failure_ema_alpha",
  "bin_base_weight",
)

_COLLISION_TERM_NAMES = ("primary_human_collision", "crowd_collision")


def encounter_sampler_overrides(params: Mapping[str, Any]) -> dict[str, Any]:
  """Extract optional :class:`EncounterSampler` keyword overrides."""
  return {
    name: params[name] for name in ENCOUNTER_SAMPLER_PARAM_NAMES if name in params
  }


def read_collision_terms(
  termination_manager: Any,
  num_envs: int,
  device: str | torch.device,
) -> torch.Tensor:
  """Return the current OR of the human-collision termination terms.

  Inside an event's ``reset`` the term buffers still hold the values computed
  for the episodes that just ended (the environment resets managers only
  after events, and the termination manager's reset does not clear them).
  Demo and replay configurations may lack either term; missing terms simply
  contribute nothing.
  """
  collided = torch.zeros(num_envs, dtype=torch.bool, device=device)
  for name in _COLLISION_TERM_NAMES:
    if name in termination_manager.active_terms:
      collided |= termination_manager.get_term(name).to(dtype=torch.bool)
  return collided


class EncounterSampler:
  """Sample intercept (TTC, approach speed) pairs from an adaptive bin grid.

  A small (TTC x speed) grid carries one global failure EMA per bin, updated
  edge-triggered: each schedule call attributes the outcome of the previous
  assignment exactly once via an explicit flag (a collision always resets the
  environment, so a mid-episode reschedule implies no collision). Hard bins
  (short TTC or fast approach) are down-weighted at the start of training and
  ramp to their adaptive weight over ``curriculum_ramp_s`` of simulated time.
  """

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    *,
    ttc_bin_edges_s: tuple[float, ...] = (0.5, 1.5, 2.5, 4.0),
    speed_bin_edges_mps: tuple[float, ...] = (0.75, 1.5, 3.0),
    spawn_radius_clamp_m: tuple[float, float],
    hard_ttc_below_s: float = 1.5,
    hard_speed_above_mps: float = 1.5,
    curriculum_ramp_s: float = 3000.0,
    bin_failure_ema_alpha: float = 0.05,
    bin_base_weight: float = 0.15,
  ) -> None:
    ttc_edges = tuple(float(value) for value in ttc_bin_edges_s)
    speed_edges = tuple(float(value) for value in speed_bin_edges_mps)
    radius_clamp = tuple(float(value) for value in spawn_radius_clamp_m)
    if num_envs < 1:
      raise ValueError("num_envs must be positive")
    for name, edges in (
      ("ttc_bin_edges_s", ttc_edges),
      ("speed_bin_edges_mps", speed_edges),
    ):
      if len(edges) < 2:
        raise ValueError(f"{name} requires at least two edges")
      if edges[0] <= 0.0:
        raise ValueError(f"{name} must be positive")
      if any(high <= low for low, high in zip(edges[:-1], edges[1:], strict=True)):
        raise ValueError(f"{name} must be strictly increasing")
    if len(radius_clamp) != 2 or not 0.0 < radius_clamp[0] <= radius_clamp[1]:
      raise ValueError("spawn_radius_clamp_m must be a positive (min, max) pair")
    if curriculum_ramp_s <= 0.0:
      raise ValueError("curriculum_ramp_s must be positive")
    if not 0.0 < bin_failure_ema_alpha <= 1.0:
      raise ValueError("bin_failure_ema_alpha must lie in (0, 1]")
    if bin_base_weight < 0.0:
      raise ValueError("bin_base_weight must be non-negative")

    self._num_envs = num_envs
    self._device = torch.device(device)
    self._ttc_edges = torch.tensor(ttc_edges, device=self._device)
    self._speed_edges = torch.tensor(speed_edges, device=self._device)
    self._radius_clamp = (radius_clamp[0], radius_clamp[1])
    self._curriculum_ramp_s = float(curriculum_ramp_s)
    self._alpha = float(bin_failure_ema_alpha)
    self._base_weight = float(bin_base_weight)
    self._ttc_bin_count = len(ttc_edges) - 1
    self._speed_bin_count = len(speed_edges) - 1
    bin_count = self._ttc_bin_count * self._speed_bin_count
    ttc_lows = self._ttc_edges[:-1].repeat_interleave(self._speed_bin_count)
    speed_lows = self._speed_edges[:-1].repeat(self._ttc_bin_count)
    self._hard_bins = (ttc_lows < hard_ttc_below_s) | (
      speed_lows >= hard_speed_above_mps
    )
    self._failure_ema = torch.full((bin_count,), 0.5, device=self._device)
    self._assigned_bin = torch.full(
      (num_envs,), -1, dtype=torch.long, device=self._device
    )

  @property
  def ttc_bin_count(self) -> int:
    return self._ttc_bin_count

  @property
  def speed_bin_count(self) -> int:
    return self._speed_bin_count

  @property
  def ttc_bin_edges_s(self) -> torch.Tensor:
    return self._ttc_edges

  @property
  def speed_bin_edges_mps(self) -> torch.Tensor:
    return self._speed_edges

  @property
  def hard_bin_mask(self) -> torch.Tensor:
    return self._hard_bins

  @property
  def bin_failure_ema(self) -> torch.Tensor:
    return self._failure_ema

  @property
  def assigned_bins(self) -> torch.Tensor:
    """Realized bin id of each environment's latest schedule (-1 for none)."""
    return self._assigned_bin

  def hard_bin_ramp(self, global_time_s: float) -> float:
    """Return the hard-bin weight multiplier at the given simulated time."""
    progress = min(1.0, max(0.0, global_time_s / self._curriculum_ramp_s))
    return 0.1 + 0.9 * progress

  def bin_weights(self, global_time_s: float) -> torch.Tensor:
    """Return normalized sampling weights over the flattened bin grid."""
    weights = self._base_weight + self._failure_ema
    ramp = self.hard_bin_ramp(global_time_s)
    weights = torch.where(self._hard_bins, weights * ramp, weights)
    weights = weights.clamp_min(0.05)
    return weights / weights.sum()

  def observe_terminal(
    self,
    env_ids: torch.Tensor,
    collided: torch.Tensor,
  ) -> None:
    """Attribute an ended episode's outcome without scheduling a new one.

    Used for environments whose next episode has no humans, so their pending
    assignment is neither lost nor later attributed a false no-collision.
    """
    env_ids = env_ids.to(device=self._device, dtype=torch.long)
    self._consume(env_ids, collided)

  def clear_assignments(self, env_ids: torch.Tensor) -> None:
    """Drop pending assignments without attribution (deactivation safety)."""
    env_ids = env_ids.to(device=self._device, dtype=torch.long)
    self._assigned_bin[env_ids] = -1

  def sample(
    self,
    env_ids: torch.Tensor,
    global_time_s: float,
    collided_since_last: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attribute previous outcomes for ``env_ids`` and sample new encounters.

    ``collided_since_last`` holds one flag per entry of ``env_ids``: True when
    the episode since that environment's previous schedule ended in a human
    collision. Returns ``(ttc_s, approach_speed_mps, spawn_radius_m)`` where
    the speed is recomputed from the clamped radius so callers can schedule
    consistently.
    """
    env_ids = env_ids.to(device=self._device, dtype=torch.long)
    count = int(env_ids.numel())
    self._consume(env_ids, collided_since_last)
    if count == 0:
      empty = torch.empty(0, device=self._device)
      return empty, empty.clone(), empty.clone()
    bins = torch.multinomial(self.bin_weights(global_time_s), count, replacement=True)
    ttc_bin = torch.div(bins, self._speed_bin_count, rounding_mode="floor")
    speed_bin = torch.remainder(bins, self._speed_bin_count)
    ttc_low = self._ttc_edges[ttc_bin]
    ttc_span = self._ttc_edges[ttc_bin + 1] - ttc_low
    speed_low = self._speed_edges[speed_bin]
    speed_span = self._speed_edges[speed_bin + 1] - speed_low
    ttc = ttc_low + torch.rand(count, device=self._device) * ttc_span
    speed = speed_low + torch.rand(count, device=self._device) * speed_span
    radius = torch.clamp(ttc * speed, self._radius_clamp[0], self._radius_clamp[1])
    effective_speed = radius / ttc
    # Attribution stores the REALIZED bin: the radius clamp can move the
    # effective approach speed out of the sampled bin, and the failure EMAs
    # must describe encounters as they actually ran.
    self._assigned_bin[env_ids] = self._realized_bins(ttc, effective_speed)
    return ttc, effective_speed, radius

  def _realized_bins(self, ttc: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    ttc_bin = (torch.bucketize(ttc, self._ttc_edges, right=True) - 1).clamp_(
      0, self._ttc_bin_count - 1
    )
    speed_bin = (torch.bucketize(speed, self._speed_edges, right=True) - 1).clamp_(
      0, self._speed_bin_count - 1
    )
    return ttc_bin * self._speed_bin_count + speed_bin

  def _consume(self, env_ids: torch.Tensor, collided: torch.Tensor) -> None:
    """EMA-update the pending assignments of ``env_ids`` and clear them."""
    if collided.shape != env_ids.shape:
      raise ValueError("collision flags must align one-to-one with env_ids")
    assigned = self._assigned_bin[env_ids]
    has_assignment = assigned >= 0
    bins = assigned[has_assignment]
    if bins.numel():
      flags = collided.to(device=self._device, dtype=torch.bool)
      flags = flags[has_assignment].float()
      bin_count = self._failure_ema.numel()
      counts = torch.zeros(bin_count, device=self._device)
      counts.scatter_add_(0, bins, torch.ones_like(flags))
      flag_sums = torch.zeros(bin_count, device=self._device)
      flag_sums.scatter_add_(0, bins, flags)
      present = counts > 0
      mean_flag = flag_sums / counts.clamp_min(1.0)
      # Applying the per-environment update ``ema = (1-a)*ema + a*flag`` once
      # per reporting environment, batched: n updates against the bin's mean
      # flag give ``(1-a)^n * ema + (1 - (1-a)^n) * mean`` deterministically.
      keep = (1.0 - self._alpha) ** counts
      blended = keep * self._failure_ema + (1.0 - keep) * mean_flag
      self._failure_ema = torch.where(present, blended, self._failure_ema)
    self._assigned_bin[env_ids] = -1


__all__ = [
  "ENCOUNTER_SAMPLER_PARAM_NAMES",
  "EncounterSampler",
  "encounter_sampler_overrides",
  "read_collision_terms",
]
