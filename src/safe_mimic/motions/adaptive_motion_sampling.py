"""Random-start and adaptive difficulty sampling for packed motion clips."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from safe_mimic.motions.packed_npz_motion_lib import PackedNpzMotionLib


@dataclass(frozen=True)
class AdaptiveMotionSamplingCfg:
  """Boxing-teacher-style sampling parameters.

  Difficulty is tracked in fixed-duration bins inside each clip.  Sampling is
  a mixture of a duration-weighted random prior and failure-weighted practice,
  so every phase remains reachable.  Adaptive probabilities are normalized
  independently inside each metadata group (WBC and dance by default), keeping
  the curated group mixture stable as difficulty changes.
  """

  bin_duration_s: float = 1.0
  failure_ema_alpha: float = 0.05
  uniform_ratio: float = 0.10
  temporal_kernel_size: int = 1
  temporal_kernel_lambda: float = 0.8
  group_key: str = "sampling_pool"
  pair_key: str = "pair_id"
  couple_pairs: bool = True

  def __post_init__(self) -> None:
    if self.bin_duration_s <= 0.0:
      raise ValueError("bin_duration_s must be positive")
    if not 0.0 < self.failure_ema_alpha <= 1.0:
      raise ValueError("failure_ema_alpha must be in (0, 1]")
    if not 0.0 < self.uniform_ratio <= 1.0:
      raise ValueError("uniform_ratio must be in (0, 1]")
    if self.temporal_kernel_size < 1:
      raise ValueError("temporal_kernel_size must be positive")
    if not 0.0 < self.temporal_kernel_lambda <= 1.0:
      raise ValueError("temporal_kernel_lambda must be in (0, 1]")


@dataclass(frozen=True)
class AdaptiveMotionSample:
  """Randomized reference starts drawn from adaptive phase bins."""

  motion_ids: torch.Tensor
  motion_times: torch.Tensor
  bin_ids: torch.Tensor


class AdaptiveMotionSampler:
  """Failure-aware, clip-safe reference-state initializer.

  This is the separate-clip equivalent of the boxing teacher's adaptive flat
  timeline.  A bin never straddles an NPZ boundary.  The update is fully
  vectorized over environments and stores only one scalar failure estimate per
  roughly one-second phase.
  """

  def __init__(
    self,
    library: PackedNpzMotionLib,
    cfg: AdaptiveMotionSamplingCfg | None = None,
  ) -> None:
    self.library = library
    self.cfg = cfg or AdaptiveMotionSamplingCfg()
    cfg = self.cfg
    self.device = library.device
    lengths = library.motion_lengths.detach().cpu()
    num_bins = torch.ceil(lengths / cfg.bin_duration_s).long().clamp(min=1)
    starts = torch.cat(
      [torch.zeros(1, dtype=torch.long), num_bins[:-1].cumsum(dim=0)]
    )
    total_bins = int(num_bins.sum())

    bin_ids = torch.arange(total_bins, dtype=torch.long)
    motion_ids = torch.repeat_interleave(
      torch.arange(library.num_motions(), dtype=torch.long), num_bins
    )
    local_ids = bin_ids - starts[motion_ids]
    bin_start_times = local_ids.float() * cfg.bin_duration_s
    bin_durations = torch.minimum(
      torch.full_like(bin_start_times, cfg.bin_duration_s),
      lengths[motion_ids] - bin_start_times,
    ).clamp(min=torch.finfo(torch.float32).eps)

    source_weights = torch.tensor(
      [source.weight for source in library.sources], dtype=torch.float32
    )
    base_weights = source_weights[motion_ids] * bin_durations
    base_probabilities = base_weights / base_weights.sum()

    group_names = [
      str(source.metadata.get(cfg.group_key, "all")) for source in library.sources
    ]
    unique_groups = tuple(dict.fromkeys(group_names))
    group_index = {name: index for index, name in enumerate(unique_groups)}
    motion_group_ids = torch.tensor(
      [group_index[name] for name in group_names], dtype=torch.long
    )
    bin_group_ids = motion_group_ids[motion_ids]
    group_prior_mass = torch.zeros(len(unique_groups), dtype=torch.float32)
    group_prior_mass.scatter_add_(0, bin_group_ids, base_probabilities)

    self.group_names = unique_groups
    self._motion_num_bins = num_bins.to(self.device)
    self._motion_bin_starts = starts.to(self.device)
    self._bin_motion_ids = motion_ids.to(self.device)
    self._bin_start_times = bin_start_times.to(self.device)
    self._bin_durations = bin_durations.to(self.device)
    self._bin_group_ids = bin_group_ids.to(self.device)
    self._group_prior_mass = group_prior_mass.to(self.device)
    self._base_probabilities = base_probabilities.to(self.device)
    self._pair_bin_ids = self._build_pair_bin_ids(
      starts, num_bins, motion_ids, local_ids
    ).to(self.device)

    offsets = torch.arange(cfg.temporal_kernel_size, dtype=torch.long)
    motion_ends = starts[motion_ids] + num_bins[motion_ids] - 1
    temporal_neighbors = torch.minimum(
      bin_ids[:, None] + offsets[None], motion_ends[:, None]
    )
    kernel = cfg.temporal_kernel_lambda ** torch.arange(
      cfg.temporal_kernel_size, dtype=torch.float32
    )
    self._temporal_neighbors = temporal_neighbors.to(self.device)
    self._temporal_kernel = (kernel / kernel.sum()).to(self.device)

    self.failure_ema = torch.zeros(total_bins, device=self.device)
    self.episode_counts = torch.zeros(
      total_bins, dtype=torch.long, device=self.device
    )
    self._probabilities = self._base_probabilities.clone()
    self._probabilities_dirty = False

  def _build_pair_bin_ids(
    self,
    starts: torch.Tensor,
    num_bins: torch.Tensor,
    motion_ids: torch.Tensor,
    local_ids: torch.Tensor,
  ) -> torch.Tensor:
    if not self.cfg.couple_pairs:
      return torch.arange(len(motion_ids), dtype=torch.long)
    motions_by_pair: dict[str, list[int]] = defaultdict(list)
    for motion_id, source in enumerate(self.library.sources):
      pair_id = source.metadata.get(self.cfg.pair_key)
      if pair_id is not None:
        motions_by_pair[str(pair_id)].append(motion_id)
    partner_motion = torch.arange(self.library.num_motions(), dtype=torch.long)
    for paired_motions in motions_by_pair.values():
      if len(paired_motions) == 2:
        first, second = paired_motions
        partner_motion[first] = second
        partner_motion[second] = first

    partner = partner_motion[motion_ids]
    source_count = num_bins[motion_ids]
    partner_count = num_bins[partner]
    phase = (local_ids.float() + 0.5) / source_count.float()
    partner_local = torch.floor(phase * partner_count.float()).long()
    partner_local.clamp_(min=0)
    partner_local = torch.minimum(partner_local, partner_count - 1)
    return starts[partner] + partner_local

  @property
  def num_bins(self) -> int:
    return self.failure_ema.numel()

  @property
  def resident_bytes(self) -> int:
    """Approximate resident bytes for adaptive tensors and indexes."""
    tensors = (
      self._motion_num_bins,
      self._motion_bin_starts,
      self._bin_motion_ids,
      self._bin_start_times,
      self._bin_durations,
      self._bin_group_ids,
      self._group_prior_mass,
      self._base_probabilities,
      self._pair_bin_ids,
      self._temporal_neighbors,
      self._temporal_kernel,
      self.failure_ema,
      self.episode_counts,
      self._probabilities,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

  @property
  def base_probabilities(self) -> torch.Tensor:
    return self._base_probabilities

  @property
  def probabilities(self) -> torch.Tensor:
    if self._probabilities_dirty:
      self._refresh_probabilities()
    return self._probabilities

  @property
  def bin_motion_ids(self) -> torch.Tensor:
    return self._bin_motion_ids

  @property
  def bin_group_ids(self) -> torch.Tensor:
    return self._bin_group_ids

  def _refresh_probabilities(self) -> None:
    difficulty = (
      self.failure_ema[self._temporal_neighbors] * self._temporal_kernel
    ).sum(dim=-1)
    if self.cfg.couple_pairs:
      difficulty = 0.5 * (difficulty + difficulty[self._pair_bin_ids])
    adaptive_raw = self._base_probabilities * difficulty
    adaptive_group_sum = torch.zeros_like(self._group_prior_mass)
    adaptive_group_sum.scatter_add_(0, self._bin_group_ids, adaptive_raw)

    base_group_probability = (
      self._base_probabilities
      / self._group_prior_mass[self._bin_group_ids].clamp(min=1e-12)
    )
    adaptive_group_probability = (
      adaptive_raw
      / adaptive_group_sum[self._bin_group_ids].clamp(min=1e-12)
    )
    has_failures = adaptive_group_sum[self._bin_group_ids] > 0.0
    within_group = torch.where(
      has_failures, adaptive_group_probability, base_group_probability
    )
    adaptive = within_group * self._group_prior_mass[self._bin_group_ids]
    probabilities = (
      self.cfg.uniform_ratio * self._base_probabilities
      + (1.0 - self.cfg.uniform_ratio) * adaptive
    )
    self._probabilities = probabilities / probabilities.sum()
    self._probabilities_dirty = False

  def sample(
    self,
    count: int,
    *,
    generator: torch.Generator | None = None,
  ) -> AdaptiveMotionSample:
    """Draw randomized motion/time pairs from the current difficulty mixture."""
    return self._sample_from_probabilities(
      self.probabilities, count, generator=generator
    )

  def sample_prior(
    self,
    count: int,
    *,
    generator: torch.Generator | None = None,
  ) -> AdaptiveMotionSample:
    """Draw from the duration-weighted random prior without adaptation."""
    return self._sample_from_probabilities(
      self._base_probabilities, count, generator=generator
    )

  def _sample_from_probabilities(
    self,
    probabilities: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None,
  ) -> AdaptiveMotionSample:
    if count < 0:
      raise ValueError("count must be non-negative")
    bin_ids = torch.multinomial(
      probabilities,
      num_samples=count,
      replacement=True,
      generator=generator,
    )
    phase = torch.rand(count, device=self.device, generator=generator)
    motion_times = (
      self._bin_start_times[bin_ids] + phase * self._bin_durations[bin_ids]
    )
    return AdaptiveMotionSample(
      motion_ids=self._bin_motion_ids[bin_ids],
      motion_times=motion_times,
      bin_ids=bin_ids,
    )

  def update(
    self,
    failures: torch.Tensor,
    motion_ids: torch.Tensor,
    motion_times: torch.Tensor,
  ) -> None:
    """Update per-phase failure EMAs from completed vectorized episodes."""
    failures = torch.as_tensor(failures, device=self.device).bool().reshape(-1)
    motion_ids = torch.as_tensor(
      motion_ids, dtype=torch.long, device=self.device
    ).reshape(-1)
    motion_times = torch.as_tensor(
      motion_times, dtype=torch.float32, device=self.device
    ).reshape(-1)
    if not (failures.shape == motion_ids.shape == motion_times.shape):
      raise ValueError("failures, motion_ids, and motion_times must have equal shape")
    if failures.numel() == 0:
      return
    local_bin = torch.floor(motion_times / self.cfg.bin_duration_s).long()
    local_bin.clamp_(min=0)
    local_bin = torch.minimum(local_bin, self._motion_num_bins[motion_ids] - 1)
    bin_ids = self._motion_bin_starts[motion_ids] + local_bin

    episodes = torch.bincount(bin_ids, minlength=self.num_bins)
    failed = torch.bincount(
      bin_ids, weights=failures.float(), minlength=self.num_bins
    )
    observed = episodes > 0
    observed_rate = failed / episodes.clamp(min=1)
    decay = (1.0 - self.cfg.failure_ema_alpha) ** episodes.float()
    self.failure_ema = torch.where(
      observed,
      decay * self.failure_ema + (1.0 - decay) * observed_rate,
      self.failure_ema,
    )
    self.episode_counts += episodes
    self._probabilities_dirty = True

  def metrics(self) -> dict[str, float]:
    """Return compact sampling diagnostics without retaining rollout tensors."""
    probabilities = self.probabilities
    entropy = -(probabilities * probabilities.clamp(min=1e-12).log()).sum()
    normalized_entropy = (
      entropy
      / torch.log(torch.tensor(float(self.num_bins), device=self.device))
      if self.num_bins > 1
      else torch.ones((), device=self.device)
    )
    return {
      "sampling_entropy": float(normalized_entropy),
      "sampling_top1_prob": float(probabilities.max()),
      "observed_bin_fraction": float((self.episode_counts > 0).float().mean()),
      "mean_failure_ema": float(self.failure_ema.mean()),
    }

  def state_dict(self) -> dict[str, torch.Tensor]:
    """Return checkpointable adaptive state."""
    return {
      "failure_ema": self.failure_ema.detach().clone(),
      "episode_counts": self.episode_counts.detach().clone(),
    }

  def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    """Restore adaptive state after validating it against this exact bank."""
    failure_ema = torch.as_tensor(
      state["failure_ema"], dtype=torch.float32, device=self.device
    )
    episode_counts = torch.as_tensor(
      state["episode_counts"], dtype=torch.long, device=self.device
    )
    if failure_ema.shape != self.failure_ema.shape:
      raise ValueError("adaptive state does not match this motion bank")
    if episode_counts.shape != self.episode_counts.shape:
      raise ValueError("adaptive state does not match this motion bank")
    self.failure_ema.copy_(failure_ema)
    self.episode_counts.copy_(episode_counts)
    self._probabilities_dirty = True


__all__ = [
  "AdaptiveMotionSample",
  "AdaptiveMotionSampler",
  "AdaptiveMotionSamplingCfg",
]
