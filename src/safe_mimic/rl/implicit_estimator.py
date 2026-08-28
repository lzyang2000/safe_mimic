"""Alternating PPO and predictive implicit-state representation training.

The deployable history encoder produces a dynamics latent and a small state
head predicts translational signals from it.  Training-only targets supervise
the state head and successor representation.  PPO consumes both predictions
and latent while keeping the representation components frozen during its own
update.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict


@dataclass
class ImplicitStateModelCfg(RslRlModelCfg):
  """Actor configuration for the predictive implicit-state estimator."""

  class_name: str = "safe_mimic.rl:ImplicitStateActor"
  history_obs_group: str = "proprio_history"
  target_proprio_group: str = "implicit_proprio_target"
  estimator_hidden_dims: tuple[int, ...] = (256, 128)
  target_encoder_hidden_dims: tuple[int, ...] = (128, 64)
  estimated_velocity_dim: int = 3
  estimated_root_position_error_dim: int = 3
  dynamics_latent_dim: int = 16
  num_dynamics_prototypes: int = 32
  dynamics_temperature: float = 3.0
  sinkhorn_epsilon: float = 0.05
  sinkhorn_iterations: int = 3


@dataclass
class ImplicitStatePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO configuration with a frozen-then-predictive estimator update."""

  class_name: str = "safe_mimic.rl:ImplicitStatePPO"
  implicit_state_target_group: str = "implicit_state_target"
  implicit_state_learning_rate: float = 1.0e-3
  implicit_state_num_learning_epochs: int = 1
  implicit_state_num_mini_batches: int = 4
  implicit_state_max_grad_norm: float = 1.0
  implicit_dynamics_loss_coef: float = 1.0


class ImplicitStateActor(MLPModel):
  """Actor augmented by predicted state and a supervised history latent."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    history_obs_group: str = "proprio_history",
    target_proprio_group: str = "implicit_proprio_target",
    estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    target_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    estimated_velocity_dim: int = 3,
    estimated_root_position_error_dim: int = 3,
    dynamics_latent_dim: int = 16,
    num_dynamics_prototypes: int = 32,
    dynamics_temperature: float = 3.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 3,
  ) -> None:
    active_groups = list(obs_groups[obs_set])
    if history_obs_group not in active_groups:
      raise ValueError(
        f"Actor observation set must include '{history_obs_group}', got "
        f"{active_groups}"
      )
    policy_groups = [group for group in active_groups if group != history_obs_group]
    if len(policy_groups) != 1:
      raise ValueError(
        "ImplicitStateActor expects exactly one ordinary policy observation "
        f"group, got {policy_groups}"
      )
    self.policy_obs_group = policy_groups[0]
    self.history_obs_group = history_obs_group
    self.target_proprio_group = target_proprio_group
    if target_proprio_group not in obs:
      raise ValueError(
        f"Target proprioception group '{target_proprio_group}' is missing"
      )
    self.policy_obs_dim = int(obs[self.policy_obs_group].shape[-1])
    self.history_obs_dim = int(obs[self.history_obs_group].shape[-1])
    self.target_proprio_dim = int(obs[target_proprio_group].shape[-1])
    self.estimated_velocity_dim = estimated_velocity_dim
    self.estimated_root_position_error_dim = estimated_root_position_error_dim
    self.estimated_state_dim = (
      estimated_velocity_dim + estimated_root_position_error_dim
    )
    self.dynamics_latent_dim = dynamics_latent_dim
    self.dynamics_temperature = dynamics_temperature
    self.sinkhorn_epsilon = sinkhorn_epsilon
    self.sinkhorn_iterations = sinkhorn_iterations

    # MLPModel owns normalization, the action distribution, and export-facing
    # interfaces.  Its latent-size hook is overridden below so the policy head
    # consumes current observations plus compact history-derived signals, not
    # raw history.
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self.history_encoder = MLP(
      self.policy_obs_dim + self.history_obs_dim,
      dynamics_latent_dim,
      estimator_hidden_dims,
      activation,
    )
    # Supervision gives these deployment-time signals fixed physical meaning:
    # body linear velocity followed by reference-root position error.
    self.auxiliary_state_head = torch.nn.Linear(
      dynamics_latent_dim, self.estimated_state_dim
    )
    self.target_encoder = MLP(
      self.target_proprio_dim,
      dynamics_latent_dim,
      target_encoder_hidden_dims,
      activation,
    )
    self.dynamics_prototypes = torch.nn.Embedding(
      num_dynamics_prototypes, dynamics_latent_dim
    )
    self.target_obs_normalizer: torch.nn.Module
    if obs_normalization:
      self.target_obs_normalizer = EmpiricalNormalization(self.target_proprio_dim)
    else:
      self.target_obs_normalizer = torch.nn.Identity()

  def _get_latent_dim(self) -> int:
    return self.policy_obs_dim + self.estimated_state_dim + self.dynamics_latent_dim

  def _normalized_inputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    normalized = self.obs_normalizer(raw)
    offset = 0
    policy_obs = None
    history_obs = None
    for group in self.obs_groups:
      width = int(obs[group].shape[-1])
      value = normalized[..., offset : offset + width]
      if group == self.policy_obs_group:
        policy_obs = value
      elif group == self.history_obs_group:
        history_obs = value
      offset += width
    if policy_obs is None or history_obs is None:
      raise RuntimeError("Implicit-state observation groups were not resolved")
    return policy_obs, history_obs

  def encode_history(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return training-only state prediction and shared history latent."""
    policy_obs, history_obs = self._normalized_inputs(obs)
    dynamics = self._encode_normalized_history(policy_obs, history_obs)
    return self.auxiliary_state_head(dynamics), dynamics

  def _encode_normalized_history(
    self, policy_obs: torch.Tensor, history_obs: torch.Tensor
  ) -> torch.Tensor:
    output = self.history_encoder(torch.cat((policy_obs, history_obs), dim=-1))
    return F.normalize(output, dim=-1)

  def encode_target(self, obs: TensorDict) -> torch.Tensor:
    """Embed training-only one-step proprioception into the dynamics space."""
    target = self.target_obs_normalizer(obs[self.target_proprio_group])
    return F.normalize(self.target_encoder(target), dim=-1)

  def estimate_state(self, obs: TensorDict) -> torch.Tensor:
    """Predict velocity and root-position error without exposing targets."""
    state, _ = self.encode_history(obs)
    return state

  def estimate_velocity(self, obs: TensorDict) -> torch.Tensor:
    return self.estimate_state(obs)[..., : self.estimated_velocity_dim]

  def estimate_root_position_error(self, obs: TensorDict) -> torch.Tensor:
    return self.estimate_state(obs)[..., self.estimated_velocity_dim :]

  def estimate_dynamics_latent(self, obs: TensorDict) -> torch.Tensor:
    policy_obs, history_obs = self._normalized_inputs(obs)
    return self._encode_normalized_history(policy_obs, history_obs)

  def dynamics_contrastive_loss(
    self, history_latent: torch.Tensor, target_latent: torch.Tensor
  ) -> torch.Tensor:
    """Compute HIM-style balanced swapped prototype assignments."""
    prototypes = F.normalize(self.dynamics_prototypes.weight, dim=-1)
    history_scores = history_latent @ prototypes.T
    target_scores = target_latent @ prototypes.T
    with torch.no_grad():
      history_assignments = _sinkhorn_assignments(
        history_scores, self.sinkhorn_epsilon, self.sinkhorn_iterations
      )
      target_assignments = _sinkhorn_assignments(
        target_scores, self.sinkhorn_epsilon, self.sinkhorn_iterations
      )
    history_log_prob = F.log_softmax(
      history_scores / self.dynamics_temperature, dim=-1
    )
    target_log_prob = F.log_softmax(
      target_scores / self.dynamics_temperature, dim=-1
    )
    return -0.5 * (
      history_assignments * target_log_prob
      + target_assignments * history_log_prob
    ).mean()

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    del masks, hidden_state
    policy_obs, history_obs = self._normalized_inputs(obs)
    dynamics = self._encode_normalized_history(policy_obs, history_obs)
    estimated_state = self.auxiliary_state_head(dynamics)
    return torch.cat((policy_obs, estimated_state, dynamics), dim=-1)

  def estimator_parameters(self) -> Iterable[torch.nn.Parameter]:
    yield from self.history_encoder.parameters()
    yield from self.auxiliary_state_head.parameters()
    yield from self.target_encoder.parameters()
    yield from self.dynamics_prototypes.parameters()

  def set_estimator_requires_grad(self, enabled: bool) -> None:
    for parameter in self.estimator_parameters():
      parameter.requires_grad_(enabled)

  def update_normalization(self, obs: TensorDict) -> None:
    super().update_normalization(obs)
    if self.obs_normalization:
      self.target_obs_normalizer.update(obs[self.target_proprio_group])  # type: ignore[attr-defined]

  def as_onnx(self, verbose: bool) -> torch.nn.Module:
    """Export the deployable history encoder, state head, and policy path."""
    return _OnnxImplicitStateActor(self, verbose)


class _OnnxImplicitStateActor(torch.nn.Module):
  """Flat-input deterministic wrapper for ONNX deployment."""

  def __init__(self, actor: ImplicitStateActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.input_size = actor.obs_dim
    self.obs_normalizer = deepcopy(actor.obs_normalizer)
    self.history_encoder = deepcopy(actor.history_encoder)
    self.state_head = deepcopy(actor.auxiliary_state_head)
    self.mlp = deepcopy(actor.mlp)
    if actor.distribution is None:
      self.deterministic_output = torch.nn.Identity()
    else:
      self.deterministic_output = actor.distribution.as_deterministic_output_module()

    offset = 0
    for group in actor.obs_groups:
      width = (
        actor.policy_obs_dim
        if group == actor.policy_obs_group
        else actor.history_obs_dim
      )
      if group == actor.policy_obs_group:
        self.policy_start = offset
        self.policy_stop = offset + width
      elif group == actor.history_obs_group:
        self.history_start = offset
        self.history_stop = offset + width
      offset += width

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    normalized = self.obs_normalizer(observation)
    policy_obs = normalized[..., self.policy_start : self.policy_stop]
    history_obs = normalized[..., self.history_start : self.history_stop]
    dynamics = F.normalize(
      self.history_encoder(torch.cat((policy_obs, history_obs), dim=-1)), dim=-1
    )
    estimated_state = self.state_head(dynamics)
    policy_output = self.mlp(
      torch.cat((policy_obs, estimated_state, dynamics), dim=-1)
    )
    return self.deterministic_output(policy_output)


class ImplicitStatePPO(PPO):
  """Alternate ordinary PPO updates with predictive estimator updates.

  Freezing the estimator during PPO prevents the policy objective from turning
  its outputs into arbitrary policy latents.  The subsequent update supervises
  auxiliary state and predicts actual successor proprioception contrastively.
  """

  actor: ImplicitStateActor

  def __init__(
    self,
    *args,
    implicit_state_target_group: str = "implicit_state_target",
    implicit_state_learning_rate: float = 1.0e-3,
    implicit_state_num_learning_epochs: int = 1,
    implicit_state_num_mini_batches: int = 4,
    implicit_state_max_grad_norm: float = 1.0,
    implicit_dynamics_loss_coef: float = 1.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if not isinstance(self._raw_actor, ImplicitStateActor):
      raise TypeError("ImplicitStatePPO requires ImplicitStateActor")
    self.implicit_state_target_group = implicit_state_target_group
    self.implicit_state_num_learning_epochs = implicit_state_num_learning_epochs
    self.implicit_state_num_mini_batches = implicit_state_num_mini_batches
    self.implicit_state_max_grad_norm = implicit_state_max_grad_norm
    self.implicit_dynamics_loss_coef = implicit_dynamics_loss_coef
    self.implicit_state_optimizer = torch.optim.Adam(
      self._raw_actor.estimator_parameters(),
      lr=implicit_state_learning_rate,
    )

  def update(self) -> dict[str, float]:
    actor = self._raw_actor
    actor.set_estimator_requires_grad(False)
    try:
      loss_dict = super().update()
    finally:
      actor.set_estimator_requires_grad(True)

    observations, successor_observations = _successor_pairs(
      self.storage.observations, self.storage.dones
    )
    sample_count = int(observations.batch_size[0])
    if sample_count == 0:
      raise RuntimeError("No non-terminal successor pairs in estimator rollout")
    mini_batch_count = min(self.implicit_state_num_mini_batches, sample_count)
    mini_batch_size = sample_count // mini_batch_count
    mean_velocity_loss = 0.0
    mean_root_position_error_loss = 0.0
    mean_dynamics_loss = 0.0
    update_count = 0

    for _ in range(self.implicit_state_num_learning_epochs):
      indices = torch.randperm(sample_count, device=self.device)
      for mini_batch_index in range(mini_batch_count):
        start = mini_batch_index * mini_batch_size
        stop = (
          sample_count
          if mini_batch_index == mini_batch_count - 1
          else (mini_batch_index + 1) * mini_batch_size
        )
        batch_indices = indices[start:stop]
        batch = observations[batch_indices]
        successor_batch = successor_observations[batch_indices]
        prediction, history_latent = actor.encode_history(batch)
        target_latent = actor.encode_target(successor_batch)
        target = batch[self.implicit_state_target_group]
        velocity_dim = actor.estimated_velocity_dim
        velocity_loss = torch.nn.functional.mse_loss(
          prediction[..., :velocity_dim], target[..., :velocity_dim]
        )
        root_position_error_loss = torch.nn.functional.mse_loss(
          prediction[..., velocity_dim:], target[..., velocity_dim:]
        )
        dynamics_loss = actor.dynamics_contrastive_loss(
          history_latent, target_latent
        )
        estimator_loss = (
          velocity_loss
          + root_position_error_loss
          + self.implicit_dynamics_loss_coef * dynamics_loss
        )

        self.implicit_state_optimizer.zero_grad()
        estimator_loss.backward()
        if self.is_multi_gpu:
          self._reduce_estimator_gradients()
        torch.nn.utils.clip_grad_norm_(
          actor.estimator_parameters(), self.implicit_state_max_grad_norm
        )
        self.implicit_state_optimizer.step()
        mean_velocity_loss += float(velocity_loss.detach())
        mean_root_position_error_loss += float(root_position_error_loss.detach())
        mean_dynamics_loss += float(dynamics_loss.detach())
        update_count += 1

    loss_dict["velocity_estimator"] = mean_velocity_loss / max(update_count, 1)
    loss_dict["root_position_error_estimator"] = (
      mean_root_position_error_loss / max(update_count, 1)
    )
    loss_dict["dynamics_contrastive"] = mean_dynamics_loss / max(update_count, 1)
    return loss_dict

  def _reduce_estimator_gradients(self) -> None:
    """Average only estimator gradients across distributed workers."""
    parameters = [
      parameter
      for parameter in self._raw_actor.estimator_parameters()
      if parameter.grad is not None
    ]
    gradients = torch.cat([parameter.grad.view(-1) for parameter in parameters])
    torch.distributed.all_reduce(gradients, op=torch.distributed.ReduceOp.SUM)
    gradients /= self.gpu_world_size
    offset = 0
    for parameter in parameters:
      size = parameter.numel()
      parameter.grad.copy_(gradients[offset : offset + size].view_as(parameter))
      offset += size

  def save(self) -> dict:
    state = super().save()
    state["implicit_state_optimizer_state_dict"] = (
      self.implicit_state_optimizer.state_dict()
    )
    return state

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    should_load_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
    if (
      should_load_optimizer
      and "implicit_state_optimizer_state_dict" in loaded_dict
    ):
      self.implicit_state_optimizer.load_state_dict(
        loaded_dict["implicit_state_optimizer_state_dict"]
      )
    return load_iteration


@torch.no_grad()
def _sinkhorn_assignments(
  scores: torch.Tensor, epsilon: float, iterations: int
) -> torch.Tensor:
  """Return batch-balanced prototype assignments for contrastive prediction."""
  shifted_scores = scores.float() - scores.float().max()
  assignments = torch.exp(shifted_scores / epsilon).T
  prototype_count, batch_size = assignments.shape
  tiny = torch.finfo(assignments.dtype).tiny
  assignments /= assignments.sum().clamp_min(tiny)
  for _ in range(iterations):
    assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(tiny)
    assignments /= prototype_count
    assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(tiny)
    assignments /= batch_size
  return (assignments * batch_size).T.to(dtype=scores.dtype)


def _successor_pairs(
  observations: TensorDict, dones: torch.Tensor
) -> tuple[TensorDict, TensorDict]:
  """Pair rollout observations with real successors, excluding reset edges."""
  if len(observations.batch_size) != 2 or observations.batch_size[0] < 2:
    raise ValueError("Successor prediction requires a [time, env] rollout")
  current = observations[:-1].flatten(0, 1)
  successor = observations[1:].flatten(0, 1)
  valid = ~dones[:-1].flatten(0, 1).squeeze(-1).bool()
  return current[valid], successor[valid]
