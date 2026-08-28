"""Tests for the implicit translational-state estimator."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from safe_mimic.rl import ImplicitStateActor
from safe_mimic.rl.implicit_estimator import _successor_pairs


def _observations(batch_size: int = 8) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(batch_size, 154),
      "proprio_history": torch.randn(batch_size, 930),
      "critic": torch.randn(batch_size, 286),
      "implicit_state_target": torch.randn(batch_size, 6),
      "implicit_proprio_target": torch.randn(batch_size, 93),
    },
    batch_size=[batch_size],
  )


def test_actor_consumes_predicted_state_and_dynamics_latent() -> None:
  obs = _observations()
  actor = ImplicitStateActor(
    obs,
    {"actor": ["actor", "proprio_history"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    estimator_hidden_dims=(32, 16),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )

  assert actor.mlp[0].in_features == 176
  assert actor.estimate_state(obs).shape == (8, 6)
  assert actor.estimate_velocity(obs).shape == (8, 3)
  assert actor.estimate_root_position_error(obs).shape == (8, 3)
  assert actor.estimate_dynamics_latent(obs).shape == (8, 16)
  assert actor.encode_target(obs).shape == (8, 16)
  assert actor(obs).shape == (8, 29)


def test_policy_path_uses_state_prediction_head() -> None:
  obs = _observations()
  actor = ImplicitStateActor(
    obs,
    {"actor": ["actor", "proprio_history"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    estimator_hidden_dims=(32, 16),
  )

  actor(obs).sum().backward()

  assert any(
    parameter.grad is not None for parameter in actor.history_encoder.parameters()
  )
  assert any(
    parameter.grad is not None
    for parameter in actor.auxiliary_state_head.parameters()
  )


def test_explicit_state_supervision_updates_encoder_and_auxiliary_head() -> None:
  obs = _observations()
  actor = ImplicitStateActor(
    obs,
    {"actor": ["actor", "proprio_history"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    estimator_hidden_dims=(32, 16),
  )
  loss = torch.nn.functional.mse_loss(
    actor.estimate_state(obs), obs["implicit_state_target"]
  )
  loss.backward()

  assert any(
    parameter.grad is not None for parameter in actor.history_encoder.parameters()
  )
  assert any(
    parameter.grad is not None
    for parameter in actor.auxiliary_state_head.parameters()
  )
  assert all(parameter.grad is None for parameter in actor.target_encoder.parameters())
  assert actor.dynamics_prototypes.weight.grad is None
  assert all(parameter.grad is None for parameter in actor.mlp.parameters())


def test_contrastive_loss_updates_both_encoders_and_prototypes() -> None:
  obs = _observations()
  successor_obs = _observations()
  actor = ImplicitStateActor(
    obs,
    {"actor": ["actor", "proprio_history"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    estimator_hidden_dims=(32, 16),
    target_encoder_hidden_dims=(32, 16),
  )
  _, history_latent = actor.encode_history(obs)
  target_latent = actor.encode_target(successor_obs)
  actor.dynamics_contrastive_loss(history_latent, target_latent).backward()

  assert any(
    parameter.grad is not None for parameter in actor.history_encoder.parameters()
  )
  assert any(
    parameter.grad is not None for parameter in actor.target_encoder.parameters()
  )
  assert actor.dynamics_prototypes.weight.grad is not None
  assert all(parameter.grad is None for parameter in actor.mlp.parameters())


def test_policy_gradient_stops_at_frozen_estimator_outputs() -> None:
  obs = _observations()
  actor = ImplicitStateActor(
    obs,
    {"actor": ["actor", "proprio_history"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    estimator_hidden_dims=(32, 16),
    target_encoder_hidden_dims=(32, 16),
  )
  actor.set_estimator_requires_grad(False)

  actor(obs).sum().backward()

  assert any(parameter.grad is not None for parameter in actor.mlp.parameters())
  assert all(parameter.grad is None for parameter in actor.estimator_parameters())


def test_onnx_wrapper_preserves_deployable_actor_path() -> None:
  obs = _observations()
  actor = ImplicitStateActor(
    obs,
    {"actor": ["actor", "proprio_history"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    estimator_hidden_dims=(32, 16),
    target_encoder_hidden_dims=(32, 16),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  exported = actor.as_onnx(verbose=False)
  flat_observation = torch.cat((obs["actor"], obs["proprio_history"]), dim=-1)

  assert exported.input_size == 1084
  assert torch.allclose(exported(flat_observation), actor(obs), atol=1.0e-6)


def test_successor_pairs_exclude_terminal_reset_edges() -> None:
  observations = TensorDict(
    {"step": torch.arange(12).reshape(3, 2, 2)}, batch_size=[3, 2]
  )
  dones = torch.zeros(3, 2, 1, dtype=torch.uint8)
  dones[0, 1] = 1

  current, successor = _successor_pairs(observations, dones)

  assert current["step"].tolist() == [[0, 1], [4, 5], [6, 7]]
  assert successor["step"].tolist() == [[4, 5], [8, 9], [10, 11]]
