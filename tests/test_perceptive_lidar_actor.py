"""Tests for the deployable circular LiDAR policy path."""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from safe_mimic.rl import AvoidanceAuxiliaryPPO, PerceptiveLidarActor
from safe_mimic.rl.perceptive_lidar import _avoidance_supervision_losses


def _observations(batch_size: int = 4) -> TensorDict:
  lidar_dim = 2 * 3 * 6 + 1
  return TensorDict(
    {
      "actor": torch.randn(batch_size, 154),
      "lidar": torch.rand(batch_size, lidar_dim),
      "critic": torch.randn(batch_size, 286),
    },
    batch_size=[batch_size],
  )


def _actor(obs: TensorDict) -> PerceptiveLidarActor:
  return PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_channels=(8, 16),
    lidar_kernel_sizes=(5, 3),
    lidar_strides=(2, 2),
    lidar_direction_channels=4,
    lidar_direction_bins=6,
    lidar_latent_dim=24,
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def test_perceptive_actor_fuses_tracking_and_two_scan_channels() -> None:
  obs = _observations()
  actor = _actor(obs)

  output = actor(obs)

  assert output.shape == (4, 29)
  assert actor.mlp[0].in_features == 57
  assert actor.obs_normalizer._mean.shape == (1, 154)
  assert actor.lidar_encoder.scan_count == 2
  assert actor.lidar_encoder.elevation_samples == 27
  assert actor.lidar_encoder.direction_bins == 6
  assert actor.lidar_encoder.elevation_bins == 3
  assert actor.lidar_encoder.projection.in_features == 2 * 3 * 6
  assert "avoidance_output_scale" not in actor.state_dict()


def test_perceptive_actor_retains_obstacle_azimuth() -> None:
  obs = _observations(batch_size=1)
  actor = _actor(obs)
  encoder = actor.lidar_encoder
  scan = torch.ones(1, 2, 3, 6)
  scan[..., 1] = 0.2
  rotated = torch.roll(scan, shifts=2, dims=-1)

  assert not torch.allclose(
    encoder(scan.flatten(start_dim=1)),
    encoder(rotated.flatten(start_dim=1)),
  )


def test_policy_gradient_reaches_both_tracking_and_lidar_encoders() -> None:
  obs = _observations()
  actor = _actor(obs)

  actor(obs).sum().backward()

  assert any(
    parameter.grad is not None for parameter in actor.tracking_encoder.parameters()
  )
  assert any(
    parameter.grad is not None for parameter in actor.lidar_encoder.parameters()
  )
  assert any(parameter.grad is not None for parameter in actor.mlp.parameters())


def test_auxiliary_correction_head_is_deployable_and_used_by_policy() -> None:
  obs = _observations()
  actor = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=24,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(32,),
  )

  prediction = actor.predict_avoidance(obs)

  assert prediction.shape == (4, 31)
  assert actor.mlp[0].in_features == 88
  assert torch.all(torch.abs(prediction) <= 1.5)
  assert "avoidance_output_scale" in actor.state_dict()
  actor(obs).sum().backward()
  assert any(
    parameter.grad is not None for parameter in actor.avoidance_head.parameters()
  )


def test_auxiliary_policy_accepts_zero_and_oracle_correction_ablations() -> None:
  obs = _observations()
  actor = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=24,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(32,),
  )
  predicted = actor.predict_avoidance(obs)
  zero = torch.zeros_like(predicted)
  oracle = torch.randn_like(predicted)

  torch.testing.assert_close(
    actor.action_with_avoidance_override(obs, predicted),
    actor(obs),
  )
  assert actor.action_with_avoidance_override(obs, zero).shape == (4, 29)
  assert actor.action_with_avoidance_override(obs, oracle).shape == (4, 29)


def test_auxiliary_arm_residual_is_applied_in_joint_action_units() -> None:
  obs = _observations(batch_size=1)
  mask = [False] * 29
  mask[16] = True
  actor = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(32,),
    tracking_hidden_dims=(32,),
    tracking_latent_dim=16,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=16,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(16,),
    avoidance_joint_action_residual_gain=0.5,
    avoidance_joint_action_scales=[0.25] * 29,
    avoidance_joint_action_mask=mask,
  )
  for parameter in actor.mlp.parameters():
    parameter.data.zero_()
  correction = torch.zeros(1, 31)
  correction[0, 2 + 16] = 0.4
  correction[0, 2] = 0.4

  action = actor.action_with_avoidance_override(obs, correction)

  expected = torch.zeros_like(action)
  expected[0, 16] = 0.5 * 0.4 / 0.25
  torch.testing.assert_close(action, expected)
  flat = torch.cat((obs["actor"], obs["lidar"]), dim=-1)
  torch.testing.assert_close(actor.as_onnx(verbose=False)(flat), actor(obs))


def test_direct_joint_residual_does_not_change_checkpoint_contract() -> None:
  obs = _observations()
  baseline = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=24,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(32,),
  )
  residual = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=24,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(32,),
    avoidance_joint_action_residual_gain=0.5,
    avoidance_joint_action_scales=[0.25] * 29,
    avoidance_joint_action_mask=[True] * 29,
  )

  residual.load_state_dict(baseline.state_dict(), strict=True)

  assert set(residual.state_dict()) == set(baseline.state_dict())


def test_sparse_active_joint_gets_elementwise_weight() -> None:
  prediction = torch.zeros(1, 4)
  target = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

  _, uniform_joint_loss, _ = _avoidance_supervision_losses(
    prediction,
    target,
    planar_dim=2,
    active_sample_weight=1.0,
    active_joint_weight=1.0,
    active_threshold=1.0e-3,
    active_joint_threshold=1.0e-3,
  )
  _, weighted_joint_loss, diagnostics = _avoidance_supervision_losses(
    prediction,
    target,
    planar_dim=2,
    active_sample_weight=1.0,
    active_joint_weight=12.0,
    active_threshold=1.0e-3,
    active_joint_threshold=1.0e-3,
  )

  assert weighted_joint_loss > uniform_joint_loss
  torch.testing.assert_close(
    diagnostics["joint_active_fraction"], torch.tensor(0.5)
  )
  torch.testing.assert_close(
    diagnostics["joint_active_mae"], torch.tensor(1.0)
  )


def test_auxiliary_loss_shares_the_ppo_optimizer_step() -> None:
  batch_size = 4
  obs = _observations(batch_size)
  obs["avoidance_teacher"] = torch.randn(batch_size, 31) * 0.1
  obs["avoidance_robustness"] = torch.zeros(batch_size, 31)
  actor = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=24,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(32,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  critic = MLPModel(
    obs,
    {"critic": ["critic"]},
    "critic",
    output_dim=1,
    hidden_dims=(32,),
  )
  storage = RolloutStorage(
    "rl",
    batch_size,
    1,
    obs,
    [29],
  )
  algorithm = AvoidanceAuxiliaryPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    avoidance_num_learning_epochs=1,
    avoidance_num_mini_batches=1,
    desired_kl=None,
  )
  step_count = 0
  original_step = algorithm.optimizer.step

  def counted_step(*args, **kwargs):
    nonlocal step_count
    step_count += 1
    return original_step(*args, **kwargs)

  algorithm.optimizer.step = counted_step
  algorithm.act(obs)
  algorithm.process_env_step(
    obs,
    torch.ones(batch_size),
    torch.zeros(batch_size, dtype=torch.bool),
    {},
  )
  algorithm.compute_returns(obs)

  losses = algorithm.update()

  assert step_count == 1
  assert losses["avoidance_joint_active_mae"] > 0.0
  assert any(parameter.grad is not None for parameter in actor.mlp.parameters())
  assert any(
    parameter.grad is not None for parameter in actor.avoidance_head.parameters()
  )


def test_auxiliary_loss_reaches_deployable_encoders_but_not_policy_head() -> None:
  obs = _observations()
  actor = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=24,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(32,),
  )

  torch.nn.functional.smooth_l1_loss(
    actor.predict_avoidance(obs), torch.randn(4, 31)
  ).backward()

  assert any(
    parameter.grad is not None for parameter in actor.tracking_encoder.parameters()
  )
  assert any(
    parameter.grad is not None for parameter in actor.lidar_encoder.parameters()
  )
  assert any(
    parameter.grad is not None for parameter in actor.avoidance_head.parameters()
  )
  assert all(parameter.grad is None for parameter in actor.mlp.parameters())


def test_perceptive_actor_export_preserves_flat_policy_path(tmp_path) -> None:
  obs = _observations()
  actor = _actor(obs)
  exported = actor.as_onnx(verbose=False)
  flat = torch.cat((obs["actor"], obs["lidar"]), dim=-1)

  assert exported.input_size == 191
  torch.testing.assert_close(exported(flat), actor(obs), atol=1.0e-6, rtol=1.0e-6)
  torch.onnx.export(
    exported,
    flat[:1],
    str(tmp_path / "policy.onnx"),
    input_names=exported.input_names,
    output_names=exported.output_names,
    opset_version=17,
  )


def test_auxiliary_actor_export_uses_no_privileged_input() -> None:
  obs = _observations()
  actor = PerceptiveLidarActor(
    obs,
    {"actor": ["actor", "lidar"]},
    "actor",
    output_dim=29,
    hidden_dims=(64, 32),
    tracking_hidden_dims=(64,),
    tracking_latent_dim=32,
    lidar_direction_bins=6,
    lidar_elevation_bins=3,
    lidar_latent_dim=24,
    avoidance_planar_dim=2,
    avoidance_joint_dim=29,
    avoidance_hidden_dims=(32,),
  )
  exported = actor.as_onnx(verbose=False)
  flat = torch.cat((obs["actor"], obs["lidar"]), dim=-1)

  assert exported.input_size == 191
  torch.testing.assert_close(exported(flat), actor(obs), atol=1.0e-6, rtol=1.0e-6)
