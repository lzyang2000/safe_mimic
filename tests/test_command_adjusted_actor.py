"""Tests for the co-trained command-slice adjustment path."""

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from safe_mimic.rl import AvoidanceAuxiliaryPPO, PerceptiveLidarActor

_OFFSET = 7
_JOINT_DIM = 29


def _gaussian_cfg() -> dict:
  # Fresh dict per use: rsl_rl consumes keys from the distribution cfg.
  return {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }


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


def _adjusting_actor(
  obs: TensorDict,
  *,
  obs_normalization: bool = False,
  distribution_cfg: dict | None = None,
  residual_gain: float = 0.0,
  adjust: bool = True,
  offset: int = _OFFSET,
) -> PerceptiveLidarActor:
  residual_kwargs = {}
  if residual_gain:
    residual_kwargs = {
      "avoidance_joint_action_residual_gain": residual_gain,
      "avoidance_joint_action_scales": [0.25] * _JOINT_DIM,
      "avoidance_joint_action_mask": [True] * _JOINT_DIM,
    }
  return PerceptiveLidarActor(
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
    avoidance_joint_dim=_JOINT_DIM,
    avoidance_hidden_dims=(32,),
    obs_normalization=obs_normalization,
    distribution_cfg=distribution_cfg,
    command_joint_pos_offset=offset,
    adjust_command_with_joint_prediction=adjust,
    **residual_kwargs,
  )


class _CapturingEncoder(torch.nn.Module):
  """Wrap an encoder and record a detached copy of every input."""

  def __init__(self, inner: torch.nn.Module) -> None:
    super().__init__()
    self.inner = inner
    self.inputs: list[torch.Tensor] = []

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    self.inputs.append(value.detach().clone())
    return self.inner(value)


def test_forward_injects_prediction_into_command_slice_only() -> None:
  obs = _observations()
  raw_tracking = obs["actor"].clone()
  actor = _adjusting_actor(obs)
  prediction = actor.predict_avoidance(obs).detach()
  capture = _CapturingEncoder(actor.tracking_encoder)
  actor.tracking_encoder = capture

  actor(obs)

  assert len(capture.inputs) == 2
  adjuster_pass, mimic_pass = capture.inputs
  torch.testing.assert_close(adjuster_pass, raw_tracking)
  expected = raw_tracking.clone()
  expected[..., _OFFSET : _OFFSET + _JOINT_DIM] += prediction[..., 2:]
  torch.testing.assert_close(mimic_pass, expected)
  torch.testing.assert_close(mimic_pass[..., :_OFFSET], raw_tracking[..., :_OFFSET])
  torch.testing.assert_close(
    mimic_pass[..., _OFFSET + _JOINT_DIM :],
    raw_tracking[..., _OFFSET + _JOINT_DIM :],
  )
  # The stored rollout observation is never mutated by the injection.
  torch.testing.assert_close(obs["actor"], raw_tracking)


class _CapturingMlp(torch.nn.Module):
  """Wrap the policy MLP and record a detached copy of every input."""

  def __init__(self, inner: torch.nn.Module) -> None:
    super().__init__()
    self.inner = inner
    self.inputs: list[torch.Tensor] = []

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    self.inputs.append(value.detach().clone())
    return self.inner(value)


def test_adjusted_mode_concats_planar_compass_but_not_joints() -> None:
  obs = _observations()
  actor = _adjusting_actor(obs)
  prediction = actor.predict_avoidance(obs).detach()

  # Joint part is superseded by the command injection (width 59, not 88);
  # the planar compass keeps its 2-D concat.
  assert actor.mlp[0].in_features == 32 + 24 + 1 + 2
  torch.testing.assert_close(actor.mlp(actor.get_latent(obs)), actor(obs))

  capture = _CapturingMlp(actor.mlp)
  actor.mlp = capture
  actor(obs)
  torch.testing.assert_close(capture.inputs[-1][..., -2:], prediction[..., :2])


@pytest.mark.parametrize("adjust", [True, False])
def test_update_reconstruction_matches_rollout_log_prob(adjust: bool) -> None:
  batch_size = 4
  obs = _observations(batch_size)
  obs["avoidance_teacher"] = torch.randn(batch_size, 31).clamp(-1.0, 1.0)
  # Nonzero stored noise: it lives in the observation group, so the same
  # TensorDict must reproduce the same conditioning at update time.
  obs["avoidance_robustness"] = (
    torch.rand(batch_size, 31) * 2.0 - 1.0
  )
  actor = _adjusting_actor(
    obs,
    distribution_cfg=_gaussian_cfg(),
    adjust=adjust,
    offset=_OFFSET if adjust else -1,
  )
  critic = MLPModel(
    obs,
    {"critic": ["critic"]},
    "critic",
    output_dim=1,
    hidden_dims=(32,),
  )
  storage = RolloutStorage("rl", batch_size, 1, obs, [29])
  algorithm = AvoidanceAuxiliaryPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    avoidance_num_learning_epochs=1,
    avoidance_num_mini_batches=1,
    desired_kl=None,
    avoidance_teacher_mix_start=0.5,
    avoidance_teacher_mix_end=0.5,
  )

  with torch.no_grad():
    stored_actions = algorithm.act(obs)
    stored_log_prob = algorithm.transition.actions_log_prob.clone()
    # Frozen parameters: replay exactly what update() runs before computing
    # PPO's importance ratio.
    algorithm._rollout_action_distribution(obs)  # noqa: SLF001
    replayed_log_prob = actor.get_output_log_prob(stored_actions)

  ratio = torch.exp(replayed_log_prob - stored_log_prob)
  torch.testing.assert_close(
    ratio, torch.ones_like(ratio), atol=1.0e-6, rtol=1.0e-6
  )


@pytest.mark.parametrize("adjust", [True, False])
def test_real_update_reproduces_rollout_log_prob(adjust: bool) -> None:
  """End-to-end: the log-prob computed INSIDE update() matches the rollout.

  The shared-method replay above cannot catch an update() that bypasses the
  shared routing; this test captures actor.get_output_log_prob during the
  real update() and checks PPO's importance ratio is exactly one for the
  stored on-policy transitions.
  """
  batch_size = 4
  obs = _observations(batch_size)
  obs["avoidance_teacher"] = torch.randn(batch_size, 31).clamp(-1.0, 1.0)
  obs["avoidance_robustness"] = torch.rand(batch_size, 31) * 2.0 - 1.0
  actor = _adjusting_actor(
    obs,
    distribution_cfg=_gaussian_cfg(),
    adjust=adjust,
    offset=_OFFSET if adjust else -1,
  )
  critic = MLPModel(
    obs,
    {"critic": ["critic"]},
    "critic",
    output_dim=1,
    hidden_dims=(32,),
  )
  storage = RolloutStorage("rl", batch_size, 1, obs, [29])
  algorithm = AvoidanceAuxiliaryPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    avoidance_num_learning_epochs=1,
    avoidance_num_mini_batches=1,
    desired_kl=None,
    avoidance_teacher_mix_start=0.5,
    avoidance_teacher_mix_end=0.5,
  )
  log_prob_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
  original_log_prob = actor.get_output_log_prob

  def capturing(actions: torch.Tensor) -> torch.Tensor:
    value = original_log_prob(actions)
    log_prob_calls.append((actions.detach().clone(), value.detach().clone()))
    return value

  actor.get_output_log_prob = capturing
  stored_actions = algorithm.act(obs)
  stored_log_prob = algorithm.transition.actions_log_prob.detach().flatten()
  algorithm.process_env_step(
    obs,
    torch.ones(batch_size),
    torch.zeros(batch_size, dtype=torch.bool),
    {},
  )
  algorithm.compute_returns(obs)
  algorithm.update()

  # One epoch, one minibatch: the last capture is update()'s reconstruction.
  update_actions, update_log_prob = log_prob_calls[-1]
  update_actions = update_actions.reshape(-1, stored_actions.shape[-1])
  update_log_prob = update_log_prob.flatten()
  for index in range(batch_size):
    matches = (
      (update_actions == stored_actions[index]).all(dim=-1).nonzero().flatten()
    )
    assert len(matches) == 1
    ratio = torch.exp(update_log_prob[matches[0]] - stored_log_prob[index])
    torch.testing.assert_close(
      ratio, torch.ones_like(ratio), atol=1.0e-5, rtol=1.0e-5
    )


def test_override_injects_teacher_joints_and_residual_stays_inert() -> None:
  obs = _observations(batch_size=1)
  raw_tracking = obs["actor"].clone()
  actor = _adjusting_actor(obs, residual_gain=0.5)
  capture = _CapturingEncoder(actor.tracking_encoder)
  actor.tracking_encoder = capture
  teacher = torch.zeros(1, 31)
  teacher[0, 0] = 0.3
  teacher[0, 2 + 4] = 0.4

  for parameter in actor.mlp.parameters():
    parameter.data.zero_()
  action = actor.action_with_avoidance_override(obs, teacher)

  assert len(capture.inputs) == 1
  expected = raw_tracking.clone()
  expected[..., _OFFSET + 4] += 0.4
  # Only the joint block enters the command; the planar part is ignored here.
  torch.testing.assert_close(capture.inputs[0], expected)
  # Despite the configured 0.5 residual gain, adjust mode keeps the explicit
  # action residual inert, so a zeroed policy head yields exactly zero.
  torch.testing.assert_close(action, torch.zeros_like(action))


def test_algorithm_act_routes_full_mix_teacher_into_command() -> None:
  batch_size = 4
  obs = _observations(batch_size)
  raw_tracking = obs["actor"].clone()
  teacher = torch.randn(batch_size, 31).clamp(-1.0, 1.0)
  obs["avoidance_teacher"] = teacher
  obs["avoidance_robustness"] = torch.zeros(batch_size, 31)
  actor = _adjusting_actor(obs, distribution_cfg=_gaussian_cfg())
  critic = MLPModel(
    obs,
    {"critic": ["critic"]},
    "critic",
    output_dim=1,
    hidden_dims=(32,),
  )
  storage = RolloutStorage("rl", batch_size, 1, obs, [29])
  algorithm = AvoidanceAuxiliaryPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    avoidance_num_learning_epochs=1,
    avoidance_num_mini_batches=1,
    desired_kl=None,
    avoidance_teacher_mix_start=1.0,
    avoidance_teacher_mix_end=1.0,
  )
  capture = _CapturingEncoder(actor.tracking_encoder)
  actor.tracking_encoder = capture

  algorithm.act(obs)

  # First pass is the raw adjuster encoding; the action pass must see the
  # teacher target injected exactly (mix 1.0, zero stored noise, |t| < scale).
  expected = raw_tracking.clone()
  expected[..., _OFFSET : _OFFSET + _JOINT_DIM] += teacher[..., 2:]
  torch.testing.assert_close(capture.inputs[0], raw_tracking)
  torch.testing.assert_close(capture.inputs[-1], expected)


def test_adjusted_update_shares_optimizer_and_reaches_adjuster() -> None:
  batch_size = 4
  obs = _observations(batch_size)
  obs["avoidance_teacher"] = torch.randn(batch_size, 31) * 0.1
  obs["avoidance_robustness"] = torch.zeros(batch_size, 31)
  actor = _adjusting_actor(obs, distribution_cfg=_gaussian_cfg())
  critic = MLPModel(
    obs,
    {"critic": ["critic"]},
    "critic",
    output_dim=1,
    hidden_dims=(32,),
  )
  storage = RolloutStorage("rl", batch_size, 1, obs, [29])
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


def test_adjusted_actor_export_parity(tmp_path) -> None:
  obs = _observations()
  actor = _adjusting_actor(obs, obs_normalization=True)
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


def test_adjust_mode_keeps_checkpoint_keys_and_validates_cfg() -> None:
  obs = _observations()
  adjusted = _adjusting_actor(obs)
  baseline = _adjusting_actor(obs, adjust=False, offset=-1)

  assert set(adjusted.state_dict()) == set(baseline.state_dict())

  with pytest.raises(ValueError, match="command_joint_pos_offset"):
    _adjusting_actor(obs, offset=-1)
  with pytest.raises(ValueError, match="command_joint_pos_offset"):
    _adjusting_actor(obs, offset=154 - _JOINT_DIM + 1)
  with pytest.raises(ValueError, match="joint avoidance prediction"):
    PerceptiveLidarActor(
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
      command_joint_pos_offset=_OFFSET,
      adjust_command_with_joint_prediction=True,
    )
