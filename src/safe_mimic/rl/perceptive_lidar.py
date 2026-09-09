"""Deployable tracking actor with a circular-azimuth LiDAR encoder."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict


@dataclass
class PerceptiveLidarModelCfg(RslRlModelCfg):
  """Configuration for the tracking-plus-held-scan policy."""

  class_name: str = "safe_mimic.rl:PerceptiveLidarActor"
  tracking_obs_group: str = "actor"
  lidar_obs_group: str = "lidar"
  azimuth_samples: int = 185
  elevation_samples: int = 27
  scan_count: int = 2
  timing_dim: int = 1
  tracking_hidden_dims: tuple[int, ...] = (256, 128)
  tracking_latent_dim: int = 128
  lidar_channels: tuple[int, ...] = (32, 64, 64)
  lidar_kernel_sizes: tuple[int, ...] = (7, 5, 3)
  lidar_strides: tuple[int, ...] = (2, 2, 1)
  lidar_direction_channels: int = 8
  lidar_direction_bins: int = 120
  lidar_elevation_bins: int = 9
  lidar_latent_dim: int = 128
  avoidance_planar_dim: int = 0
  avoidance_joint_dim: int = 0
  avoidance_hidden_dims: tuple[int, ...] = (128,)
  avoidance_planar_output_scale: float = 1.5
  avoidance_joint_output_scale: float = 1.5
  avoidance_joint_action_residual_gain: float = 0.0
  avoidance_joint_action_scales: tuple[float, ...] = ()
  avoidance_joint_action_mask: tuple[bool, ...] = ()
  command_joint_pos_offset: int = -1
  adjust_command_with_joint_prediction: bool = False


@dataclass
class AvoidanceAuxiliaryPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Ordinary PPO plus training-only privileged reference-filter labels."""

  class_name: str = "safe_mimic.rl:AvoidanceAuxiliaryPPO"
  avoidance_target_group: str = "avoidance_teacher"
  avoidance_noise_group: str = "avoidance_robustness"
  avoidance_num_learning_epochs: int = 1
  avoidance_num_mini_batches: int = 4
  avoidance_planar_loss_coef: float = 1.0
  avoidance_joint_loss_coef: float = 1.0
  avoidance_active_sample_weight: float = 4.0
  avoidance_active_joint_weight: float = 12.0
  avoidance_active_threshold: float = 1.0e-3
  avoidance_active_joint_threshold: float = 1.0e-3
  avoidance_teacher_mix_start: float = 0.1
  avoidance_teacher_mix_end: float = 0.0
  avoidance_teacher_mix_decay_updates: int = 5000
  avoidance_conditioning_noise_scale: float = 0.03


class CircularAzimuthEncoder(torch.nn.Module):
  """Mix scan/elevation channels while preserving 360-degree continuity."""

  def __init__(
    self,
    input_channels: int,
    output_dim: int,
    channels: tuple[int, ...] | list[int],
    kernel_sizes: tuple[int, ...] | list[int],
    strides: tuple[int, ...] | list[int],
    direction_channels: int,
    direction_bins: int,
    activation: str,
  ) -> None:
    super().__init__()
    if not (len(channels) == len(kernel_sizes) == len(strides)):
      raise ValueError("LiDAR channels, kernels, and strides must have equal lengths")
    if not channels:
      raise ValueError("LiDAR encoder requires at least one convolution")
    if direction_channels < 1:
      raise ValueError("LiDAR encoder direction channels must be positive")
    if direction_bins < 2:
      raise ValueError("LiDAR encoder requires at least two direction bins")
    activation_cls = {
      "elu": torch.nn.ELU,
      "relu": torch.nn.ReLU,
      "selu": torch.nn.SELU,
      "tanh": torch.nn.Tanh,
    }.get(activation.lower())
    if activation_cls is None:
      raise ValueError(f"unsupported LiDAR activation: {activation}")
    layers: list[torch.nn.Module] = []
    previous_channels = input_channels
    for output_channels, kernel_size, stride in zip(
      channels, kernel_sizes, strides, strict=True
    ):
      if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("LiDAR kernels must be positive odd integers")
      layers.extend(
        (
          torch.nn.Conv1d(
            previous_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            padding_mode="circular",
          ),
          activation_cls(),
        )
      )
      previous_channels = output_channels
    self.convolutions = torch.nn.Sequential(*layers)
    self.direction_compression = torch.nn.Sequential(
      torch.nn.Conv1d(previous_channels, direction_channels, kernel_size=1),
      activation_cls(),
    )
    self.direction_resample = torch.nn.Upsample(
      size=direction_bins,
      mode="linear",
      align_corners=False,
    )
    self.direction_bins = direction_bins
    self.projection = torch.nn.Linear(
      direction_channels * direction_bins,
      output_dim,
    )
    self.activation = activation_cls()

  def forward(self, scan: torch.Tensor) -> torch.Tensor:
    features = self.convolutions(scan)
    directional = self.direction_resample(self.direction_compression(features)).flatten(
      start_dim=-2
    )
    return self.activation(self.projection(directional))


class DirectionalFeatureEncoder(torch.nn.Module):
  """Project pre-pooled closest-return direction cells."""

  def __init__(
    self,
    *,
    scan_count: int,
    elevation_samples: int,
    azimuth_samples: int,
    elevation_bins: int,
    direction_bins: int,
    output_dim: int,
    activation: str,
  ) -> None:
    super().__init__()
    activation_cls = {
      "elu": torch.nn.ELU,
      "relu": torch.nn.ReLU,
      "selu": torch.nn.SELU,
      "tanh": torch.nn.Tanh,
    }.get(activation.lower())
    if activation_cls is None:
      raise ValueError(f"unsupported LiDAR activation: {activation}")
    if scan_count < 1:
      raise ValueError("LiDAR scan count must be positive")
    if elevation_bins < 1 or direction_bins < 1:
      raise ValueError("LiDAR directional bins must be positive")
    self.scan_count = scan_count
    self.elevation_samples = elevation_samples
    self.azimuth_samples = azimuth_samples
    self.elevation_bins = elevation_bins
    self.direction_bins = direction_bins
    self.projection = torch.nn.Linear(
      scan_count * elevation_bins * direction_bins,
      output_dim,
    )
    self.activation = activation_cls()

  def forward(self, directional: torch.Tensor) -> torch.Tensor:
    return self.activation(self.projection(directional))


class PerceptiveLidarActor(MLPModel):
  """Fuse nominal tracking features with two directional held LiDAR scans."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    tracking_obs_group: str = "actor",
    lidar_obs_group: str = "lidar",
    azimuth_samples: int = 185,
    elevation_samples: int = 27,
    scan_count: int = 2,
    timing_dim: int = 1,
    tracking_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    tracking_latent_dim: int = 128,
    lidar_channels: tuple[int, ...] | list[int] = (32, 64, 64),
    lidar_kernel_sizes: tuple[int, ...] | list[int] = (7, 5, 3),
    lidar_strides: tuple[int, ...] | list[int] = (2, 2, 1),
    lidar_direction_channels: int = 8,
    lidar_direction_bins: int = 24,
    lidar_elevation_bins: int = 3,
    lidar_latent_dim: int = 128,
    avoidance_planar_dim: int = 0,
    avoidance_joint_dim: int = 0,
    avoidance_hidden_dims: tuple[int, ...] | list[int] = (128,),
    avoidance_planar_output_scale: float = 1.5,
    avoidance_joint_output_scale: float = 1.5,
    avoidance_joint_action_residual_gain: float = 0.0,
    avoidance_joint_action_scales: tuple[float, ...] | list[float] = (),
    avoidance_joint_action_mask: tuple[bool, ...] | list[bool] = (),
    command_joint_pos_offset: int = -1,
    adjust_command_with_joint_prediction: bool = False,
  ) -> None:
    active_groups = list(obs_groups[obs_set])
    if active_groups != [tracking_obs_group, lidar_obs_group]:
      raise ValueError(
        "PerceptiveLidarActor expects observation groups "
        f"[{tracking_obs_group!r}, {lidar_obs_group!r}], got {active_groups}"
      )
    self.tracking_obs_group = tracking_obs_group
    self.lidar_obs_group = lidar_obs_group
    self.tracking_obs_dim = int(obs[tracking_obs_group].shape[-1])
    self.lidar_obs_dim = int(obs[lidar_obs_group].shape[-1])
    self.azimuth_samples = azimuth_samples
    self.elevation_samples = elevation_samples
    self.scan_count = scan_count
    self.timing_dim = timing_dim
    self.directional_value_dim = (
      scan_count * lidar_elevation_bins * lidar_direction_bins
    )
    if self.lidar_obs_dim != self.directional_value_dim + timing_dim:
      raise ValueError(
        f"LiDAR group has {self.lidar_obs_dim} values; expected "
        f"{self.directional_value_dim + timing_dim}"
      )
    self.tracking_latent_dim = tracking_latent_dim
    self.lidar_latent_dim = lidar_latent_dim
    if (avoidance_planar_dim == 0) != (avoidance_joint_dim == 0):
      raise ValueError(
        "planar and joint avoidance prediction must be enabled together"
      )
    if avoidance_planar_dim < 0 or avoidance_joint_dim < 0:
      raise ValueError("avoidance prediction dimensions must be non-negative")
    if avoidance_planar_dim and (
      avoidance_planar_output_scale <= 0.0
      or avoidance_joint_output_scale <= 0.0
    ):
      raise ValueError("avoidance prediction output scales must be positive")
    if avoidance_joint_action_residual_gain < 0.0:
      raise ValueError("joint action residual gain must be non-negative")
    if avoidance_joint_action_residual_gain and output_dim != avoidance_joint_dim:
      raise ValueError(
        "direct joint residual requires one avoidance joint per policy action"
      )
    if avoidance_joint_action_residual_gain and (
      len(avoidance_joint_action_scales) != avoidance_joint_dim
      or len(avoidance_joint_action_mask) != avoidance_joint_dim
    ):
      raise ValueError(
        "direct joint residual requires one action scale and mask per joint"
      )
    if any(scale <= 0.0 for scale in avoidance_joint_action_scales):
      raise ValueError("joint action scales must be positive")
    self.avoidance_planar_dim = avoidance_planar_dim
    self.avoidance_joint_dim = avoidance_joint_dim
    self.avoidance_prediction_dim = avoidance_planar_dim + avoidance_joint_dim
    self.avoidance_joint_action_residual_gain = (
      avoidance_joint_action_residual_gain
    )
    if adjust_command_with_joint_prediction:
      if avoidance_joint_dim <= 0:
        raise ValueError(
          "command adjustment requires joint avoidance prediction"
        )
      if command_joint_pos_offset < 0 or (
        command_joint_pos_offset + avoidance_joint_dim > self.tracking_obs_dim
      ):
        raise ValueError(
          "command_joint_pos_offset must place the joint command slice "
          "inside the tracking observation"
        )
    self.adjust_command_with_joint_prediction = (
      adjust_command_with_joint_prediction
    )
    self.command_joint_pos_offset = command_joint_pos_offset
    normalize_tracking = obs_normalization
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      # LiDAR ranges are already normalized by the observation adapter. A
      # second empirical normalizer over directional values performs expensive
      # per-step reductions and makes misses drift away from their fixed value.
      # Keep empirical normalization only for the 154 tracking values.
      obs_normalization=False,
      distribution_cfg=distribution_cfg,
    )
    self.obs_normalization = normalize_tracking
    self.obs_normalizer = (
      EmpiricalNormalization(self.tracking_obs_dim)
      if normalize_tracking
      else torch.nn.Identity()
    )
    self.tracking_encoder = MLP(
      self.tracking_obs_dim,
      tracking_latent_dim,
      tracking_hidden_dims,
      activation,
    )
    # Keep the legacy CNN arguments in the public configuration so old command
    # lines remain parseable. Dense closest-return pooling now happens in the
    # observation adapter so PPO never copies raw scans into rollout storage.
    del lidar_channels, lidar_kernel_sizes, lidar_strides, lidar_direction_channels
    self.lidar_encoder = DirectionalFeatureEncoder(
      scan_count=scan_count,
      elevation_samples=elevation_samples,
      azimuth_samples=azimuth_samples,
      elevation_bins=lidar_elevation_bins,
      direction_bins=lidar_direction_bins,
      output_dim=lidar_latent_dim,
      activation=activation,
    )
    if self.avoidance_prediction_dim:
      self.avoidance_head: torch.nn.Module | None = MLP(
        self._base_latent_dim(),
        self.avoidance_prediction_dim,
        avoidance_hidden_dims,
        activation,
      )
      output_scale = torch.cat(
        (
          torch.full(
            (avoidance_planar_dim,), avoidance_planar_output_scale
          ),
          torch.full(
            (avoidance_joint_dim,), avoidance_joint_output_scale
          ),
        )
      )
    else:
      self.avoidance_head = None
      output_scale = torch.empty(0)
    self.register_buffer(
      "avoidance_output_scale",
      output_scale,
      persistent=bool(self.avoidance_prediction_dim),
    )
    action_scales = torch.as_tensor(
      avoidance_joint_action_scales, dtype=torch.float32
    )
    action_mask = torch.as_tensor(
      avoidance_joint_action_mask, dtype=torch.float32
    )
    # These are fixed interface constants rather than learned state. Keeping
    # them non-persistent allows checkpoints trained before the explicit
    # residual path to load unchanged.
    self.register_buffer(
      "avoidance_joint_action_scales", action_scales, persistent=False
    )
    self.register_buffer(
      "avoidance_joint_action_mask", action_mask, persistent=False
    )

  def _get_latent_dim(self) -> int:
    if self.adjust_command_with_joint_prediction:
      # The joint correction is visible inside the adjusted command, so it is
      # not also concatenated; the planar part has no command channel and
      # stays concatenated as the escape compass.
      return self._base_latent_dim() + self.avoidance_planar_dim
    return self._base_latent_dim() + self.avoidance_prediction_dim

  def _base_latent_dim(self) -> int:
    return self.tracking_latent_dim + self.lidar_latent_dim + self.timing_dim

  def _normalized_groups(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    return self.obs_normalizer(obs[self.tracking_obs_group]), obs[self.lidar_obs_group]

  def update_normalization(self, obs: TensorDict) -> None:
    """Update statistics only for tracking state, not normalized ranges.

    Statistics always come from the RAW tracking observation, also in
    command-adjustment mode: one convention shared by the adjuster pass and
    the (adjusted) mimic pass, whose corrections are bounded residuals.
    """
    if self.obs_normalization:
      self.obs_normalizer.update(obs[self.tracking_obs_group])

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    del masks, hidden_state
    base_latent = self.encode_deployable_observations(obs)
    if self.avoidance_head is None:
      return base_latent
    prediction = self._predict_avoidance_from_latent(base_latent)
    if self.adjust_command_with_joint_prediction:
      return self._command_adjusted_latent(obs, prediction)
    return torch.cat((base_latent, prediction), dim=-1)

  def encode_deployable_observations(self, obs: TensorDict) -> torch.Tensor:
    """Fuse only observations that are available on the physical robot."""
    tracking, lidar = self._normalized_groups(obs)
    directional = lidar[..., : self.directional_value_dim]
    timing = lidar[..., self.directional_value_dim :]
    return torch.cat(
      (self.tracking_encoder(tracking), self.lidar_encoder(directional), timing),
      dim=-1,
    )

  def _predict_avoidance_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
    if self.avoidance_head is None:
      raise RuntimeError("avoidance prediction is not enabled for this actor")
    return torch.tanh(self.avoidance_head(latent)) * self.avoidance_output_scale

  def predict_avoidance(self, obs: TensorDict) -> torch.Tensor:
    """Predict body-frame planar and joint corrections for deployment."""
    return self._predict_avoidance_from_latent(
      self.encode_deployable_observations(obs)
    )

  def _command_adjusted_latent(
    self,
    obs: TensorDict,
    avoidance: torch.Tensor,
  ) -> torch.Tensor:
    """Encode observations after adding a joint correction to the command.

    The joint part of ``avoidance`` (radians) is added to the joint-position
    command targets inside the raw tracking observation; joint-velocity
    targets and every other value stay untouched, and empirical normalization
    applies to the adjusted vector. The planar part has no command channel
    (the pelvis path is deliberately unchanged), so it stays concatenated to
    the policy input as the escape compass. The adjuster conditions on the
    nominal reference, so the tracking encoder deliberately runs twice per
    step: once on the raw command and once here on the adjusted one. The
    functional concatenation keeps the stored rollout observation intact and
    exports cleanly to ONNX.
    """
    tracking = obs[self.tracking_obs_group]
    lidar = obs[self.lidar_obs_group]
    expected_shape = (*tracking.shape[:-1], self.avoidance_prediction_dim)
    if avoidance.shape != expected_shape:
      raise ValueError(
        f"avoidance correction has shape {tuple(avoidance.shape)}, expected "
        f"{expected_shape}"
      )
    joint_correction = avoidance[..., self.avoidance_planar_dim :]
    start = self.command_joint_pos_offset
    end = start + self.avoidance_joint_dim
    adjusted_tracking = torch.cat(
      (
        tracking[..., :start],
        tracking[..., start:end] + joint_correction,
        tracking[..., end:],
      ),
      dim=-1,
    )
    adjusted_tracking = self.obs_normalizer(adjusted_tracking)
    directional = lidar[..., : self.directional_value_dim]
    timing = lidar[..., self.directional_value_dim :]
    return torch.cat(
      (
        self.tracking_encoder(adjusted_tracking),
        self.lidar_encoder(directional),
        timing,
        avoidance[..., : self.avoidance_planar_dim],
      ),
      dim=-1,
    )

  def _forward_with_command_adjustment(
    self,
    obs: TensorDict,
    avoidance: torch.Tensor,
    *,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    """Run the mimic head on observations whose command carries the correction."""
    mlp_output = self.mlp(self._command_adjusted_latent(obs, avoidance))
    if self.distribution is None:
      return mlp_output
    if stochastic_output:
      self.distribution.update(mlp_output)
      return self.distribution.sample()
    return self.distribution.deterministic_output(mlp_output)

  def action_with_avoidance_override(
    self,
    obs: TensorDict,
    avoidance: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluate the policy after replacing only its correction prediction."""
    if self.avoidance_head is None:
      raise RuntimeError("avoidance prediction is not enabled for this actor")
    if self.adjust_command_with_joint_prediction:
      return self._forward_with_command_adjustment(obs, avoidance)
    base_latent = self.encode_deployable_observations(obs)
    return self._forward_avoidance_latent(base_latent, avoidance)

  def _forward_avoidance_latent(
    self,
    base_latent: torch.Tensor,
    avoidance: torch.Tensor,
    *,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    """Run the action head from an already encoded deployable observation."""
    expected_shape = (*base_latent.shape[:-1], self.avoidance_prediction_dim)
    if avoidance.shape != expected_shape:
      raise ValueError(
        f"avoidance override has shape {tuple(avoidance.shape)}, expected "
        f"{expected_shape}"
      )
    mlp_output = self.mlp(torch.cat((base_latent, avoidance), dim=-1))
    mlp_output = mlp_output + self._joint_action_residual(avoidance)
    if self.distribution is None:
      return mlp_output
    if stochastic_output:
      self.distribution.update(mlp_output)
      return self.distribution.sample()
    return self.distribution.deterministic_output(mlp_output)

  def _joint_action_residual(self, avoidance: torch.Tensor) -> torch.Tensor:
    """Convert predicted safe-reference radians into normalized G1 actions."""
    # In command-adjustment mode the correction already enters through the
    # observed reference; the co-adjust task also configures gain zero, but
    # gate defensively so a correction can never be applied twice.
    if (
      self.adjust_command_with_joint_prediction
      or self.avoidance_joint_action_residual_gain == 0.0
    ):
      return torch.zeros_like(avoidance[..., : self.avoidance_joint_dim])
    joint_residual_rad = avoidance[..., self.avoidance_planar_dim :]
    return (
      self.avoidance_joint_action_residual_gain
      * self.avoidance_joint_action_mask
      * joint_residual_rad
      / self.avoidance_joint_action_scales
    )

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    """Run the deployable actor including its explicit joint residual path."""
    del masks, hidden_state
    base_latent = self.encode_deployable_observations(obs)
    if self.avoidance_head is None:
      mlp_output = self.mlp(base_latent)
      if self.distribution is None:
        return mlp_output
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    avoidance = self._predict_avoidance_from_latent(base_latent)
    if self.adjust_command_with_joint_prediction:
      return self._forward_with_command_adjustment(
        obs,
        avoidance,
        stochastic_output=stochastic_output,
      )
    return self._forward_avoidance_latent(
      base_latent,
      avoidance,
      stochastic_output=stochastic_output,
    )

  def as_jit(self) -> torch.nn.Module:
    return _ExportPerceptiveLidarActor(self, verbose=False)

  def as_onnx(self, verbose: bool) -> torch.nn.Module:
    return _ExportPerceptiveLidarActor(self, verbose=verbose)


class _ExportPerceptiveLidarActor(torch.nn.Module):
  """Flat-input deterministic deployment wrapper."""

  def __init__(self, actor: PerceptiveLidarActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.input_size = actor.obs_dim
    self.tracking_obs_dim = actor.tracking_obs_dim
    self.directional_value_dim = actor.directional_value_dim
    self.obs_normalizer = deepcopy(actor.obs_normalizer)
    self.tracking_encoder = deepcopy(actor.tracking_encoder)
    self.lidar_encoder = deepcopy(actor.lidar_encoder)
    self.avoidance_head = deepcopy(actor.avoidance_head)
    self.register_buffer(
      "avoidance_output_scale", actor.avoidance_output_scale.detach().clone()
    )
    self.avoidance_planar_dim = actor.avoidance_planar_dim
    self.avoidance_joint_dim = actor.avoidance_joint_dim
    self.adjust_command_with_joint_prediction = (
      actor.adjust_command_with_joint_prediction
    )
    self.command_joint_pos_offset = actor.command_joint_pos_offset
    self.avoidance_joint_action_residual_gain = (
      actor.avoidance_joint_action_residual_gain
    )
    self.register_buffer(
      "avoidance_joint_action_scales",
      actor.avoidance_joint_action_scales.detach().clone(),
    )
    self.register_buffer(
      "avoidance_joint_action_mask",
      actor.avoidance_joint_action_mask.detach().clone(),
    )
    self.mlp = deepcopy(actor.mlp)
    if actor.distribution is None:
      self.deterministic_output = torch.nn.Identity()
    else:
      self.deterministic_output = actor.distribution.as_deterministic_output_module()

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    tracking_raw = observation[..., : self.tracking_obs_dim]
    tracking = self.obs_normalizer(tracking_raw)
    lidar = observation[..., self.tracking_obs_dim :]
    directional = lidar[..., : self.directional_value_dim]
    timing = lidar[..., self.directional_value_dim :]
    lidar_latent = self.lidar_encoder(directional)
    base_latent = torch.cat(
      (self.tracking_encoder(tracking), lidar_latent, timing),
      dim=-1,
    )
    if self.avoidance_head is None:
      latent = base_latent
    else:
      prediction = (
        torch.tanh(self.avoidance_head(base_latent))
        * self.avoidance_output_scale
      )
      if self.adjust_command_with_joint_prediction:
        joint_correction = prediction[..., self.avoidance_planar_dim :]
        start = self.command_joint_pos_offset
        end = start + self.avoidance_joint_dim
        adjusted_tracking = self.obs_normalizer(
          torch.cat(
            (
              tracking_raw[..., :start],
              tracking_raw[..., start:end] + joint_correction,
              tracking_raw[..., end:],
            ),
            dim=-1,
          )
        )
        adjusted_latent = torch.cat(
          (
            self.tracking_encoder(adjusted_tracking),
            lidar_latent,
            timing,
            prediction[..., : self.avoidance_planar_dim],
          ),
          dim=-1,
        )
        return self.deterministic_output(self.mlp(adjusted_latent))
      latent = torch.cat((base_latent, prediction), dim=-1)
    mlp_output = self.mlp(latent)
    if self.avoidance_head is not None and (
      self.avoidance_joint_action_residual_gain != 0.0
    ):
      joint_residual_rad = prediction[..., self.avoidance_planar_dim :]
      mlp_output = mlp_output + (
        self.avoidance_joint_action_residual_gain
        * self.avoidance_joint_action_mask
        * joint_residual_rad
        / self.avoidance_joint_action_scales
      )
    return self.deterministic_output(mlp_output)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


def _avoidance_supervision_losses(
  prediction: torch.Tensor,
  target: torch.Tensor,
  *,
  planar_dim: int,
  active_sample_weight: float,
  active_joint_weight: float,
  active_threshold: float,
  active_joint_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
  """Return balanced planar/joint losses and interpretable diagnostics."""
  planar_prediction = prediction[..., :planar_dim]
  planar_target = target[..., :planar_dim]
  joint_prediction = prediction[..., planar_dim:]
  joint_target = target[..., planar_dim:]
  planar_active = (
    torch.linalg.vector_norm(planar_target, dim=-1) >= active_threshold
  )
  joint_active = torch.abs(joint_target) >= active_joint_threshold
  sample_active = planar_active | torch.any(joint_active, dim=-1)
  sample_weights = 1.0 + sample_active.float() * (active_sample_weight - 1.0)

  planar_per_sample = torch.nn.functional.smooth_l1_loss(
    planar_prediction, planar_target, reduction="none"
  ).mean(dim=-1)
  planar_loss = (planar_per_sample * sample_weights).sum() / (
    sample_weights.sum().clamp_min(1.0)
  )

  joint_error = torch.nn.functional.smooth_l1_loss(
    joint_prediction, joint_target, reduction="none"
  )
  joint_weights = sample_weights[:, None] * (
    1.0 + joint_active.float() * (active_joint_weight - 1.0)
  )
  joint_loss = (joint_error * joint_weights).sum() / (
    joint_weights.sum().clamp_min(1.0)
  )

  joint_active_count = joint_active.count_nonzero().clamp_min(1)
  joint_active_mae = (
    torch.abs(joint_prediction - joint_target) * joint_active.float()
  ).sum() / joint_active_count
  planar_cosine = torch.nn.functional.cosine_similarity(
    planar_prediction,
    planar_target,
    dim=-1,
    eps=1.0e-6,
  )
  planar_active_count = planar_active.count_nonzero().clamp_min(1)
  planar_active_cosine = (
    planar_cosine * planar_active.float()
  ).sum() / planar_active_count
  diagnostics = {
    "active_fraction": sample_active.float().mean(),
    "joint_active_fraction": joint_active.float().mean(),
    "joint_active_mae": joint_active_mae,
    "planar_active_cosine": planar_active_cosine,
  }
  return planar_loss, joint_loss, diagnostics


class AvoidanceAuxiliaryPPO(PPO):
  """PPO with correction supervision in the same optimizer step."""

  actor: PerceptiveLidarActor

  def __init__(
    self,
    *args,
    avoidance_target_group: str = "avoidance_teacher",
    avoidance_noise_group: str = "avoidance_robustness",
    avoidance_num_learning_epochs: int = 1,
    avoidance_num_mini_batches: int = 4,
    avoidance_planar_loss_coef: float = 1.0,
    avoidance_joint_loss_coef: float = 1.0,
    avoidance_active_sample_weight: float = 4.0,
    avoidance_active_joint_weight: float = 12.0,
    avoidance_active_threshold: float = 1.0e-3,
    avoidance_active_joint_threshold: float = 1.0e-3,
    avoidance_teacher_mix_start: float = 0.1,
    avoidance_teacher_mix_end: float = 0.0,
    avoidance_teacher_mix_decay_updates: int = 5000,
    avoidance_conditioning_noise_scale: float = 0.03,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    actor = self._raw_actor
    if not isinstance(actor, PerceptiveLidarActor):
      raise TypeError("AvoidanceAuxiliaryPPO requires PerceptiveLidarActor")
    if actor.avoidance_prediction_dim == 0:
      raise ValueError("avoidance prediction must be enabled on the actor")
    if self.rnd is not None or self.symmetry is not None:
      raise ValueError("avoidance auxiliary PPO does not support RND or symmetry")
    if actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("avoidance auxiliary PPO requires feed-forward models")
    if not 1 <= avoidance_num_learning_epochs <= self.num_learning_epochs:
      raise ValueError("auxiliary epochs must fit within PPO learning epochs")
    if avoidance_num_mini_batches != self.num_mini_batches:
      raise ValueError("auxiliary and PPO mini-batch counts must match")
    if avoidance_planar_loss_coef < 0.0 or avoidance_joint_loss_coef < 0.0:
      raise ValueError("avoidance loss coefficients must be non-negative")
    if avoidance_active_sample_weight < 1.0:
      raise ValueError("active avoidance samples cannot be down-weighted")
    if avoidance_active_joint_weight < 1.0:
      raise ValueError("active joints cannot be down-weighted")
    if avoidance_active_threshold < 0.0 or avoidance_active_joint_threshold < 0.0:
      raise ValueError("avoidance thresholds must be non-negative")
    if not 0.0 <= avoidance_teacher_mix_start <= 1.0:
      raise ValueError("teacher mix start must be in [0, 1]")
    if not 0.0 <= avoidance_teacher_mix_end <= 1.0:
      raise ValueError("teacher mix end must be in [0, 1]")
    if avoidance_teacher_mix_decay_updates < 1:
      raise ValueError("teacher mix decay must be positive")
    if avoidance_conditioning_noise_scale < 0.0:
      raise ValueError("conditioning noise scale must be non-negative")

    self.avoidance_target_group = avoidance_target_group
    self.avoidance_noise_group = avoidance_noise_group
    self.avoidance_num_learning_epochs = avoidance_num_learning_epochs
    self.avoidance_num_mini_batches = avoidance_num_mini_batches
    self.avoidance_planar_loss_coef = avoidance_planar_loss_coef
    self.avoidance_joint_loss_coef = avoidance_joint_loss_coef
    self.avoidance_active_sample_weight = avoidance_active_sample_weight
    self.avoidance_active_joint_weight = avoidance_active_joint_weight
    self.avoidance_active_threshold = avoidance_active_threshold
    self.avoidance_active_joint_threshold = avoidance_active_joint_threshold
    self.avoidance_teacher_mix_start = avoidance_teacher_mix_start
    self.avoidance_teacher_mix_end = avoidance_teacher_mix_end
    self.avoidance_teacher_mix_decay_updates = (
      avoidance_teacher_mix_decay_updates
    )
    self.avoidance_conditioning_noise_scale = (
      avoidance_conditioning_noise_scale
    )
    self._avoidance_updates_completed = 0

  @property
  def avoidance_teacher_mix(self) -> float:
    progress = min(
      self._avoidance_updates_completed
      / self.avoidance_teacher_mix_decay_updates,
      1.0,
    )
    return self.avoidance_teacher_mix_start + progress * (
      self.avoidance_teacher_mix_end - self.avoidance_teacher_mix_start
    )

  def _prediction_and_conditioning(
    self,
    observations: TensorDict,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actor = self._raw_actor
    base_latent = actor.encode_deployable_observations(observations)
    prediction = actor._predict_avoidance_from_latent(base_latent)  # noqa: SLF001
    target = observations[self.avoidance_target_group]
    noise = observations[self.avoidance_noise_group]
    if target.shape != prediction.shape or noise.shape != prediction.shape:
      raise RuntimeError(
        "avoidance teacher/noise shape must match the correction prediction"
      )
    conditioning = prediction + self.avoidance_teacher_mix * (
      target - prediction
    )
    conditioning = conditioning + (
      self.avoidance_conditioning_noise_scale
      * actor.avoidance_output_scale
      * noise
    )
    limit = actor.avoidance_output_scale
    conditioning = torch.maximum(torch.minimum(conditioning, limit), -limit)
    return base_latent, prediction, conditioning

  def _rollout_action_distribution(
    self,
    observations: TensorDict,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the rollout-time action forward with teacher-mixed conditioning.

    Shared by ``act`` and ``update`` so PPO's importance ratio is exact by
    construction. Returns the sampled actions and the raw correction
    prediction used by the supervised loss.
    """
    actor = self._raw_actor
    base_latent, prediction, conditioning = self._prediction_and_conditioning(
      observations
    )
    if actor.adjust_command_with_joint_prediction:
      actions = actor._forward_with_command_adjustment(  # noqa: SLF001
        observations,
        conditioning,
        stochastic_output=True,
      )
    else:
      actions = actor._forward_avoidance_latent(  # noqa: SLF001
        base_latent,
        conditioning,
        stochastic_output=True,
      )
    return actions, prediction

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample with stored teacher interpolation/noise during training only."""
    actor = self._raw_actor
    self.transition.hidden_states = (
      actor.get_hidden_state(),
      self.critic.get_hidden_state(),
    )
    actions, _ = self._rollout_action_distribution(obs)
    self.transition.actions = actions.detach()
    self.transition.values = self.critic(obs).detach()
    self.transition.actions_log_prob = actor.get_output_log_prob(
      self.transition.actions
    ).detach()
    self.transition.distribution_params = tuple(
      parameter.detach() for parameter in actor.output_distribution_params
    )
    self.transition.observations = obs
    return self.transition.actions

  def update(self) -> dict[str, float]:
    """Run upstream PPO and auxiliary gradients in shared optimizer steps."""
    actor = self._raw_actor
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches,
      self.num_learning_epochs,
    )
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    auxiliary_sums = {
      "avoidance_planar": 0.0,
      "avoidance_joint": 0.0,
      "avoidance_active_fraction": 0.0,
      "avoidance_joint_active_fraction": 0.0,
      "avoidance_joint_active_mae": 0.0,
      "avoidance_planar_active_cosine": 0.0,
    }
    auxiliary_batches = 0

    for batch_index, batch in enumerate(generator):
      observations = batch.observations
      if observations is None:
        raise RuntimeError("PPO batch is missing observations")
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1.0e-8)

      _, prediction = self._rollout_action_distribution(observations)
      actions_log_prob = actor.get_output_log_prob(batch.actions)
      values = self.critic(observations)
      distribution_params = actor.output_distribution_params
      entropy = actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = actor.get_kl_divergence(
            batch.old_distribution_params,
            distribution_params,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(
              kl_mean,
              op=torch.distributed.ReduceOp.SUM,
            )
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio,
        1.0 - self.clip_param,
        1.0 + self.clip_param,
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param,
          self.clip_param,
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()
      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      epoch_index = batch_index // self.num_mini_batches
      if epoch_index < self.avoidance_num_learning_epochs:
        target = observations[self.avoidance_target_group]
        planar_loss, joint_loss, diagnostics = (
          _avoidance_supervision_losses(
            prediction,
            target,
            planar_dim=actor.avoidance_planar_dim,
            active_sample_weight=self.avoidance_active_sample_weight,
            active_joint_weight=self.avoidance_active_joint_weight,
            active_threshold=self.avoidance_active_threshold,
            active_joint_threshold=self.avoidance_active_joint_threshold,
          )
        )
        loss = loss + self.avoidance_planar_loss_coef * planar_loss
        loss = loss + self.avoidance_joint_loss_coef * joint_loss
        auxiliary_sums["avoidance_planar"] += planar_loss.item()
        auxiliary_sums["avoidance_joint"] += joint_loss.item()
        auxiliary_sums["avoidance_active_fraction"] += diagnostics[
          "active_fraction"
        ].item()
        auxiliary_sums["avoidance_joint_active_fraction"] += diagnostics[
          "joint_active_fraction"
        ].item()
        auxiliary_sums["avoidance_joint_active_mae"] += diagnostics[
          "joint_active_mae"
        ].item()
        auxiliary_sums["avoidance_planar_active_cosine"] += diagnostics[
          "planar_active_cosine"
        ].item()
        auxiliary_batches += 1

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      **{
        name: value / max(auxiliary_batches, 1)
        for name, value in auxiliary_sums.items()
      },
      "avoidance_teacher_mix": self.avoidance_teacher_mix,
    }
    self._avoidance_updates_completed += 1
    self.storage.clear()
    return loss_dict

  def save(self) -> dict:
    saved = super().save()
    saved["avoidance_updates_completed"] = self._avoidance_updates_completed
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    loaded_iteration = super().load(loaded_dict, load_cfg, strict)
    self._avoidance_updates_completed = int(
      loaded_dict.get("avoidance_updates_completed", 0)
    )
    return loaded_iteration
