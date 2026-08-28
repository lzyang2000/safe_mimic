"""Multi-motion command backed by the packed NPZ library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from safe_mimic.motions.adaptive_motion_sampling import (
  AdaptiveMotionSample,
  AdaptiveMotionSampler,
  AdaptiveMotionSamplingCfg,
)
from safe_mimic.motions.packed_npz_motion_lib import (
  PackedNpzMotionLib,
)

if TYPE_CHECKING:
  from collections.abc import Callable

  import viser
  from mjlab.envs import ManagerBasedRlEnv


class PackedMotionCommand(MotionCommand):
  """Track a different packed-library motion and phase in every environment."""

  cfg: PackedMotionCommandCfg

  def __init__(self, cfg: PackedMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
    # MotionCommand.__init__ hardcodes the single-NPZ MotionLoader, so initialize
    # its CommandTerm base and reproduce only the storage-independent setup.
    CommandTerm.__init__(self, cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    body_indexes, body_names = self.robot.find_bodies(
      cfg.body_names, preserve_order=True
    )
    if tuple(body_names) != cfg.body_names:
      raise ValueError(
        f"resolved tracking bodies {tuple(body_names)} do not match {cfg.body_names}"
      )
    self.body_indexes = torch.tensor(
      body_indexes, dtype=torch.long, device=self.device
    )

    self.motion = PackedNpzMotionLib(
      cfg.motion_file,
      self.body_indexes,
      device=self.device,
      splits=cfg.manifest_splits,
    )
    sampler_cfg = AdaptiveMotionSamplingCfg(
      bin_duration_s=cfg.adaptive_bin_duration_s,
      failure_ema_alpha=cfg.adaptive_alpha,
      uniform_ratio=cfg.adaptive_uniform_ratio,
      temporal_kernel_size=cfg.adaptive_kernel_size,
      temporal_kernel_lambda=cfg.adaptive_lambda,
      group_key=cfg.adaptive_group_key,
      pair_key=cfg.adaptive_pair_key,
      couple_pairs=cfg.adaptive_couple_pairs,
    )
    self.adaptive_sampler = AdaptiveMotionSampler(self.motion, sampler_cfg)
    self.motion_ids = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.motion_times = torch.zeros(self.num_envs, device=self.device)
    self._has_reference = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._play_cursor = torch.arange(
      self.num_envs, dtype=torch.long, device=self.device
    ) % self.motion.num_motions()
    self._frame = self.motion.get_frame(self.motion_ids, self.motion_times)

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[..., 0] = 1.0

    metric_names = (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_entropy",
      "sampling_top1_prob",
      "sampling_observed_bin_fraction",
      "sampling_mean_failure_ema",
    )
    for name in metric_names:
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False
    self._update_sampling_metrics()

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._frame.joint_pos

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._frame.joint_vel

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._frame.body_pos_w + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._frame.body_quat_w

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._frame.body_lin_vel_w

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._frame.body_ang_vel_w

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  def _refresh_frame(self) -> None:
    self._frame = self.motion.get_frame(self.motion_ids, self.motion_times)

  def _record_finished_references(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode != "adaptive":
      return
    valid_ids = env_ids[self._has_reference[env_ids]]
    if valid_ids.numel() == 0:
      return
    failures = self._env.termination_manager.terminated[valid_ids]
    self.adaptive_sampler.update(
      failures,
      self.motion_ids[valid_ids],
      self.motion_times[valid_ids],
    )

  def _sample_references(self, env_ids: torch.Tensor) -> AdaptiveMotionSample:
    count = len(env_ids)
    if self.cfg.sampling_mode == "adaptive":
      return self.adaptive_sampler.sample(count)
    if self.cfg.sampling_mode == "uniform":
      return self.adaptive_sampler.sample_prior(count)
    if self.cfg.sampling_mode != "start":
      raise ValueError(f"unsupported sampling mode {self.cfg.sampling_mode!r}")
    motion_ids = self._play_cursor[env_ids]
    self._play_cursor[env_ids] = (
      self._play_cursor[env_ids] + 1
    ) % self.motion.num_motions()
    return AdaptiveMotionSample(
      motion_ids=motion_ids,
      motion_times=torch.zeros(count, device=self.device),
      bin_ids=torch.zeros(count, dtype=torch.long, device=self.device),
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._record_finished_references(env_ids)
    sample = self._sample_references(env_ids)
    self.motion_ids[env_ids] = sample.motion_ids
    self.motion_times[env_ids] = sample.motion_times
    self._has_reference[env_ids] = True
    self._refresh_frame()

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    pose_ranges = torch.tensor(
      [
        self.cfg.pose_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    pose_delta = sample_uniform(
      pose_ranges[:, 0],
      pose_ranges[:, 1],
      (len(env_ids), 6),
      device=self.device,
    )
    root_pos += pose_delta[:, :3]
    root_ori = quat_mul(
      quat_from_euler_xyz(
        pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]
      ),
      root_ori,
    )

    velocity_ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    velocity_delta = sample_uniform(
      velocity_ranges[:, 0],
      velocity_ranges[:, 1],
      (len(env_ids), 6),
      device=self.device,
    )
    root_lin_vel += velocity_delta[:, :3]
    root_ang_vel += velocity_delta[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,
    )
    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True
    self._update_sampling_metrics()

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    advance_ids = (
      torch.arange(self.num_envs, device=self.device)
      if env_ids is None
      else env_ids
    )
    self.motion_times[advance_ids] += self._env.step_dt
    lengths = self.motion.get_motion_length(self.motion_ids[advance_ids])
    wrap_ids = advance_ids[self.motion_times[advance_ids] >= lengths]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)
    else:
      self._refresh_frame()

    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()

  def _update_sampling_metrics(self) -> None:
    probabilities = self.adaptive_sampler.probabilities
    entropy = -(probabilities * probabilities.clamp(min=1e-12).log()).sum()
    if self.adaptive_sampler.num_bins > 1:
      entropy /= torch.log(
        torch.tensor(float(self.adaptive_sampler.num_bins), device=self.device)
      )
    else:
      entropy.fill_(1.0)
    self.metrics["sampling_entropy"][:] = entropy
    self.metrics["sampling_top1_prob"][:] = probabilities.max()
    self.metrics["sampling_observed_bin_fraction"][:] = (
      self.adaptive_sampler.episode_counts > 0
    ).float().mean()
    self.metrics["sampling_mean_failure_ema"][:] = (
      self.adaptive_sampler.failure_ema.mean()
    )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """The single-motion frame scrubber does not apply to a motion library."""
    del name, server, get_env_idx, on_change, request_action

  def adaptive_state_dict(self) -> dict[str, torch.Tensor]:
    """Return difficulty statistics for inclusion in a training checkpoint."""
    return self.adaptive_sampler.state_dict()

  def load_adaptive_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    """Restore difficulty statistics from a training checkpoint."""
    self.adaptive_sampler.load_state_dict(state)
    self._update_sampling_metrics()


@dataclass(kw_only=True)
class PackedMotionCommandCfg(MotionCommandCfg):
  """Configuration for :class:`PackedMotionCommand`."""

  manifest_splits: tuple[str, ...] | None = ("train",)
  adaptive_bin_duration_s: float = 1.0
  adaptive_group_key: str = "sampling_pool"
  adaptive_pair_key: str = "pair_id"
  adaptive_couple_pairs: bool = True

  def build(self, env: ManagerBasedRlEnv) -> PackedMotionCommand:
    return PackedMotionCommand(self, env)


__all__ = ["PackedMotionCommand", "PackedMotionCommandCfg"]
