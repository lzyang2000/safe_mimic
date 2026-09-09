"""CUDA-resident skeleton bank and online walk/action/walk composition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from safe_mimic.motions.capsule_path_bank import OnlineHumanPoses
from safe_mimic.motions.human_capsules import SOMA_CAPSULE_SPECS, CapsuleSpec


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
  lw, lx, ly, lz = left.unbind(dim=-1)
  rw, rx, ry, rz = right.unbind(dim=-1)
  return torch.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    dim=-1,
  )


def _quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
  result = quaternion.clone()
  result[..., 1:] *= -1.0
  return result / (quaternion * quaternion).sum(dim=-1, keepdim=True).clamp_min(
    1e-12
  )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  xyz = quaternion[..., 1:]
  uv = torch.cross(xyz, vector, dim=-1)
  uuv = torch.cross(xyz, uv, dim=-1)
  return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def _quat_nlerp(
  first: torch.Tensor, second: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  sign = torch.where(
    (first * second).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0
  )
  result = first * (1.0 - blend) + second * sign * blend
  return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(
    1e-12
  )


def _quat_log(quaternion: torch.Tensor) -> torch.Tensor:
  quaternion = quaternion / torch.linalg.vector_norm(
    quaternion, dim=-1, keepdim=True
  ).clamp_min(1e-12)
  quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
  xyz = quaternion[..., 1:]
  norm = torch.linalg.vector_norm(xyz, dim=-1)
  angle = 2.0 * torch.atan2(norm, quaternion[..., 0].clamp_min(1e-12))
  scale = torch.where(norm > 1e-8, angle / norm, torch.full_like(norm, 2.0))
  return xyz * scale[..., None]


def _quat_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
  angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
  half = 0.5 * angle
  scale = torch.where(
    angle > 1e-8,
    torch.sin(half) / angle,
    torch.full_like(angle, 0.5),
  )
  return torch.cat(
    (torch.cos(half)[..., None], rotation_vector * scale[..., None]), dim=-1
  )


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
  return torch.stack(
    (
      torch.cos(0.5 * yaw),
      torch.zeros_like(yaw),
      torch.zeros_like(yaw),
      torch.sin(0.5 * yaw),
    ),
    dim=-1,
  )


def _rotate_xy(points: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  cosine = torch.cos(yaw)
  sine = torch.sin(yaw)
  while cosine.ndim < points.ndim - 1:
    cosine = cosine.unsqueeze(-1)
    sine = sine.unsqueeze(-1)
  return torch.stack(
    (
      cosine * points[..., 0] - sine * points[..., 1],
      sine * points[..., 0] + cosine * points[..., 1],
      points[..., 2],
    ),
    dim=-1,
  )


def _quat_from_z(vectors: torch.Tensor) -> torch.Tensor:
  length = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
  unit = vectors / length.clamp_min(1e-12)
  dot = unit[..., 2].clamp(-1.0, 1.0)
  quaternion = torch.stack(
    (1.0 + dot, -unit[..., 1], unit[..., 0], torch.zeros_like(dot)), dim=-1
  )
  opposite = dot < -0.999999
  fallback = torch.zeros_like(quaternion)
  fallback[..., 1] = 1.0
  quaternion = torch.where(opposite[..., None], fallback, quaternion)
  return quaternion / torch.linalg.vector_norm(
    quaternion, dim=-1, keepdim=True
  ).clamp_min(1e-12)


class SkeletonPathBank:
  """Memory-mapped compact bank containing only transition-approved segments."""

  def __init__(self, path: Path | str) -> None:
    self.path = Path(path)
    with (self.path / "bank.json").open() as source:
      self.config = json.load(source)
    if self.config.get("format_version") != 1 or not self.config.get("complete"):
      raise ValueError(f"{self.path} is not a complete skeleton path bank")
    self.fps = float(self.config["fps"])
    self.max_frames = int(self.config["max_frames"])
    self.joint_names = tuple(self.config["joint_names"])
    self.local_positions = np.load(
      self.path / "local_positions.npy", mmap_mode="r"
    )
    self.local_quaternions = np.load(
      self.path / "local_quaternions.npy", mmap_mode="r"
    )
    self.frame_counts = np.load(self.path / "frame_counts.npy", mmap_mode="r")
    self.ground_z = np.load(self.path / "ground_z.npy", mmap_mode="r")
    self.source_path_ids = np.load(
      self.path / "source_path_ids.npy", mmap_mode="r"
    )
    self.parents = np.load(self.path / "parents.npy")
    self._validate()

  def _validate(self) -> None:
    paths = int(self.config["path_count"])
    joints = len(self.joint_names)
    if self.local_positions.shape != (paths, self.max_frames, joints, 3):
      raise ValueError("invalid skeleton local-position shape")
    if self.local_quaternions.shape != (paths, self.max_frames, joints, 4):
      raise ValueError("invalid skeleton local-quaternion shape")
    if self.frame_counts.shape != (paths,) or self.source_path_ids.shape != (paths,):
      raise ValueError("invalid skeleton path metadata shape")
    if self.parents.shape != (joints,):
      raise ValueError("invalid skeleton parent shape")
    if len(np.unique(self.source_path_ids)) != paths:
      raise ValueError("skeleton source path ids must be unique")

  def __len__(self) -> int:
    return len(self.frame_counts)


class OnlineComposedHumanSampler:
  """Compose transition-matched triplets and stream capsule poses on-device."""

  def __init__(
    self,
    bank: SkeletonPathBank,
    num_envs: int,
    device: str | torch.device,
    *,
    transition_duration_s: float = 0.2,
    update_hz: float = 10.0,
    inactive_height_m: float = -100.0,
    retain_joint_poses: bool = False,
    lock_root_xy: bool = False,
    capsule_specs: tuple[CapsuleSpec, ...] = SOMA_CAPSULE_SPECS,
  ) -> None:
    if num_envs < 1 or update_hz <= 0.0 or transition_duration_s <= 0.0:
      raise ValueError("invalid online composed-human sampler configuration")
    self.bank = bank
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.transition_duration_s = transition_duration_s
    self.update_period_s = 1.0 / update_hz
    self.inactive_height_m = inactive_height_m
    self.retain_joint_poses = retain_joint_poses
    self.lock_root_xy = lock_root_xy
    self.capsule_specs = capsule_specs
    if not self.capsule_specs:
      raise ValueError("at least one capsule spec is required")

    def load(array: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
      tensor = torch.as_tensor(np.asarray(array).copy(), device=self.device)
      return tensor.to(dtype=dtype) if dtype is not None else tensor

    self.local_positions = load(bank.local_positions)
    self.local_quaternions = load(bank.local_quaternions)
    self.frame_counts = load(bank.frame_counts, dtype=torch.long)
    self.ground_z = load(bank.ground_z, dtype=torch.float32)
    self.source_path_ids = load(bank.source_path_ids, dtype=torch.long)
    self.parents = load(bank.parents, dtype=torch.long)
    lookup_size = int(bank.config["source_path_count"])
    self.source_to_row = torch.full(
      (lookup_size,), -1, dtype=torch.long, device=self.device
    )
    self.source_to_row[self.source_path_ids] = torch.arange(
      len(bank), device=self.device
    )
    resident = (
      self.local_positions,
      self.local_quaternions,
      self.frame_counts,
      self.ground_z,
      self.source_path_ids,
      self.parents,
      self.source_to_row,
    )
    self.device_storage_bytes = sum(
      tensor.numel() * tensor.element_size() for tensor in resident
    )

    joint_count = len(bank.joint_names)
    self.sequence_rows = torch.zeros(
      (num_envs, 3), dtype=torch.long, device=self.device
    )
    self.segment_durations_s = torch.zeros((num_envs, 3), device=self.device)
    self.segment_starts_s = torch.zeros((num_envs, 3), device=self.device)
    self.total_durations_s = torch.zeros(num_envs, device=self.device)
    self.alignment_yaw = torch.zeros((num_envs, 3), device=self.device)
    self.alignment_translation = torch.zeros((num_envs, 3, 3), device=self.device)
    transition_shape = (num_envs, 2, joint_count, 3)
    self.position_offsets = torch.zeros(transition_shape, device=self.device)
    self.position_velocity_offsets = torch.zeros(transition_shape, device=self.device)
    self.rotation_offsets = torch.zeros(transition_shape, device=self.device)
    self.rotation_velocity_offsets = torch.zeros(transition_shape, device=self.device)
    self.global_intersection_times_s = torch.zeros(num_envs, device=self.device)
    self.local_intersection_times_s = torch.zeros(num_envs, device=self.device)
    self.placement_yaw = torch.zeros(num_envs, device=self.device)
    self.translation_w = torch.zeros((num_envs, 3), device=self.device)
    self.current_translation_w = torch.zeros((num_envs, 3), device=self.device)
    self.root_anchor_w = torch.zeros((num_envs, 3), device=self.device)
    self.playback_speed = torch.ones(num_envs, device=self.device)
    self.body_scale_xyz = torch.ones((num_envs, 3), device=self.device)
    self.radius_scale = torch.ones(num_envs, device=self.device)
    self.radius_margin_m = torch.zeros(num_envs, device=self.device)
    self.next_update_times_s = torch.full(
      (num_envs,), -torch.inf, device=self.device
    )
    self.dirty = torch.ones(num_envs, dtype=torch.bool, device=self.device)
    # Deactivated (parked) humans keep their stale schedule fields; expiry
    # must never fire for them or they would be revived mid-episode.
    self.scheduled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    self.last_updated_env_ids = torch.empty(0, dtype=torch.long, device=self.device)

    capsule_count = len(self.capsule_specs)
    self._poses = OnlineHumanPoses(
      centers_w=torch.zeros((num_envs, capsule_count, 3), device=self.device),
      quaternions_wxyz=torch.zeros(
        (num_envs, capsule_count, 4), device=self.device
      ),
      radii_m=torch.zeros((num_envs, capsule_count), device=self.device),
      half_lengths_m=torch.zeros((num_envs, capsule_count), device=self.device),
      root_positions_w=torch.zeros((num_envs, 3), device=self.device),
      active=torch.zeros(num_envs, dtype=torch.bool, device=self.device),
    )
    self._poses.quaternions_wxyz[..., 0] = 1.0
    self.joint_positions_local_m: torch.Tensor | None = None
    self.joint_quaternions_local_wxyz: torch.Tensor | None = None
    if retain_joint_poses:
      self.joint_positions_local_m = torch.zeros(
        (num_envs, joint_count, 3), device=self.device
      )
      self.joint_quaternions_local_wxyz = torch.zeros(
        (num_envs, joint_count, 4), device=self.device
      )
      self.joint_quaternions_local_wxyz[..., 0] = 1.0
    joint_indices = {name: index for index, name in enumerate(bank.joint_names)}
    self.anchor_index = joint_indices["Hips"]
    self.foot_indices = torch.as_tensor(
      [
        joint_indices[name]
        for name in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
      ],
      dtype=torch.long,
      device=self.device,
    )
    self.capsule_start_indices = torch.as_tensor(
      [joint_indices[spec.start_joint] for spec in self.capsule_specs],
      dtype=torch.long,
      device=self.device,
    )
    self.capsule_end_indices = torch.as_tensor(
      [
        joint_indices[spec.end_joint]
        if spec.end_joint is not None
        else joint_indices[spec.start_joint]
        for spec in self.capsule_specs
      ],
      dtype=torch.long,
      device=self.device,
    )
    self.base_radii = torch.as_tensor(
      [spec.radius_m for spec in self.capsule_specs],
      dtype=torch.float32,
      device=self.device,
    )
    self.is_sphere = torch.as_tensor(
      [spec.end_joint is None for spec in self.capsule_specs],
      dtype=torch.bool,
      device=self.device,
    )

  def _sample_local(
    self, rows: torch.Tensor, times_s: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    last = self.frame_counts[rows] - 1
    coordinate = times_s * self.bank.fps
    coordinate = torch.minimum(torch.clamp_min(coordinate, 0.0), last.float())
    first = torch.floor(coordinate).long()
    second = torch.minimum(first + 1, last)
    blend = (coordinate - first).float()
    positions = torch.lerp(
      self.local_positions[rows, first].float(),
      self.local_positions[rows, second].float(),
      blend[:, None, None],
    )
    quaternions = _quat_nlerp(
      self.local_quaternions[rows, first].float(),
      self.local_quaternions[rows, second].float(),
      blend[:, None, None],
    )
    return positions, quaternions

  def _apply_alignment(
    self,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    yaw: torch.Tensor,
    translation: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.clone()
    quaternions = quaternions.clone()
    positions[:, 0] = _rotate_xy(positions[:, 0], yaw) + translation
    quaternions[:, 0] = _quat_multiply(
      _yaw_quaternion(yaw), quaternions[:, 0]
    )
    return positions, quaternions

  def _forward_kinematics(
    self, local_positions: torch.Tensor, local_quaternions: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.empty_like(local_positions)
    quaternions = torch.empty_like(local_quaternions)
    for joint_index, parent_index in enumerate(self.bank.parents):
      parent = int(parent_index)
      if parent < 0:
        positions[:, joint_index] = local_positions[:, joint_index]
        quaternions[:, joint_index] = local_quaternions[:, joint_index]
      else:
        positions[:, joint_index] = positions[:, parent] + _quat_apply(
          quaternions[:, parent], local_positions[:, joint_index]
        )
        quaternions[:, joint_index] = _quat_multiply(
          quaternions[:, parent], local_quaternions[:, joint_index]
        )
    return positions, quaternions

  def _prepare_sequences(
    self, env_ids: torch.Tensor, sequence_source_ids: torch.Tensor
  ) -> None:
    rows = self.source_to_row[sequence_source_ids]
    if torch.any(rows < 0):
      raise ValueError("composed sequence contains a path outside the pruned bank")
    self.sequence_rows[env_ids] = rows
    durations = (self.frame_counts[rows] - 1).float() / self.bank.fps
    starts = torch.stack(
      (
        torch.zeros(len(env_ids), device=self.device),
        durations[:, 0],
        durations[:, 0] + durations[:, 1],
      ),
      dim=-1,
    )
    self.segment_durations_s[env_ids] = durations
    self.segment_starts_s[env_ids] = starts
    self.total_durations_s[env_ids] = durations.sum(dim=-1)

    count = len(env_ids)
    align_yaw = torch.zeros((count, 3), device=self.device)
    align_translation = torch.zeros((count, 3, 3), device=self.device)
    for boundary in range(2):
      outgoing_rows = rows[:, boundary]
      incoming_rows = rows[:, boundary + 1]
      outgoing_duration = durations[:, boundary]
      outgoing_prev_time = torch.clamp_min(
        outgoing_duration - 1.0 / self.bank.fps, 0.0
      )
      incoming_next_time = torch.minimum(
        torch.full_like(outgoing_duration, 1.0 / self.bank.fps),
        durations[:, boundary + 1],
      )
      outgoing_last = self._sample_local(outgoing_rows, outgoing_duration)
      outgoing_prev = self._sample_local(outgoing_rows, outgoing_prev_time)
      outgoing_last = self._apply_alignment(
        *outgoing_last,
        align_yaw[:, boundary],
        align_translation[:, boundary],
      )
      outgoing_prev = self._apply_alignment(
        *outgoing_prev,
        align_yaw[:, boundary],
        align_translation[:, boundary],
      )
      incoming_first = self._sample_local(
        incoming_rows, torch.zeros_like(outgoing_duration)
      )
      incoming_second = self._sample_local(incoming_rows, incoming_next_time)
      outgoing_global = self._forward_kinematics(*outgoing_last)
      incoming_global = self._forward_kinematics(*incoming_first)
      outgoing_forward = _quat_apply(
        outgoing_global[1][:, self.anchor_index],
        torch.tensor((0.0, 1.0, 0.0), device=self.device).expand(count, 3),
      )
      incoming_forward = _quat_apply(
        incoming_global[1][:, self.anchor_index],
        torch.tensor((0.0, 1.0, 0.0), device=self.device).expand(count, 3),
      )
      outgoing_yaw = torch.atan2(outgoing_forward[:, 1], outgoing_forward[:, 0])
      incoming_yaw = torch.atan2(incoming_forward[:, 1], incoming_forward[:, 0])
      yaw = outgoing_yaw - incoming_yaw
      rotated_incoming_anchor = _rotate_xy(
        incoming_global[0][:, self.anchor_index], yaw
      )
      translation = (
        outgoing_global[0][:, self.anchor_index] - rotated_incoming_anchor
      )
      align_yaw[:, boundary + 1] = yaw
      align_translation[:, boundary + 1] = translation
      incoming_first = self._apply_alignment(
        *incoming_first, yaw, translation
      )
      incoming_second = self._apply_alignment(
        *incoming_second, yaw, translation
      )

      # Inertialize in global skeleton space. Root translation is split across
      # the synthetic Root and Hips joints in the SOMA hierarchy; subtracting
      # their local coordinates after world alignment double-counts that frame
      # change and produces multi-meter decay during walk/action boundaries.
      outgoing_global = self._forward_kinematics(*outgoing_last)
      outgoing_prev_global = self._forward_kinematics(*outgoing_prev)
      incoming_global = self._forward_kinematics(*incoming_first)
      incoming_next_global = self._forward_kinematics(*incoming_second)
      out_pos, out_quat = outgoing_global
      out_prev_pos, out_prev_quat = outgoing_prev_global
      in_pos, in_quat = incoming_global
      in_next_pos, in_next_quat = incoming_next_global
      self.position_offsets[env_ids, boundary] = out_pos - in_pos
      self.position_velocity_offsets[env_ids, boundary] = (
        (out_pos - out_prev_pos) - (in_next_pos - in_pos)
      ) * self.bank.fps
      self.rotation_offsets[env_ids, boundary] = _quat_log(
        _quat_multiply(out_quat, _quat_inverse(in_quat))
      )
      outgoing_angular_velocity = _quat_log(
        _quat_multiply(out_quat, _quat_inverse(out_prev_quat))
      ) * self.bank.fps
      incoming_angular_velocity = _quat_log(
        _quat_multiply(in_next_quat, _quat_inverse(in_quat))
      ) * self.bank.fps
      self.rotation_velocity_offsets[env_ids, boundary] = (
        outgoing_angular_velocity - incoming_angular_velocity
      )

    self.alignment_yaw[env_ids] = align_yaw
    self.alignment_translation[env_ids] = align_translation

  def _sample_chain(
    self, env_ids: torch.Tensor, local_times_s: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = self.segment_starts_s[env_ids]
    segment = (local_times_s[:, None] >= starts[:, 1:]).sum(dim=-1).long()
    source_times = local_times_s - starts.gather(1, segment[:, None]).squeeze(1)
    rows = self.sequence_rows[env_ids].gather(1, segment[:, None]).squeeze(1)
    local_positions, local_quaternions = self._sample_local(rows, source_times)
    yaw = self.alignment_yaw[env_ids].gather(1, segment[:, None]).squeeze(1)
    translation = self.alignment_translation[env_ids].gather(
      1, segment[:, None, None].expand(-1, 1, 3)
    ).squeeze(1)
    local_positions, local_quaternions = self._apply_alignment(
      local_positions, local_quaternions, yaw, translation
    )

    in_transition = (segment > 0) & (
      source_times <= self.transition_duration_s + 1e-9
    )
    boundary = torch.clamp_min(segment - 1, 0)
    phase = (source_times / self.transition_duration_s).clamp(0.0, 1.0)
    h00 = 2.0 * phase**3 - 3.0 * phase**2 + 1.0
    h10 = phase**3 - 2.0 * phase**2 + phase
    position_offset = self.position_offsets[env_ids, boundary]
    position_velocity = self.position_velocity_offsets[env_ids, boundary]
    position_decay = (
      h00[:, None, None] * position_offset
      + h10[:, None, None]
      * self.transition_duration_s
      * position_velocity
    )
    rotation_offset = self.rotation_offsets[env_ids, boundary]
    rotation_velocity = self.rotation_velocity_offsets[env_ids, boundary]
    rotation_decay = (
      h00[:, None, None] * rotation_offset
      + h10[:, None, None]
      * self.transition_duration_s
      * rotation_velocity
    )
    positions, quaternions = self._forward_kinematics(
      local_positions, local_quaternions
    )
    positions = positions + torch.where(
      in_transition[:, None, None], position_decay, 0.0
    )
    decayed_quaternions = _quat_multiply(
      _quat_exp(rotation_decay), quaternions
    )
    quaternions = torch.where(
      in_transition[:, None, None], decayed_quaternions, quaternions
    )
    active = (local_times_s >= 0.0) & (
      local_times_s <= self.total_durations_s[env_ids]
    )
    return positions, quaternions, active

  def schedule_intersections(
    self,
    env_ids: torch.Tensor,
    *,
    sequence_source_ids: torch.Tensor,
    global_intersection_times_s: torch.Tensor,
    robot_positions_at_intersection_w: torch.Tensor,
    robot_yaw_at_intersection: torch.Tensor,
    robot_path_heading_at_intersection: torch.Tensor,
    action_phase: torch.Tensor | float = 0.5,
    crossing_angle_rad: torch.Tensor | float = torch.pi / 2.0,
    ground_height_m: torch.Tensor | float = 0.0,
    body_scale_xyz: torch.Tensor | None = None,
    radius_scale: torch.Tensor | float = 1.0,
    radius_margin_m: torch.Tensor | float = 0.0,
    playback_speed: torch.Tensor | float = 1.0,
    align_to_facing: bool = False,
    placement_yaw_override: torch.Tensor | None = None,
    translation_override_w: torch.Tensor | None = None,
  ) -> None:
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    count = len(env_ids)
    sequence_source_ids = torch.as_tensor(
      sequence_source_ids, dtype=torch.long, device=self.device
    )
    if sequence_source_ids.shape != (count, 3):
      raise ValueError("sequence_source_ids must have shape [num_envs, 3]")
    self._prepare_sequences(env_ids, sequence_source_ids)

    def tensor(value: object, shape: tuple[int, ...]) -> torch.Tensor:
      return torch.broadcast_to(
        torch.as_tensor(value, dtype=torch.float32, device=self.device), shape
      )

    global_times = tensor(global_intersection_times_s, (count,))
    robot_positions = tensor(robot_positions_at_intersection_w, (count, 3))
    robot_yaw = tensor(robot_yaw_at_intersection, (count,))
    robot_heading = tensor(robot_path_heading_at_intersection, (count,))
    phase = tensor(action_phase, (count,))
    crossing = tensor(crossing_angle_rad, (count,))
    ground = tensor(ground_height_m, (count,))
    body_scale = (
      torch.ones((count, 3), device=self.device)
      if body_scale_xyz is None
      else tensor(body_scale_xyz, (count, 3))
    )
    radius_scale_tensor = tensor(radius_scale, (count,))
    radius_margin_tensor = tensor(radius_margin_m, (count,))
    playback_speed_tensor = tensor(playback_speed, (count,))
    if torch.any((phase < 0.0) | (phase > 1.0)):
      raise ValueError("action phase must be in [0, 1]")
    if (
      torch.any(body_scale <= 0.0)
      or torch.any(radius_scale_tensor <= 0.0)
      or torch.any(playback_speed_tensor <= 0.0)
    ):
      raise ValueError("body, radius, and playback scales must be positive")
    if torch.any(radius_margin_tensor < 0.0):
      raise ValueError("radius margin must be non-negative")
    self.body_scale_xyz[env_ids] = body_scale
    self.radius_scale[env_ids] = radius_scale_tensor
    self.radius_margin_m[env_ids] = radius_margin_tensor
    self.playback_speed[env_ids] = playback_speed_tensor

    local_intersection = (
      self.segment_durations_s[env_ids, 0]
      + phase * self.segment_durations_s[env_ids, 1]
    )
    positions, quaternions, _ = self._sample_chain(env_ids, local_intersection)
    positions = positions * body_scale[:, None]
    window = 2.0 / self.bank.fps
    before = self._sample_chain(env_ids, local_intersection - window)[0]
    after = self._sample_chain(env_ids, local_intersection + window)[0]
    before = before * body_scale[:, None]
    after = after * body_scale[:, None]
    displacement = (
      after[:, self.anchor_index, :2] - before[:, self.anchor_index, :2]
    )
    source_heading = torch.atan2(displacement[:, 1], displacement[:, 0])
    forward = _quat_apply(
      quaternions[:, self.anchor_index],
      torch.tensor((0.0, 1.0, 0.0), device=self.device).expand(count, 3),
    )
    facing_yaw = torch.atan2(forward[:, 1], forward[:, 0])
    source_heading = torch.where(
      torch.linalg.vector_norm(displacement, dim=-1) < 0.03,
      facing_yaw,
      source_heading,
    )
    if align_to_facing:
      source_heading = facing_yaw
    placement_yaw = robot_heading + crossing - source_heading
    if placement_yaw_override is not None:
      placement_yaw = tensor(placement_yaw_override, (count,))
    local_root = positions[:, self.anchor_index]
    rotated_root = _rotate_xy(local_root, placement_yaw)
    target_xy = robot_positions[:, :2]
    foot_ground = positions[:, self.foot_indices, 2].min(dim=-1).values
    translation = torch.cat(
      (
        target_xy - rotated_root[:, :2],
        (ground - foot_ground).unsqueeze(-1),
      ),
      dim=-1,
    )
    if translation_override_w is not None:
      translation = tensor(translation_override_w, (count, 3))
    del robot_yaw  # Reserved for future robot-frame intersection offsets.
    self.global_intersection_times_s[env_ids] = global_times
    self.local_intersection_times_s[env_ids] = local_intersection
    self.placement_yaw[env_ids] = placement_yaw
    self.translation_w[env_ids] = translation
    self.current_translation_w[env_ids] = translation
    self.root_anchor_w[env_ids] = robot_positions
    self.dirty[env_ids] = True
    self.scheduled[env_ids] = True

  def _update(self, global_time_s: float, env_ids: torch.Tensor) -> None:
    local_times = (
      global_time_s
      - self.global_intersection_times_s[env_ids]
    ) * self.playback_speed[env_ids] + self.local_intersection_times_s[env_ids]
    positions, joint_quaternions, active = self._sample_chain(env_ids, local_times)
    if self.joint_positions_local_m is not None:
      assert self.joint_quaternions_local_wxyz is not None
      self.joint_positions_local_m[env_ids] = positions
      self.joint_quaternions_local_wxyz[env_ids] = joint_quaternions
    body_scale = self.body_scale_xyz[env_ids]
    positions = positions * body_scale[:, None]
    positions = _rotate_xy(positions, self.placement_yaw[env_ids])
    positions += self.translation_w[env_ids, None]
    visual_translation = self.translation_w[env_ids].clone()
    if self.lock_root_xy:
      root_shift_xy = (
        self.root_anchor_w[env_ids, :2]
        - positions[:, self.anchor_index, :2]
      )
      positions[..., :2] += root_shift_xy[:, None]
      visual_translation[:, :2] += root_shift_xy
    self.current_translation_w[env_ids] = visual_translation

    start = positions[:, self.capsule_start_indices]
    end = positions[:, self.capsule_end_indices]
    vectors = end - start
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    half_lengths = 0.5 * lengths
    half_lengths = torch.where(self.is_sphere[None], 0.0, half_lengths)
    centers = 0.5 * (start + end)
    centers = torch.where(self.is_sphere[None, :, None], start, centers)
    quaternions = _quat_from_z(vectors)
    identity = torch.zeros_like(quaternions)
    identity[..., 0] = 1.0
    quaternions = torch.where(
      self.is_sphere[None, :, None], identity, quaternions
    )
    cross_section_scale = body_scale[:, :2].mean(dim=-1)
    radii = (
      self.base_radii[None]
      * cross_section_scale[:, None]
      * self.radius_scale[env_ids, None]
      + self.radius_margin_m[env_ids, None]
    )
    inactive_centers = torch.zeros_like(centers)
    inactive_centers[..., 2] = self.inactive_height_m
    centers = torch.where(active[:, None, None], centers, inactive_centers)

    self._poses.centers_w[env_ids] = centers
    self._poses.quaternions_wxyz[env_ids] = quaternions
    self._poses.radii_m[env_ids] = radii
    self._poses.half_lengths_m[env_ids] = half_lengths
    self._poses.root_positions_w[env_ids] = positions[:, self.anchor_index]
    self._poses.active[env_ids] = active
    self.dirty[env_ids] = False
    self.next_update_times_s[env_ids] = global_time_s + self.update_period_s

  def sample_held(self, global_time_s: float) -> OnlineHumanPoses:
    due = self.dirty | (global_time_s + 1e-12 >= self.next_update_times_s)
    env_ids = due.nonzero().flatten()
    if len(env_ids):
      self._update(global_time_s, env_ids)
    self.last_updated_env_ids = env_ids
    return self._poses

  def expired_env_ids(self, global_time_s: float) -> torch.Tensor:
    local_times = (
      global_time_s
      - self.global_intersection_times_s
    ) * self.playback_speed + self.local_intersection_times_s
    expired = (local_times > self.total_durations_s + 1e-9) & self.scheduled
    return expired.nonzero().flatten()
