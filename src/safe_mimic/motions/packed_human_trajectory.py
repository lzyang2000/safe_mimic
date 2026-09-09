"""GPU-resident playback for offline-compiled human capsule trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from safe_mimic.motions.capsule_path_bank import OnlineHumanPoses


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
  unit = vectors / length.clamp_min(1.0e-12)
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
  ).clamp_min(1.0e-12)


@dataclass(frozen=True)
class PackedCapsuleSpec:
  """Capsule definition encoded in a packed trajectory bank."""

  name: str
  start_keypoint: str
  end_keypoint: str | None
  radius_m: float


class PackedHumanTrajectoryBank:
  """Memory-map offline-stitched keypoint trajectories."""

  def __init__(self, path: Path | str) -> None:
    self.path = Path(path)
    with (self.path / "bank.json").open() as source:
      self.config = json.load(source)
    if self.config.get("format_version") != 1 or not self.config.get("complete"):
      raise ValueError(f"{self.path} is not a complete packed human bank")
    self.fps = float(self.config["fps"])
    self.max_frames = int(self.config["max_frames"])
    self.keypoint_names = tuple(self.config["keypoint_names"])
    self.capsule_specs = tuple(
      PackedCapsuleSpec(
        name=str(record["name"]),
        start_keypoint=str(record["start_keypoint"]),
        end_keypoint=(
          None if record.get("end_keypoint") is None else str(record["end_keypoint"])
        ),
        radius_m=float(record["radius_m"]),
      )
      for record in self.config["capsules"]
    )
    self.root_keypoint = str(self.config.get("root_keypoint", "Hips"))
    self.foot_keypoints = tuple(
      self.config.get(
        "foot_keypoints",
        ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"),
      )
    )
    self.keypoints = np.load(self.path / "keypoints.npy", mmap_mode="r")
    self.facing_yaw = np.load(self.path / "facing_yaw.npy", mmap_mode="r")
    self.frame_counts = np.load(self.path / "frame_counts.npy", mmap_mode="r")
    self.action_frames = np.load(self.path / "action_frames.npy", mmap_mode="r")
    self.entry_travel_m = np.load(self.path / "entry_travel_m.npy", mmap_mode="r")
    self.sequence_path_ids = np.load(self.path / "sequence_path_ids.npy", mmap_mode="r")
    self._validate()

  def _validate(self) -> None:
    path_count = int(self.config["path_count"])
    keypoint_count = len(self.keypoint_names)
    if self.keypoints.shape != (
      path_count,
      self.max_frames,
      keypoint_count,
      3,
    ):
      raise ValueError(f"invalid packed keypoint shape {self.keypoints.shape}")
    if self.facing_yaw.shape != (path_count, self.max_frames):
      raise ValueError("invalid packed facing-yaw shape")
    if self.frame_counts.shape != (path_count,):
      raise ValueError("invalid packed frame-count shape")
    if self.action_frames.shape != (path_count,):
      raise ValueError("invalid packed action-frame shape")
    if self.entry_travel_m.shape != (path_count,):
      raise ValueError("invalid packed entry-travel shape")
    if self.sequence_path_ids.shape != (path_count, 3):
      raise ValueError("invalid packed source-sequence shape")
    if np.any(self.frame_counts < 2) or np.any(self.frame_counts > self.max_frames):
      raise ValueError("packed trajectory frame count lies outside capacity")
    if self.root_keypoint not in self.keypoint_names:
      raise ValueError("packed root keypoint is missing")
    missing_feet = set(self.foot_keypoints) - set(self.keypoint_names)
    if missing_feet:
      raise ValueError(f"packed foot keypoints are missing: {sorted(missing_feet)}")
    known = set(self.keypoint_names)
    for spec in self.capsule_specs:
      if spec.start_keypoint not in known or (
        spec.end_keypoint is not None and spec.end_keypoint not in known
      ):
        raise ValueError(f"packed capsule {spec.name} references a missing keypoint")

  def __len__(self) -> int:
    return len(self.frame_counts)


class PackedHumanTrajectorySampler:
  """Gather and place precompiled human keypoint frames entirely on-device."""

  def __init__(
    self,
    bank: PackedHumanTrajectoryBank,
    num_agents: int,
    device: str | torch.device,
    *,
    update_hz: float,
    inactive_height_m: float = -100.0,
    loop: bool = False,
    lock_root_xy: bool = False,
  ) -> None:
    if num_agents < 1 or update_hz <= 0.0:
      raise ValueError("packed sampler dimensions and update rate must be positive")
    self.bank = bank
    self.num_agents = num_agents
    self.device = torch.device(device)
    self.update_period_s = 1.0 / update_hz
    self.inactive_height_m = inactive_height_m
    self.loop = loop
    self.lock_root_xy = lock_root_xy

    def load(array: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
      tensor = torch.as_tensor(np.asarray(array).copy(), device=self.device)
      return tensor.to(dtype=dtype) if dtype is not None else tensor

    self.keypoints = load(bank.keypoints)
    self.facing_yaw = load(bank.facing_yaw)
    self.frame_counts = load(bank.frame_counts, dtype=torch.long)
    self.action_frames = load(bank.action_frames, dtype=torch.long)
    self.entry_travel_m = load(bank.entry_travel_m, dtype=torch.float32)
    self.sequence_path_ids = load(bank.sequence_path_ids, dtype=torch.long)
    names = {name: index for index, name in enumerate(bank.keypoint_names)}
    self.root_index = names[bank.root_keypoint]
    self.foot_indices = torch.as_tensor(
      [names[name] for name in bank.foot_keypoints],
      dtype=torch.long,
      device=self.device,
    )
    self.start_indices = torch.as_tensor(
      [names[spec.start_keypoint] for spec in bank.capsule_specs],
      dtype=torch.long,
      device=self.device,
    )
    self.end_indices = torch.as_tensor(
      [
        names[spec.end_keypoint]
        if spec.end_keypoint is not None
        else names[spec.start_keypoint]
        for spec in bank.capsule_specs
      ],
      dtype=torch.long,
      device=self.device,
    )
    self.is_sphere = torch.as_tensor(
      [spec.end_keypoint is None for spec in bank.capsule_specs],
      dtype=torch.bool,
      device=self.device,
    )
    self.base_radii = torch.as_tensor(
      [spec.radius_m for spec in bank.capsule_specs],
      dtype=torch.float32,
      device=self.device,
    )

    self.path_ids = torch.zeros(num_agents, dtype=torch.long, device=self.device)
    self.global_intersection_times_s = torch.zeros(num_agents, device=self.device)
    self.local_intersection_times_s = torch.zeros(num_agents, device=self.device)
    self.playback_speed = torch.ones(num_agents, device=self.device)
    self.placement_yaw = torch.zeros(num_agents, device=self.device)
    self.translation_w = torch.zeros((num_agents, 3), device=self.device)
    self.root_anchor_w = torch.zeros((num_agents, 3), device=self.device)
    self.body_scale_xyz = torch.ones((num_agents, 3), device=self.device)
    self.radius_scale = torch.ones(num_agents, device=self.device)
    self.radius_margin_m = torch.zeros(num_agents, device=self.device)
    self.enabled = torch.zeros(num_agents, dtype=torch.bool, device=self.device)
    self.dirty = torch.zeros(num_agents, dtype=torch.bool, device=self.device)
    self._next_global_update_s = -float("inf")
    self.last_updated_ids = torch.empty(0, dtype=torch.long, device=self.device)

    capsule_count = len(bank.capsule_specs)
    self._poses = OnlineHumanPoses(
      centers_w=torch.zeros((num_agents, capsule_count, 3), device=self.device),
      quaternions_wxyz=torch.zeros((num_agents, capsule_count, 4), device=self.device),
      radii_m=torch.zeros((num_agents, capsule_count), device=self.device),
      half_lengths_m=torch.zeros((num_agents, capsule_count), device=self.device),
      root_positions_w=torch.zeros((num_agents, 3), device=self.device),
      active=torch.zeros(num_agents, dtype=torch.bool, device=self.device),
    )
    self._poses.quaternions_wxyz[..., 0] = 1.0
    resident = (
      self.keypoints,
      self.facing_yaw,
      self.frame_counts,
      self.action_frames,
      self.entry_travel_m,
      self.sequence_path_ids,
    )
    self.device_storage_bytes = sum(
      tensor.numel() * tensor.element_size() for tensor in resident
    )

  def action_times_s(self, path_ids: torch.Tensor) -> torch.Tensor:
    frames = self.action_frames[path_ids]
    if torch.any(frames < 0):
      raise ValueError("selected packed trajectory has no annotated action frame")
    return frames.float() / self.bank.fps

  def _sample_keypoints(
    self, path_ids: torch.Tensor, local_times_s: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last = self.frame_counts[path_ids] - 1
    coordinate = local_times_s * self.bank.fps
    if self.loop:
      # Ping-pong playback keeps the loop boundary position-continuous without
      # requiring the third offline segment to match the first pose exactly.
      # Crowd clips are stationary arm actions, so the endpoint velocity
      # reversal is preferable to a visible pose teleport.
      duration_frames = last.clamp_min(1).float()
      cycle_frames = 2.0 * duration_frames
      coordinate = torch.remainder(coordinate, cycle_frames)
      coordinate = torch.where(
        coordinate <= duration_frames,
        coordinate,
        cycle_frames - coordinate,
      )
      active = torch.ones_like(coordinate, dtype=torch.bool)
    else:
      active = (coordinate >= 0.0) & (coordinate <= last)
    coordinate = torch.minimum(torch.clamp_min(coordinate, 0.0), last.float())
    first = torch.floor(coordinate).long()
    second = torch.minimum(first + 1, last)
    blend = coordinate - first
    first_points = self.keypoints[path_ids, first].float()
    second_points = self.keypoints[path_ids, second].float()
    points = torch.lerp(first_points, second_points, blend[:, None, None])
    first_yaw = self.facing_yaw[path_ids, first].float()
    second_yaw = self.facing_yaw[path_ids, second].float()
    delta = torch.atan2(
      torch.sin(second_yaw - first_yaw), torch.cos(second_yaw - first_yaw)
    )
    yaw = first_yaw + blend * delta
    return points, yaw, active

  def entry_times_for_distance(
    self,
    path_ids: torch.Tensor,
    desired_distance_m: torch.Tensor,
  ) -> torch.Tensor:
    """Choose a pre-action frame with the requested remaining root travel."""

    action_frames = self.action_frames[path_ids]
    if torch.any(action_frames <= 0):
      raise ValueError("entry search requires positive action-frame annotations")
    roots = self.keypoints[path_ids, :, self.root_index, :2].float()
    action_roots = roots.gather(
      1, action_frames[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    distance = torch.linalg.vector_norm(action_roots[:, None] - roots, dim=-1)
    frames = torch.arange(self.bank.max_frames, device=self.device)[None]
    valid = frames <= action_frames[:, None]
    error = torch.where(
      valid,
      torch.abs(distance - desired_distance_m[:, None]),
      torch.full_like(distance, torch.inf),
    )
    return torch.argmin(error, dim=-1).float() / self.bank.fps

  def _path_heading(
    self,
    path_ids: torch.Tensor,
    local_times_s: torch.Tensor,
    *,
    align_to_facing: bool,
  ) -> torch.Tensor:
    points, facing, _ = self._sample_keypoints(path_ids, local_times_s)
    if align_to_facing:
      return facing
    window = 2.0 / self.bank.fps
    before = self._sample_keypoints(path_ids, local_times_s - window)[0]
    after = self._sample_keypoints(path_ids, local_times_s + window)[0]
    displacement = after[:, self.root_index, :2] - before[:, self.root_index, :2]
    heading = torch.atan2(displacement[:, 1], displacement[:, 0])
    return torch.where(
      torch.linalg.vector_norm(displacement, dim=-1) < 0.03,
      facing,
      heading,
    )

  def schedule_intersections(
    self,
    agent_ids: torch.Tensor,
    *,
    path_ids: torch.Tensor,
    global_intersection_times_s: torch.Tensor,
    local_intersection_times_s: torch.Tensor,
    target_positions_w: torch.Tensor,
    target_heading: torch.Tensor,
    crossing_angle_rad: torch.Tensor | float,
    ground_height_m: torch.Tensor,
    body_scale_xyz: torch.Tensor,
    radius_scale: torch.Tensor,
    radius_margin_m: torch.Tensor,
    playback_speed: torch.Tensor,
    align_to_facing: bool = False,
    heading_start_times_s: torch.Tensor | None = None,
  ) -> None:
    agent_ids = agent_ids.to(device=self.device, dtype=torch.long)
    path_ids = path_ids.to(device=self.device, dtype=torch.long)
    count = len(agent_ids)
    if path_ids.shape != (count,):
      raise ValueError("packed agent and path vectors must have equal length")
    if torch.any((path_ids < 0) | (path_ids >= len(self.bank))):
      raise ValueError("packed path id lies outside the bank")
    global_intersection_times_s = global_intersection_times_s.to(
      device=self.device, dtype=torch.float32
    )
    local_intersection_times_s = local_intersection_times_s.to(
      device=self.device, dtype=torch.float32
    )
    target_positions_w = target_positions_w.to(device=self.device, dtype=torch.float32)
    target_heading = target_heading.to(device=self.device, dtype=torch.float32)
    ground_height_m = ground_height_m.to(device=self.device, dtype=torch.float32)
    body_scale_xyz = body_scale_xyz.to(device=self.device, dtype=torch.float32)
    radius_scale = radius_scale.to(device=self.device, dtype=torch.float32)
    radius_margin_m = radius_margin_m.to(device=self.device, dtype=torch.float32)
    playback_speed = playback_speed.to(device=self.device, dtype=torch.float32)
    if heading_start_times_s is not None:
      heading_start_times_s = heading_start_times_s.to(
        device=self.device, dtype=torch.float32
      )
    vector_shapes = (
      global_intersection_times_s.shape,
      local_intersection_times_s.shape,
      target_heading.shape,
      ground_height_m.shape,
      radius_scale.shape,
      radius_margin_m.shape,
      playback_speed.shape,
    )
    if any(shape != (count,) for shape in vector_shapes):
      raise ValueError("packed schedule vectors must have one value per agent")
    if target_positions_w.shape != (count, 3) or body_scale_xyz.shape != (count, 3):
      raise ValueError("packed position and body-scale arrays must have shape (N, 3)")
    if heading_start_times_s is not None and heading_start_times_s.shape != (count,):
      raise ValueError("packed heading-start times must have one value per agent")
    if torch.any(body_scale_xyz <= 0.0) or torch.any(playback_speed <= 0.0):
      raise ValueError("packed body scales and playback speeds must be positive")
    crossing = torch.broadcast_to(
      torch.as_tensor(
        crossing_angle_rad,
        dtype=torch.float32,
        device=self.device,
      ),
      (count,),
    )
    fallback_heading = self._path_heading(
      path_ids, local_intersection_times_s, align_to_facing=align_to_facing
    )
    if heading_start_times_s is None:
      source_heading = fallback_heading
    else:
      start_keypoints = self._sample_keypoints(path_ids, heading_start_times_s)[0]
      end_keypoints = self._sample_keypoints(path_ids, local_intersection_times_s)[0]
      displacement = (
        end_keypoints[:, self.root_index, :2] - start_keypoints[:, self.root_index, :2]
      ) * body_scale_xyz[:, :2]
      source_heading = torch.atan2(displacement[:, 1], displacement[:, 0])
      source_heading = torch.where(
        torch.linalg.vector_norm(displacement, dim=-1) < 0.03,
        fallback_heading,
        source_heading,
      )
    placement_yaw = target_heading + crossing - source_heading
    keypoints = self._sample_keypoints(path_ids, local_intersection_times_s)[0]
    keypoints = keypoints * body_scale_xyz[:, None]
    root = keypoints[:, self.root_index]
    rotated_root = _rotate_xy(root, placement_yaw)
    foot_ground = keypoints[:, self.foot_indices, 2].min(dim=-1).values
    translation = torch.cat(
      (
        target_positions_w[:, :2] - rotated_root[:, :2],
        (ground_height_m - foot_ground).unsqueeze(-1),
      ),
      dim=-1,
    )
    self.path_ids[agent_ids] = path_ids
    self.global_intersection_times_s[agent_ids] = global_intersection_times_s
    self.local_intersection_times_s[agent_ids] = local_intersection_times_s
    self.playback_speed[agent_ids] = playback_speed
    self.placement_yaw[agent_ids] = placement_yaw
    self.translation_w[agent_ids] = translation
    self.root_anchor_w[agent_ids] = target_positions_w
    self.body_scale_xyz[agent_ids] = body_scale_xyz
    self.radius_scale[agent_ids] = radius_scale
    self.radius_margin_m[agent_ids] = radius_margin_m
    self.enabled[agent_ids] = True
    self.dirty[agent_ids] = True

  def deactivate(self, agent_ids: torch.Tensor) -> None:
    if agent_ids.numel() == 0:
      return
    agent_ids = agent_ids.to(device=self.device, dtype=torch.long)
    self.enabled[agent_ids] = False
    self.dirty[agent_ids] = False
    self._poses.centers_w[agent_ids] = 0.0
    self._poses.centers_w[agent_ids, :, 2] = self.inactive_height_m
    self._poses.quaternions_wxyz[agent_ids] = 0.0
    self._poses.quaternions_wxyz[agent_ids, :, 0] = 1.0
    self._poses.radii_m[agent_ids] = self.base_radii
    self._poses.half_lengths_m[agent_ids] = torch.where(
      self.is_sphere,
      torch.zeros_like(self.base_radii),
      torch.full_like(self.base_radii, 0.05),
    )
    self._poses.active[agent_ids] = False

  def _update(self, global_time_s: float, agent_ids: torch.Tensor) -> None:
    local_times = (
      global_time_s - self.global_intersection_times_s[agent_ids]
    ) * self.playback_speed[agent_ids] + self.local_intersection_times_s[agent_ids]
    keypoints, _, active = self._sample_keypoints(self.path_ids[agent_ids], local_times)
    scale = self.body_scale_xyz[agent_ids]
    keypoints = keypoints * scale[:, None]
    keypoints = _rotate_xy(keypoints, self.placement_yaw[agent_ids])
    keypoints += self.translation_w[agent_ids, None]
    if self.lock_root_xy:
      root_shift = self.root_anchor_w[agent_ids, :2] - keypoints[:, self.root_index, :2]
      keypoints[..., :2] += root_shift[:, None]

    start = keypoints[:, self.start_indices]
    end = keypoints[:, self.end_indices]
    vectors = end - start
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    half_lengths = torch.where(self.is_sphere[None], 0.0, 0.5 * lengths)
    centers = torch.where(self.is_sphere[None, :, None], start, 0.5 * (start + end))
    quaternions = _quat_from_z(vectors)
    identity = torch.zeros_like(quaternions)
    identity[..., 0] = 1.0
    quaternions = torch.where(self.is_sphere[None, :, None], identity, quaternions)
    cross_section_scale = scale[:, :2].mean(dim=-1)
    radii = (
      self.base_radii[None]
      * cross_section_scale[:, None]
      * self.radius_scale[agent_ids, None]
      + self.radius_margin_m[agent_ids, None]
    )
    active &= self.enabled[agent_ids]
    inactive_centers = torch.zeros_like(centers)
    inactive_centers[..., 2] = self.inactive_height_m
    centers = torch.where(active[:, None, None], centers, inactive_centers)
    self._poses.centers_w[agent_ids] = centers
    self._poses.quaternions_wxyz[agent_ids] = quaternions
    self._poses.radii_m[agent_ids] = radii
    self._poses.half_lengths_m[agent_ids] = half_lengths
    self._poses.root_positions_w[agent_ids] = keypoints[:, self.root_index]
    self._poses.active[agent_ids] = active
    self.dirty[agent_ids] = False

  def update_due(self, global_time_s: float) -> bool:
    return global_time_s + 1.0e-9 >= self._next_global_update_s

  def sample_held(
    self,
    global_time_s: float,
    *,
    force_ids: torch.Tensor | None = None,
  ) -> OnlineHumanPoses:
    ids = torch.empty(0, dtype=torch.long, device=self.device)
    if self.update_due(global_time_s):
      ids = self.enabled.nonzero().flatten()
      self._next_global_update_s = global_time_s + self.update_period_s
    if force_ids is not None and force_ids.numel():
      force_ids = force_ids.to(device=self.device, dtype=torch.long)
      ids = torch.unique(torch.cat((ids, force_ids[self.enabled[force_ids]])))
    if ids.numel():
      self._update(global_time_s, ids)
    self.last_updated_ids = ids
    return self._poses

  def expired_ids(self, global_time_s: float) -> torch.Tensor:
    if self.loop:
      return torch.empty(0, dtype=torch.long, device=self.device)
    ids = self.enabled.nonzero().flatten()
    if ids.numel() == 0:
      return ids
    local_times = (
      global_time_s - self.global_intersection_times_s[ids]
    ) * self.playback_speed[ids] + self.local_intersection_times_s[ids]
    duration = (self.frame_counts[self.path_ids[ids]] - 1).float() / self.bank.fps
    return ids[local_times > duration]


__all__ = [
  "PackedCapsuleSpec",
  "PackedHumanTrajectoryBank",
  "PackedHumanTrajectorySampler",
]
