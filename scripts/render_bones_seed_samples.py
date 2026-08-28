#!/usr/bin/env python3
# ruff: noqa: I001
"""Render sampled BONES-SEED motions with mesh/capsule overlays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from safe_mimic.motions import (
  SOMA_CAPSULE_SPECS,
  BvhMotion,
  CapsuleDomainRandomization,
  CapsuleFit,
  SomaMeshSkin,
  fit_soma_capsules,
  load_bvh_samples,
  load_bvh_window,
  load_soma_mesh_skin,
  sample_capsule_domain_randomization,
)


GROUP_COLORS = {
  "walking": (74, 189, 242),
  "kick": (102, 210, 142),
}

PLAIN_WALK_DESCRIPTIONS = (
  "walking facing forward",
  "walk backwards",
  "walking sideways to the right",
  "walking sideways to the left",
  "walking clockwise in an arc",
  "walking counterclockwise in an arc, leftside",
  "walk front right diagonal",
  "advancing with a leftward diagonal stride",
  "neutral casual random walk",
  "strolling forward",
)


@dataclass(frozen=True)
class Sample:
  group: str
  row: dict[str, str]


@dataclass
class RenderState:
  model: mujoco.MjModel
  data: mujoco.MjData
  renderer: mujoco.Renderer
  camera: mujoco.MjvCamera
  mesh_id: int
  mesh_geom_id: int
  mesh_rotation: np.ndarray
  capsule_geom_ids: np.ndarray
  capsule_mocap_ids: np.ndarray


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/bones-seed"))
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/bones-seed/renders/plain_walk_kick_capsules.mp4"),
  )
  parser.add_argument("--seed", type=int, default=20260826)
  parser.add_argument("--fps", type=int, default=24)
  parser.add_argument("--seconds-per-clip", type=float, default=4.0)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--preview-only", action="store_true")
  return parser.parse_args()


def _one_per_description(
  rows: list[dict[str, str]],
  descriptions: tuple[str, ...],
  rng: random.Random,
) -> list[dict[str, str]]:
  chosen: list[dict[str, str]] = []
  used_actors: set[str] = set()
  for description in descriptions:
    candidates = [
      row
      for row in rows
      if row["content_short_description"] == description
      and int(row["move_duration_frames"]) >= 480
    ]
    rng.shuffle(candidates)
    candidate = next(
      (row for row in candidates if row["take_actor"] not in used_actors),
      candidates[0],
    )
    chosen.append(candidate)
    used_actors.add(candidate["take_actor"])
  return chosen


def _diverse_actor_sample(
  rows: list[dict[str, str]], count: int, rng: random.Random
) -> list[dict[str, str]]:
  candidates = rows.copy()
  rng.shuffle(candidates)
  chosen: list[dict[str, str]] = []
  used_actors: set[str] = set()
  for row in candidates:
    if row["take_actor"] in used_actors:
      continue
    chosen.append(row)
    used_actors.add(row["take_actor"])
    if len(chosen) == count:
      return chosen
  raise ValueError(f"Only found {len(chosen)} distinct actors for {count} samples")


def select_samples(metadata_path: Path, seed: int) -> list[Sample]:
  with metadata_path.open(newline="") as source:
    rows = list(csv.DictReader(source))
  originals = [row for row in rows if row["is_mirror"] == "False"]
  rng = random.Random(seed)

  walking = [
    row
    for row in originals
    if row["package"] == "Locomotion"
    and row["category"] == "Basic Locomotion Neutral"
    and row["content_type_of_movement"] in {"walking", "walking, turning"}
    and row["content_props"] == "0"
    and row["content_uniform_style"] == "neutral"
  ]
  kicks = [
    row
    for row in originals
    if row["content_short_description"] == "kicking trash"
    and row["content_uniform_style"] == "neutral"
    and row["content_props"] == "0"
    and int(row["move_duration_frames"]) >= 600
  ]

  groups = {
    "walking": _one_per_description(walking, PLAIN_WALK_DESCRIPTIONS, rng),
    "kick": _diverse_actor_sample(kicks, 10, rng),
  }
  return [Sample(group, row) for group, selected in groups.items() for row in selected]


def _window_start_time(motion: BvhMotion, group: str, duration_s: float) -> float:
  max_start_s = max(0.0, motion.duration_s - duration_s)
  if group != "kick" or max_start_s == 0.0:
    return 0.5 * max_start_s

  indices = {name: index for index, name in enumerate(motion.joint_names)}
  hips = motion.positions_m[:, indices["Hips"]]
  foot_positions = np.stack(
    [
      motion.positions_m[:, indices["LeftToeBase"]] - hips,
      motion.positions_m[:, indices["RightToeBase"]] - hips,
    ],
    axis=1,
  )
  times_s = motion.frame_indices * motion.source_frame_time
  delta_t = np.maximum(np.diff(times_s), 1e-9)
  speeds = np.linalg.norm(np.diff(foot_positions, axis=0), axis=-1) / delta_t[:, None]
  peak_speed = speeds.max(axis=1)
  smoothing_width = min(7, len(peak_speed))
  if smoothing_width > 1:
    kernel = np.full(smoothing_width, 1.0 / smoothing_width)
    peak_speed = np.convolve(peak_speed, kernel, mode="same")
  peak_time_s = times_s[int(np.argmax(peak_speed)) + 1]
  return float(np.clip(peak_time_s - 0.5 * duration_s, 0.0, max_start_s))


def _load_realtime_motion(
  path: Path, group: str, duration_s: float, output_fps: int
) -> tuple[BvhMotion, float]:
  probe = load_bvh_samples(path, sample_count=480)
  start_time_s = _window_start_time(probe, group, duration_s)
  motion = load_bvh_window(
    path,
    start_time_s=start_time_s,
    duration_s=duration_s,
    output_fps=output_fps,
  )
  return motion, start_time_s


def _build_model_xml(mesh_path: Path, fit: CapsuleFit, width: int, height: int) -> str:
  bodies = []
  for geom_index, spec in enumerate(SOMA_CAPSULE_SPECS):
    radius = fit.radii_m[geom_index]
    half_length = fit.half_lengths_m[geom_index]
    if spec.end_joint is None:
      geom = f'type="sphere" size="{radius:.7g}"'
    else:
      geom = f'type="capsule" size="{radius:.7g} {half_length:.7g}"'
    bodies.append(
      f'<body name="capsule_{spec.name}" mocap="true">'
      f'<geom name="capsule_geom_{spec.name}" {geom} '
      'contype="1" conaffinity="1" group="1" '
      'rgba="0.15 0.75 1 0.34"/>'
      "</body>"
    )
  return (
    "<mujoco>"
    '<option gravity="0 0 0"/>'
    f'<visual><global offwidth="{width}" offheight="{height}"/>'
    '<map znear="0.01"/></visual>'
    "<asset>"
    f'<mesh name="soma_debug_mesh" file="{mesh_path.resolve()}"/>'
    "</asset>"
    "<worldbody>"
    '<light pos="-2 -4 6" dir="0.25 0.4 -1" diffuse="1 1 1"/>'
    '<light pos="3 1 4" dir="-0.4 -0.1 -1" diffuse="0.45 0.5 0.6"/>'
    '<geom name="floor" type="plane" size="5 5 0.1" '
    'rgba="0.10 0.12 0.15 1" contype="0" conaffinity="0"/>'
    '<body name="debug_mesh_body">'
    '<geom name="debug_mesh" type="mesh" mesh="soma_debug_mesh" '
    'rgba="0.80 0.63 0.48 0.58" contype="0" conaffinity="0" group="2"/>'
    "</body>" + "".join(bodies) + "</worldbody></mujoco>"
  )


def _quat_rotation(quaternion: np.ndarray) -> np.ndarray:
  w, x, y, z = quaternion
  return np.asarray(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def make_render_state(
  mesh_path: Path, fit: CapsuleFit, width: int, height: int
) -> RenderState:
  model = mujoco.MjModel.from_xml_string(
    _build_model_xml(mesh_path, fit, width, height)
  )
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=height, width=width)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = (0.0, 0.0, 0.92)
  camera.distance = 3.15
  camera.azimuth = 145.0
  camera.elevation = -7.0
  mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "soma_debug_mesh")
  mesh_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "debug_mesh")
  capsule_geom_ids = []
  capsule_mocap_ids = []
  for spec in SOMA_CAPSULE_SPECS:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"capsule_{spec.name}")
    capsule_mocap_ids.append(model.body_mocapid[body_id])
    capsule_geom_ids.append(
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"capsule_geom_{spec.name}")
    )
  return RenderState(
    model=model,
    data=data,
    renderer=renderer,
    camera=camera,
    mesh_id=mesh_id,
    mesh_geom_id=mesh_geom_id,
    mesh_rotation=_quat_rotation(model.mesh_quat[mesh_id]),
    capsule_geom_ids=np.asarray(capsule_geom_ids, dtype=np.int32),
    capsule_mocap_ids=np.asarray(capsule_mocap_ids, dtype=np.int32),
  )


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
  edges_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
  edges_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
  face_normals = np.cross(edges_a, edges_b)
  normals = np.zeros_like(vertices)
  for corner in range(3):
    np.add.at(normals, triangles[:, corner], face_normals)
  normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
  return normals


def update_scene(
  state: RenderState,
  mesh: SomaMeshSkin,
  mesh_vertices: np.ndarray,
  fit: CapsuleFit,
  frame_index: int,
  group: str,
) -> np.ndarray:
  root_xy = fit.root_path_m[frame_index, :2]
  center_offset = np.asarray([root_xy[0], root_xy[1], 0.0])
  centered_vertices = mesh_vertices - center_offset
  mesh_pos = state.model.mesh_pos[state.mesh_id]
  internal_vertices = (centered_vertices - mesh_pos) @ state.mesh_rotation
  state.model.mesh_vert[:] = internal_vertices
  normals = _vertex_normals(centered_vertices, mesh.triangles)
  state.model.mesh_normal[:] = normals @ state.mesh_rotation
  mujoco.mjr_uploadMesh(state.model, state.renderer._mjr_context, state.mesh_id)

  centers = fit.centers_m[frame_index] - center_offset
  state.data.mocap_pos[state.capsule_mocap_ids] = centers
  state.data.mocap_quat[state.capsule_mocap_ids] = fit.quaternions_wxyz[frame_index]
  for index, geom_id in enumerate(state.capsule_geom_ids):
    state.model.geom_size[geom_id, 0] = fit.radii_m[index]
    if SOMA_CAPSULE_SPECS[index].end_joint is not None:
      state.model.geom_size[geom_id, 1] = fit.half_lengths_m[index]
    red, green, blue = GROUP_COLORS[group]
    state.model.geom_rgba[geom_id] = (red / 255, green / 255, blue / 255, 0.34)

  mujoco.mj_forward(state.model, state.data)
  state.renderer.update_scene(state.data, camera=state.camera)
  return state.renderer.render()


def _font(size: int) -> ImageFont.FreeTypeFont:
  return ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size
  )


def _draw_root_path(
  draw: ImageDraw.ImageDraw,
  fit: CapsuleFit,
  frame_index: int,
  origin: tuple[int, int],
  size: int,
  color: tuple[int, int, int],
) -> None:
  path = fit.root_path_m[:, :2] - fit.root_path_m[0, :2]
  extent = np.ptp(path, axis=0)
  scale = (size - 18) / max(float(extent.max()), 0.5)
  center = path.min(axis=0) + 0.5 * extent
  points = []
  for point in path:
    xy = (point - center) * scale
    points.append((origin[0] + size / 2 + xy[0], origin[1] + size / 2 - xy[1]))
  draw.rounded_rectangle(
    (origin[0], origin[1], origin[0] + size, origin[1] + size),
    radius=10,
    fill=(10, 14, 20, 205),
    outline=(110, 120, 135, 180),
    width=1,
  )
  if len(points) > 1:
    draw.line(points, fill=(*color, 100), width=3)
    draw.line(points[: frame_index + 1], fill=(*color, 255), width=4)
  x, y = points[frame_index]
  draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(*color, 255))


def annotate_frame(
  frame: np.ndarray,
  sample: Sample,
  group_index: int,
  motion: BvhMotion,
  fit: CapsuleFit,
  randomization: CapsuleDomainRandomization,
  frame_index: int,
  window_start_s: float,
  output_fps: int,
  group_total: int = 10,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
  draw = ImageDraw.Draw(overlay)
  width, height = image.size
  color = GROUP_COLORS[sample.group]
  draw.rectangle((0, 0, width, 112), fill=(7, 10, 16, 220))
  draw.text(
    (36, 20),
    f"{sample.group.upper()}  {group_index + 1}/{group_total}",
    font=_font(30),
    fill=(*color, 255),
  )
  description = sample.row["content_short_description"]
  draw.text((36, 61), description[:72], font=_font(24), fill=(245, 247, 250, 255))
  move_name = sample.row["move_name"]
  draw.text(
    (36, height - 72),
    move_name[:80],
    font=_font(18),
    fill=(225, 230, 238, 255),
  )
  scales = randomization.body_scale_xyz
  detail = (
    f"source {motion.duration_s:.1f}s  |  real-time window "
    f"{window_start_s:.1f}-{window_start_s + len(motion.positions_m) / output_fps:.1f}s"
    "  |  body scale "
    f"{scales[0]:.2f}/{scales[1]:.2f}/{scales[2]:.2f}  |  "
    f"capsule radius {randomization.radius_scale:.2f}x + "
    f"{randomization.radius_margin_m * 1000:.0f}mm"
  )
  draw.text((36, height - 43), detail, font=_font(16), fill=(170, 180, 193, 255))
  draw.text(
    (width - 218, 123),
    "pelvis path",
    font=_font(15),
    fill=(225, 230, 238, 230),
  )
  _draw_root_path(draw, fit, frame_index, (width - 218, 150), 176, color)
  return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _write_manifest(path: Path, samples: list[Sample]) -> None:
  records = [
    {
      "group": sample.group,
      "move_name": sample.row["move_name"],
      "path": sample.row["move_soma_uniform_path"],
      "description": sample.row["content_short_description"],
    }
    for sample in samples
  ]
  path.write_text(json.dumps(records, indent=2) + "\n")


def _ffmpeg_process(path: Path, width: int, height: int, fps: int):
  command = [
    "ffmpeg",
    "-y",
    "-loglevel",
    "error",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-s",
    f"{width}x{height}",
    "-r",
    str(fps),
    "-i",
    "-",
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(path),
  ]
  return subprocess.Popen(command, stdin=subprocess.PIPE)


def main() -> None:
  args = parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  mesh_usd = args.dataset_root / "soma_shapes/soma_base_rig/soma_base_skel_minimal.usd"
  mesh_obj = mesh_usd.with_name("soma_base_debug.obj")
  mesh = load_soma_mesh_skin(mesh_usd)
  if not mesh_obj.exists():
    mesh.write_mujoco_obj(mesh_obj)

  samples = select_samples(
    args.dataset_root / "metadata/seed_metadata_v004.csv", args.seed
  )
  _write_manifest(args.output.with_suffix(".samples.json"), samples)
  rng = np.random.default_rng(args.seed)

  first_motion, first_window_start_s = _load_realtime_motion(
    args.dataset_root / samples[0].row["move_soma_uniform_path"],
    samples[0].group,
    args.seconds_per_clip,
    args.fps,
  )
  first_randomization = sample_capsule_domain_randomization(rng)
  first_fit = fit_soma_capsules(
    first_motion.joint_names,
    first_motion.positions_m,
    first_randomization,
  )
  state = make_render_state(mesh_obj, first_fit, args.width, args.height)

  ffmpeg = (
    None
    if args.preview_only
    else _ffmpeg_process(args.output, args.width, args.height, args.fps)
  )
  group_counts: dict[str, int] = defaultdict(int)
  try:
    for sample_index, sample in enumerate(samples):
      if sample_index == 0:
        motion = first_motion
        window_start_s = first_window_start_s
        randomization = first_randomization
        fit = first_fit
      else:
        motion, window_start_s = _load_realtime_motion(
          args.dataset_root / sample.row["move_soma_uniform_path"],
          sample.group,
          args.seconds_per_clip,
          args.fps,
        )
        randomization = sample_capsule_domain_randomization(rng)
        fit = fit_soma_capsules(motion.joint_names, motion.positions_m, randomization)
      mesh_frames = [
        mesh.skin_frame_mujoco(motion, frame_index, randomization.body_scale_xyz)
        for frame_index in range(len(motion.positions_m))
      ]
      group_index = group_counts[sample.group]
      group_counts[sample.group] += 1
      for frame_index, vertices in enumerate(mesh_frames):
        frame = update_scene(state, mesh, vertices, fit, frame_index, sample.group)
        frame = annotate_frame(
          frame,
          sample,
          group_index,
          motion,
          fit,
          randomization,
          frame_index,
          window_start_s,
          args.fps,
        )
        if args.preview_only:
          preview_path = args.output.with_suffix(".preview.png")
          Image.fromarray(frame).save(preview_path)
          print(preview_path)
          return
        assert ffmpeg is not None and ffmpeg.stdin is not None
        ffmpeg.stdin.write(frame.tobytes())
      print(
        f"rendered {sample_index + 1:02d}/{len(samples)} "
        f"{sample.group}: {sample.row['move_name']}",
        flush=True,
      )
  finally:
    state.renderer.close()
    if ffmpeg is not None and ffmpeg.stdin is not None:
      ffmpeg.stdin.close()
      return_code = ffmpeg.wait()
      if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")

  print(args.output)


if __name__ == "__main__":
  main()
