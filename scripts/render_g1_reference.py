#!/usr/bin/env python3
"""Render a G1 reference motion as an offscreen kinematic replay.

Accepts either a raw BONES-SEED G1 CSV or a 50 Hz tracker NPZ, so the same
command previews a dataset clip and a motion file a task already points at.
The robot is posed straight from the reference trajectory, so the video shows
exactly what the tracking command asks the policy to follow. No checkpoint is
involved; use ``play_motion_library.py`` to watch a trained policy instead.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from safe_mimic.motions.g1_dataset import load_g1_csv  # noqa: E402
from safe_mimic.motions.g1_tracker_npz import resample_g1_motion  # noqa: E402

TRAIL_MARKERS = 220


def load_reference(
  motion_path: Path, native_fps: float, target_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """Return root position, root quaternion, joint angles, and the frame rate.

  A tracker NPZ is already resampled and stores the pelvis as body zero, so it
  is replayed at its own rate; a CSV is interpolated onto the target grid.
  """
  if motion_path.suffix == ".npz":
    data = np.load(motion_path)
    return (
      data["body_pos_w"][:, 0].astype(np.float64),
      data["body_quat_w"][:, 0].astype(np.float64),
      data["joint_pos"].astype(np.float64),
      float(np.atleast_1d(data["fps"])[0]),
    )
  return resample_g1_motion(
    *load_g1_csv(motion_path), native_fps=native_fps, target_fps=target_fps
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "motion_path", type=Path, help="BONES-SEED G1 CSV or 50 Hz tracker NPZ."
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--label", default="", help="Caption; defaults to the stem.")
  parser.add_argument("--native-fps", type=float, default=120.0)
  parser.add_argument("--fps", type=int, default=50)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--azimuth", type=float, default=135.0)
  parser.add_argument(
    "--face",
    action="store_true",
    help="Derive the azimuth from the clip's mean root yaw so the dancer faces "
    "the camera. BONES-SEED canonicalises every clip to yaw -90 degrees, which a "
    "fixed world azimuth renders as dancing off to one side.",
  )
  parser.add_argument(
    "--face-offset-deg",
    type=float,
    default=0.0,
    help="Rotate off dead-front with --face; 30 gives a three-quarter view.",
  )
  parser.add_argument("--elevation", type=float, default=-12.0)
  parser.add_argument("--distance", type=float, default=2.5)
  parser.add_argument(
    "--orbit-deg",
    type=float,
    default=0.0,
    help="Total azimuth swept across the clip.",
  )
  parser.add_argument(
    "--track",
    action="store_true",
    help="Follow the root instead of holding a fixed lookat; for travelling clips.",
  )
  parser.add_argument(
    "--track-window-s",
    type=float,
    default=1.0,
    help="Smoothing window for the tracking camera.",
  )
  parser.add_argument(
    "--trail", action="store_true", help="Mark the root ground track."
  )
  parser.add_argument("--no-caption", action="store_true")
  return parser.parse_args()


def build_model(width: int, height: int, trail_markers: int) -> mujoco.MjModel:
  """Compile the G1 model into a lit scene with a scaled floor and trail dots."""
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.visual.global_.offwidth = width
  spec.visual.global_.offheight = height
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    width=300,
    height=300,
    rgb1=[0.16, 0.18, 0.21],
    rgb2=[0.22, 0.25, 0.29],
  )
  material = spec.add_material(name="grid", texrepeat=[16, 16], reflectance=0.05)
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[16.0, 16.0, 0.05],
    material="grid",
    contype=0,
    conaffinity=0,
  )
  directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  spec.worldbody.add_light(
    pos=[-2.0, -3.0, 5.0],
    dir=[0.3, 0.45, -1.0],
    type=directional,
    diffuse=[0.55, 0.55, 0.55],
  )
  spec.worldbody.add_light(
    pos=[3.0, 1.5, 4.0],
    dir=[-0.5, -0.2, -1.0],
    type=directional,
    diffuse=[0.35, 0.38, 0.45],
    castshadow=0,
  )
  for index in range(trail_markers):
    body = spec.worldbody.add_body(name=f"trail_{index}", mocap=True)
    body.add_geom(
      name=f"trail_geom_{index}",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=[0.012, 0.0, 0.0],
      rgba=[1.0, 0.72, 0.25, 0.9],
      contype=0,
      conaffinity=0,
    )
  return spec.compile()


def front_azimuth_deg(root_quat_wxyz: np.ndarray, offset_deg: float) -> float:
  """Return the camera azimuth that looks at the dancer's front.

  MuJoCo places a free camera opposite its azimuth, so viewing the face means
  sitting 180 degrees from the direction the root points.
  """
  yaw = Rotation.from_quat(np.roll(root_quat_wxyz, -1, axis=1)).as_euler("xyz")[:, 2]
  mean_yaw = np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean())
  return float(np.degrees(mean_yaw) + 180.0 + offset_deg)


def smooth_track(positions: np.ndarray, window: int) -> np.ndarray:
  """Box-filter the camera target so travelling clips do not jitter."""
  if window <= 1:
    return positions
  kernel = np.ones(window) / window
  padded = np.pad(positions, ((window, window), (0, 0)), mode="edge")
  return np.stack(
    [
      np.convolve(padded[:, axis], kernel, mode="same")[window:-window]
      for axis in (0, 1)
    ],
    axis=1,
  )


def ffmpeg_process(
  path: Path, width: int, height: int, fps: int
) -> subprocess.Popen[bytes]:
  command = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
    "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
  ]  # fmt: skip
  return subprocess.Popen(command, stdin=subprocess.PIPE)


def draw_caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image, "RGBA")
  try:
    font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
  except OSError:
    font = ImageFont.load_default()
  box_height = 14 + 26 * len(lines)
  draw.rectangle((16, 16, 16 + 700, 16 + box_height), fill=(8, 10, 14, 170))
  for index, line in enumerate(lines):
    draw.text((30, 24 + 26 * index), line, fill=(238, 240, 244, 255), font=font)
  return np.asarray(image)


def main() -> None:
  args = parse_args()
  root_pos, root_quat, joint_pos, fps = load_reference(
    args.motion_path, args.native_fps, float(args.fps)
  )
  frame_count = root_pos.shape[0]
  radius = np.linalg.norm(root_pos[:, :2] - root_pos[0, :2], axis=1)
  path_length = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1).sum()

  trail_stride = max(1, frame_count // TRAIL_MARKERS)
  trail_xy = root_pos[::trail_stride, :2] if args.trail else np.empty((0, 2))
  model = build_model(args.width, args.height, len(trail_xy))
  data = mujoco.MjData(model)
  for index, position in enumerate(trail_xy):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"trail_{index}")
    data.mocap_pos[model.body_mocapid[body_id]] = (position[0], position[1], 0.005)

  if args.track:
    lookat_xy = smooth_track(
      root_pos[:, :2], max(1, int(round(args.track_window_s * fps)))
    )
  else:
    lookat_xy = np.broadcast_to(np.median(root_pos[:, :2], axis=0), (frame_count, 2))

  azimuth = (
    front_azimuth_deg(root_quat, args.face_offset_deg) if args.face else args.azimuth
  )
  camera = mujoco.MjvCamera()
  camera.distance = args.distance
  camera.elevation = args.elevation
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False

  args.output.parent.mkdir(parents=True, exist_ok=True)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  ffmpeg = ffmpeg_process(args.output, args.width, args.height, round(fps))
  try:
    for frame_id in range(frame_count):
      data.qpos[:3] = root_pos[frame_id]
      data.qpos[3:7] = root_quat[frame_id]
      data.qpos[7:] = joint_pos[frame_id]
      mujoco.mj_forward(model, data)
      camera.lookat[:] = (*lookat_xy[frame_id], 0.85)
      camera.azimuth = azimuth + args.orbit_deg * frame_id / max(1, frame_count - 1)
      renderer.update_scene(data, camera=camera, scene_option=scene_option)
      frame = renderer.render()
      if not args.no_caption:
        frame = draw_caption(
          frame,
          [
            args.label or args.motion_path.stem,
            f"t = {frame_id / fps:5.2f} s / {(frame_count - 1) / fps:5.2f} s",
            f"root offset = {radius[frame_id]:5.2f} m   "
            f"(max {radius.max():.2f} m, path {path_length:.2f} m)",
          ],
        )
      assert ffmpeg.stdin is not None
      ffmpeg.stdin.write(frame.tobytes())
  finally:
    renderer.close()
    if ffmpeg.stdin is not None:
      ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    if return_code != 0:
      raise RuntimeError(f"ffmpeg exited with status {return_code}")
  print(f"wrote {args.output} ({frame_count} frames @ {fps:g} fps)")


if __name__ == "__main__":
  main()
