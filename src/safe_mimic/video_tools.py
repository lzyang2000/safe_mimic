"""Shared helpers for the comparison-video scripts (offscreen and viser)."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def blind_lidar(observations: Any) -> torch.Tensor:
  """Overwrite the actor's LiDAR group in place with 'no returns'.

  Directional ranges become 1.0 (maximum), range rates 0.0; the trailing scan
  age is left alone. Returns the saved directional slice so the caller can
  restore it after the policy call. Identical to the benchmarks' blind mode.
  """
  lidar = observations["lidar"]
  directional = lidar[..., :-1]
  saved = directional.clone()
  cell_count = directional.shape[-1] // 2
  directional[..., :cell_count] = 1.0
  directional[..., cell_count:] = 0.0
  return saved


def pick_median_env(
  table: dict[str, torch.Tensor], *, max_bearing_deg: float, exclude: set[int]
) -> int:
  """Pick the frontal encounter closest to the batch medians of speed and distance.

  Envs whose human is not yet placed (distance ~0) or whose intercept is
  already due (huge speed) are not real encounters and are dropped before the
  medians are taken.
  """
  speed, dist, bearing = table["speed_mps"], table["distance_m"], table["bearing_deg"]
  valid = (table["ttc_s"] > 0.5) & (dist > 1.0) & (speed > 0.1) & (speed < 5.0)
  frontal = valid & (bearing.abs() <= max_bearing_deg)
  for i in exclude:
    frontal[i] = False
  if not bool(frontal.any()):
    raise RuntimeError(
      "no frontal encounter in the batch; raise --num-envs or --max-bearing-deg"
    )
  med_speed, med_dist = speed[valid].median(), dist[valid].median()
  spread_speed = speed[valid].std().clamp_min(1e-6)
  spread_dist = dist[valid].std().clamp_min(1e-6)
  score = ((speed - med_speed) / spread_speed) ** 2 + (
    (dist - med_dist) / spread_dist
  ) ** 2
  score = torch.where(frontal, score, torch.full_like(score, float("inf")))
  return int(score.argmin())


def orbit_camera_position(
  target: np.ndarray, *, distance: float, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
  """Camera position on a sphere around ``target`` (MuJoCo-style angles).

  Azimuth is measured in the ground plane from +x toward +y; a NEGATIVE
  elevation puts the camera above the target looking down, as in MuJoCo.
  """
  az = math.radians(azimuth_deg)
  el = math.radians(elevation_deg)
  offset = np.array(
    [
      distance * math.cos(el) * math.cos(az),
      distance * math.cos(el) * math.sin(az),
      -distance * math.sin(el),
    ]
  )
  return np.asarray(target, dtype=np.float64) + offset


def opencv_lookat_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Orientation quaternion (w, x, y, z) of an OpenCV-convention camera.

  viser's camera looks along its +Z axis with -Y as up; this builds the
  rotation whose +Z points from ``eye`` to ``target`` and whose -Y is as close
  to world +Z as possible.
  """
  eye = np.asarray(eye, dtype=np.float64)
  target = np.asarray(target, dtype=np.float64)
  forward = target - eye
  forward /= np.linalg.norm(forward)
  world_up = np.array([0.0, 0.0, 1.0])
  if abs(np.dot(forward, world_up)) > 0.999:
    world_up = np.array([0.0, 1.0, 0.0])
  right = np.cross(forward, world_up)
  right /= np.linalg.norm(right)
  down = np.cross(forward, right)  # camera +Y points down in OpenCV
  rot = np.stack([right, down, forward], axis=1)  # columns = camera axes in world
  # Rotation matrix -> quaternion (w, x, y, z).
  trace = np.trace(rot)
  if trace > 0:
    s = math.sqrt(trace + 1.0) * 2
    w = 0.25 * s
    x = (rot[2, 1] - rot[1, 2]) / s
    y = (rot[0, 2] - rot[2, 0]) / s
    z = (rot[1, 0] - rot[0, 1]) / s
  else:
    i = int(np.argmax(np.diag(rot)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + rot[i, i] - rot[j, j] - rot[k, k])) * 2
    q = np.zeros(4)
    q[i + 1] = 0.25 * s
    q[0] = (rot[k, j] - rot[j, k]) / s
    q[j + 1] = (rot[j, i] + rot[i, j]) / s
    q[k + 1] = (rot[k, i] + rot[i, k]) / s
    w, x, y, z = q
  q = np.array([w, x, y, z])
  return q / np.linalg.norm(q)


CAPTION_BASE_PX = 11
"""Pixel size of the default caption font; ``scale`` multiplies it."""


def caption_font(scale: float) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
  """A bold TrueType font ``scale`` times the default caption size (bitmap fallback)."""
  size = max(1, round(CAPTION_BASE_PX * scale))
  for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
    try:
      return ImageFont.truetype(name, size)
    except OSError:
      continue
  try:
    return ImageFont.load_default(size=size)
  except TypeError:  # pragma: no cover - very old Pillow without ``size``
    return ImageFont.load_default()


def caption(frame: np.ndarray, lines: list[str], *, scale: float = 1.0) -> np.ndarray:
  """Burn a translucent caption block with ``lines`` into the top of ``frame``.

  ``scale`` enlarges the text, padding and line height together (``5`` makes a
  panel title about five times taller than the default caption).
  """
  image = Image.fromarray(np.ascontiguousarray(frame[..., :3]))
  draw = ImageDraw.Draw(image, "RGBA")
  font = caption_font(scale)
  # Padding and line height grow slower than the glyphs so a big title does not
  # swallow the frame: scale 5 gives a 55 px font in a ~116 px band.
  pad, line_h = round(10 * math.sqrt(scale)), round(max(22.0, 22 * scale * 0.65))
  draw.rectangle(
    (0, 0, image.width, pad * 2 + line_h * len(lines)), fill=(0, 0, 0, 150)
  )
  for k, text in enumerate(lines):
    draw.text((pad, pad + k * line_h), text, fill=(255, 255, 255, 255), font=font)
  return np.asarray(image)


def ffmpeg_writer(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
  """Open an ffmpeg process that encodes raw RGB frames from stdin to H.264."""
  path.parent.mkdir(parents=True, exist_ok=True)
  return subprocess.Popen(
    [
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
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-crf",
      "18",
      str(path),
    ],
    stdin=subprocess.PIPE,
  )


def hstack_videos(inputs: list[str], output: Path) -> None:
  """Place videos side by side (shortest input padded by ffmpeg's hstack)."""
  if len(inputs) == 1:
    subprocess.run(
      [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        inputs[0],
        "-c",
        "copy",
        str(output),
      ],
      check=True,
    )
    return
  streams = "".join(f"[{i}:v]" for i in range(len(inputs)))
  cmd = ["ffmpeg", "-y", "-loglevel", "error"]
  for path in inputs:
    cmd += ["-i", path]
  cmd += [
    "-filter_complex",
    f"{streams}hstack=inputs={len(inputs)}:shortest=0",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "18",
    str(output),
  ]
  subprocess.run(cmd, check=True)


__all__ = [
  "blind_lidar",
  "caption",
  "ffmpeg_writer",
  "hstack_videos",
  "opencv_lookat_wxyz",
  "orbit_camera_position",
  "pick_median_env",
]
