#!/usr/bin/env python3
"""Render one encounter twice: the policy with LiDAR, and the same policy blinded.

A batch of environments is reset with a fixed seed, the environment whose
scheduled encounter is closest to the batch median (approach speed, spawn
distance) and comes from the front is picked, and that environment is
recorded offscreen with the trained policy. The environment is then rebuilt
with the same seed, which reproduces the identical human schedule, and
recorded again with the policy's LiDAR observation replaced by "no returns"
(every directional range at maximum, every range rate zero), exactly the
``blind`` mode of the avoidance benchmarks. Writes two MP4s and a
side-by-side.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import (  # noqa: E402
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from safe_mimic.evaluation import (  # noqa: E402
  ENCOUNTER_PRESETS,
  apply_encounter_preset,
  bearing_deg,
)
from safe_mimic.tasks import (  # noqa: E402
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  mdp,
)
from safe_mimic.tasks.env_cfg import (  # noqa: E402
  HUMAN_ENTITY_NAME,
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
)

DEFAULT_MOTION = Path("artifacts/motions/lafan1_dance1_subject1_demo_motion.npz")


def blind(observations: Any) -> Any:
  """Overwrite the LiDAR group in place with 'no returns'; caller restores."""
  lidar = observations["lidar"]
  directional = lidar[..., :-1]
  saved = directional.clone()
  cell_count = directional.shape[-1] // 2
  directional[..., :cell_count] = 1.0
  directional[..., cell_count:] = 0.0
  return saved


def build(args: argparse.Namespace, *, env_idx: int) -> tuple[Any, Any, Any]:
  cfg = load_env_cfg(args.task_id, play=True)
  cfg.commands["motion"].motion_file = str(args.motion_file)
  cfg.commands["motion"].sampling_mode = "start"
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.scene.env_spacing = 60.0  # keep neighbouring worlds out of the shot
  apply_encounter_preset(cfg, args.encounter_preset, keep_episode_length=True)
  cfg.episode_length_s = args.horizon_s + 5.0
  cfg.events.pop("push_robot", None)
  crowd = cfg.events[HUMAN_MOTION_EVENT_NAME].params
  if args.no_crowd:
    crowd.update(
      obstacle_free_probability=0.0, min_count=0, max_count=0, randomize_density=False
    )
  elif args.force_crowd or args.dense_crowd:
    crowd["obstacle_free_probability"] = 0.0
  if args.dense_crowd:
    crowd["randomize_density"] = False
    crowd["min_count"] = crowd["max_count"]
  if args.crowd_arc_spacing_m is not None:
    crowd["target_arc_spacing_m"] = args.crowd_arc_spacing_m
    crowd["randomize_density"] = False  # always packed to capacity
    crowd["min_count"] = crowd["max_count"]
  if args.drop_tracking_termination:
    cfg.terminations.pop("ee_body_pos", None)
  crowd["show_mesh"] = False
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["show_mesh"] = False
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["print_velocity"] = False
  for sensor in cfg.scene.sensors or ():
    sensor.debug_vis = bool(args.lidar_points) and sensor.name == "lidar_360"
  for group in cfg.observations.values():
    group.enable_corruption = False
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    env_idx=env_idx,
    max_extra_envs=0,
    width=args.width,
    height=args.height,
    distance=args.distance,
    azimuth=args.azimuth,
    elevation=args.elevation,
    geom_group=(1, 1, 1, 0, 0, 0),
  )
  raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
  # Human capsules live in the LiDAR group (3), which the renderer hides. Move
  # them to group 0 on the renderer's private model copy only.
  renderer = raw_env._offline_renderer  # noqa: SLF001
  assert renderer is not None
  for name in (PRIMARY_HUMAN_ENTITY_NAME, HUMAN_ENTITY_NAME):
    ids = raw_env.scene[name].indexing.geom_ids.cpu().numpy()
    renderer._model.geom_group[ids] = 0  # noqa: SLF001
  agent_cfg = load_rl_cfg(args.task_id)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task_id)(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)
  return raw_env, env, policy


def scenario_table(raw_env: Any) -> dict[str, torch.Tensor]:
  event = raw_env.event_manager.get_term_cfg(PRIMARY_HUMAN_EVENT_NAME).func
  sampler = event.sampler
  robot = raw_env.scene["robot"]
  now = float(raw_env.common_step_counter) * raw_env.step_dt
  ttc = sampler.global_intersection_times_s - now
  human_xy = sampler._poses.root_positions_w[:, :2]  # noqa: SLF001
  robot_xy = robot.data.root_link_pos_w[:, :2]
  distance = torch.linalg.vector_norm(human_xy - robot_xy, dim=-1)
  return {
    "ttc_s": ttc,
    "distance_m": distance,
    "speed_mps": distance / ttc.clamp_min(raw_env.step_dt),
    "bearing_deg": bearing_deg(robot_xy, robot.data.root_link_quat_w, human_xy),
  }


def pick_median_env(
  table: dict[str, torch.Tensor], *, max_bearing_deg: float, exclude: set[int]
) -> int:
  speed, dist, bearing = table["speed_mps"], table["distance_m"], table["bearing_deg"]
  # Envs whose human is not yet placed (distance ~0) or whose intercept is
  # already due (huge speed) are not real encounters; drop them before taking
  # medians, otherwise they wreck the spread used for scoring.
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


def caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image, "RGBA")
  pad, line_h = 10, 22
  draw.rectangle(
    (0, 0, image.width, pad * 2 + line_h * len(lines)), fill=(0, 0, 0, 150)
  )
  for k, text in enumerate(lines):
    draw.text((pad, pad + k * line_h), text, fill=(255, 255, 255, 255))
  return np.asarray(image)


def ffmpeg_writer(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
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


def record(
  args: argparse.Namespace,
  *,
  env_idx: int,
  blinded: bool,
  output: Path,
  checkpoint: Path | None = None,
  task_id: str | None = None,
  label_override: str | None = None,
) -> dict[str, Any]:
  if checkpoint is not None or task_id is not None:
    args = argparse.Namespace(**vars(args))
    args.checkpoint = checkpoint or args.checkpoint
    args.task_id = task_id or args.task_id
  raw_env, env, policy = build(args, env_idx=env_idx)
  observations, _ = env.reset()
  table = scenario_table(raw_env)
  scenario = {k: float(v[env_idx]) for k, v in table.items()}
  robot = raw_env.scene["robot"]
  human = raw_env.scene[PRIMARY_HUMAN_ENTITY_NAME]
  command = raw_env.command_manager.get_term("motion")
  link_names = tuple(command.cfg.link_filter.body_names)
  link_ids, _ = robot.find_bodies(link_names, preserve_order=True)
  link_ids_t = torch.tensor(link_ids, device=args.device)
  human_geoms = human.indexing.geom_ids.to(dtype=torch.long)
  fps = round(1.0 / raw_env.step_dt)
  writer = ffmpeg_writer(output, args.width, args.height, fps)
  label = "LiDAR blinded: every ray reads 'no return'" if blinded else "LiDAR on"
  if label_override is not None:
    label = label_override
  steps = int(round(args.horizon_s / raw_env.step_dt))
  min_clearance = float("inf")
  outcome = "survived"
  hold_frames = fps  # freeze the last frame for one second
  terminations: list[dict[str, Any]] = []
  banner_until = -1.0  # sim time until which the last termination banner shows
  with torch.inference_mode():
    for step in range(steps):
      clearance = mdp.capsule_link_surface_clearances(
        robot.data.body_link_pos_w[:, link_ids_t],
        human.data.geom_pos_w,
        human.data.geom_quat_w,
        raw_env.sim.model.geom_size[:, human_geoms],
        link_radius_m=command.cfg.link_filter.link_radius_m,
      ).amin(dim=(-2, -1))[env_idx]
      min_clearance = min(min_clearance, float(clearance))
      raw_frame = raw_env.render()
      frame = raw_frame
      t = step * raw_env.step_dt
      lines = [
        label,
        f"t = {t:5.2f} s   nearest human-to-link clearance {float(clearance):5.2f} m",
        "encounter: "
        f"{scenario['speed_mps']:.2f} m/s from {scenario['distance_m']:.1f} m, "
        f"bearing {scenario['bearing_deg']:+.0f} deg",
      ]
      if args.continue_after_termination:
        hits = sum("collision" in x["cause"] for x in terminations)
        lines.append(
          f"episodes ended so far: {len(terminations)}  (collisions: {hits})"
        )
        if t <= banner_until and terminations:
          lines.append(
            f"EPISODE ENDED: {terminations[-1]['cause']}  -> reset, next sequence"
          )
      frame = caption(frame, lines)
      writer.stdin.write(frame.tobytes())  # type: ignore[union-attr]
      if blinded:
        saved = blind(observations)
        actions = policy(observations)
        observations["lidar"][..., :-1].copy_(saved)
      else:
        actions = policy(observations)
      observations, _, dones, _ = env.step(actions)
      if bool(dones[env_idx]):
        manager = raw_env.termination_manager
        causes = [
          name for name in manager.active_terms if bool(manager.get_term(name)[env_idx])
        ]
        cause = ",".join(causes) or "reset"
        terminations.append({"t_s": t + raw_env.step_dt, "cause": cause})
        if args.continue_after_termination:
          banner_until = t + 1.5
          continue
        outcome = cause
        # The env has already reset by now; freeze the last pre-step frame.
        last = caption(
          raw_frame,
          [
            label,
            f"t = {t + raw_env.step_dt:5.2f} s   EPISODE ENDED: {cause}",
            f"minimum clearance {min_clearance:5.2f} m",
          ],
        )
        for _ in range(hold_frames):
          writer.stdin.write(last.tobytes())  # type: ignore[union-attr]
        break
  writer.stdin.close()  # type: ignore[union-attr]
  if writer.wait() != 0:
    raise RuntimeError("ffmpeg failed")
  if args.continue_after_termination:
    hits = sum("collision" in x["cause"] for x in terminations)
    outcome = (
      "survived"
      if not terminations
      else f"{len(terminations)} episodes ended ({hits} collision)"
    )
  result = {
    "output": str(output),
    "blinded": blinded,
    "outcome": outcome,
    "terminations": terminations,
    "min_clearance_m": min_clearance,
    "scenario": scenario,
  }
  env.close()
  del policy, env, raw_env
  torch.cuda.empty_cache()
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--task-id",
    default=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
    choices=(
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
    ),
  )
  parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
  parser.add_argument(
    "--encounter-preset", choices=tuple(ENCOUNTER_PRESETS), default="slow"
  )
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument(
    "--num-envs", type=int, default=64, help="batch to pick the median encounter from"
  )
  parser.add_argument("--max-bearing-deg", type=float, default=60.0)
  parser.add_argument("--horizon-s", type=float, default=9.0)
  parser.add_argument(
    "--max-tries",
    type=int,
    default=4,
    help="candidates to try until LiDAR-on survives and blinded collides",
  )
  parser.add_argument("--no-crowd", action="store_true")
  parser.add_argument(
    "--force-crowd",
    action="store_true",
    help="put a crowd in every environment (default: 25%% of envs are crowd-free)",
  )
  parser.add_argument(
    "--crowd-arc-spacing-m",
    type=float,
    default=None,
    help="pack the crowd ring at this arc spacing with no density randomisation "
    "(task default: 0.62 m spacing, occupancy randomised from 0 to packed)",
  )
  parser.add_argument(
    "--lidar-points",
    action="store_true",
    help="draw the LiDAR hit points (the sensor's debug markers) in the render",
  )
  parser.add_argument(
    "--continue-after-termination",
    action="store_true",
    help="do not stop recording when the tracked env terminates: flash the cause, let "
    "the env reset to its next sequence, and keep going for the full horizon",
  )
  parser.add_argument(
    "--dense-crowd",
    action="store_true",
    help="pack the outer crowd ring to capacity at every reset "
    "(no density randomisation)",
  )
  parser.add_argument(
    "--drop-tracking-termination",
    action="store_true",
    help="remove the ee_body_pos wrist/ankle height check for the render; collisions "
    "and falls still end the episode",
  )
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=540)
  parser.add_argument("--distance", type=float, default=4.5)
  parser.add_argument("--azimuth", type=float, default=135.0)
  parser.add_argument("--elevation", type=float, default=-25.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--baseline-checkpoint",
    type=Path,
    default=None,
    help="optional third panel: another checkpoint (e.g. the blind-trained baseline) "
    "run sighted on the same scenario",
  )
  parser.add_argument("--baseline-task-id", default=None)
  parser.add_argument("--baseline-label", default="blind-trained baseline")
  parser.add_argument(
    "--skip-eval-blind",
    action="store_true",
    help="omit the 'sighted policy with LiDAR blinded at evaluation' panel; compare "
    "the sighted policy only against --baseline-checkpoint",
  )
  parser.add_argument("--output-dir", type=Path, default=Path("artifacts/videos"))
  args = parser.parse_args()
  configure_torch_backends()

  # Pick the median frontal encounter from the seeded batch.
  raw_env, env, _ = build(args, env_idx=0)
  env.reset()
  table = scenario_table(raw_env)
  env.close()
  del env, raw_env
  torch.cuda.empty_cache()

  tried: set[int] = set()
  results = None
  for attempt in range(args.max_tries):
    env_idx = pick_median_env(
      table, max_bearing_deg=args.max_bearing_deg, exclude=tried
    )
    tried.add(env_idx)
    tag = f"seed{args.seed}_env{env_idx}"
    print(
      f"[attempt {attempt + 1}] env {env_idx}: "
      f"speed {float(table['speed_mps'][env_idx]):.2f} m/s, "
      f"distance {float(table['distance_m'][env_idx]):.2f} m, "
      f"bearing {float(table['bearing_deg'][env_idx]):+.0f} deg"
    )
    on = record(
      args,
      env_idx=env_idx,
      blinded=False,
      output=args.output_dir / f"avoidance_{tag}_lidar_on.mp4",
    )
    off = None
    if not args.skip_eval_blind:
      off = record(
        args,
        env_idx=env_idx,
        blinded=True,
        output=args.output_dir / f"avoidance_{tag}_lidar_blind.mp4",
      )
    print(f"   LiDAR on: {on['outcome']} (min clearance {on['min_clearance_m']:.2f} m)")
    if off is not None:
      print(
        f"   blinded: {off['outcome']} (min clearance {off['min_clearance_m']:.2f} m)"
      )
    results = {
      "lidar_on": on,
      "lidar_blind": off,
      "env_idx": env_idx,
      "seed": args.seed,
    }
    if args.baseline_checkpoint is not None:
      base = record(
        args,
        env_idx=env_idx,
        blinded=False,
        output=args.output_dir / f"avoidance_{tag}_baseline.mp4",
        checkpoint=args.baseline_checkpoint,
        task_id=args.baseline_task_id,
        label_override=args.baseline_label,
      )
      print(
        f"   baseline: {base['outcome']} "
        f"(min clearance {base['min_clearance_m']:.2f} m)"
      )
      results["baseline"] = base
    counterpart = results.get("baseline") or off
    if on["outcome"] == "survived" and (
      counterpart is None or "collision" in counterpart["outcome"]
    ):
      break
  assert results is not None
  side = (
    args.output_dir
    / f"avoidance_seed{args.seed}_env{results['env_idx']}_side_by_side.mp4"
  )
  inputs = [results["lidar_on"]["output"]]
  if results.get("lidar_blind") is not None:
    inputs.append(results["lidar_blind"]["output"])
  if "baseline" in results:
    inputs.append(results["baseline"]["output"])
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
    str(side),
  ]
  subprocess.run(cmd, check=True)
  results["side_by_side"] = str(side)
  (
    args.output_dir / f"avoidance_seed{args.seed}_env{results['env_idx']}.json"
  ).write_text(json.dumps(results, indent=2))
  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
