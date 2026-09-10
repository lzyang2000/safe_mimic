#!/usr/bin/env python3
"""Record the sighted-vs-blind comparison through viser (meshed humans, LiDAR points).

Same scenario logic as ``render_avoidance_comparison.py`` (seeded batch, median
frontal encounter, identical humans and reference sequence in every panel),
but the frames come from the viser scene the play viewer uses, so the humans
are the skinned SOMA meshes and the LiDAR hit points are drawn. A headless
Chromium (playwright) connects as the render client and ``get_render`` pulls
one frame per recorded step with a robot-tracking camera.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viser.overlays import ViserDebugOverlays
from mjlab.viewer.viser.scene import MjlabViserScene

from safe_mimic.evaluation import (
  ENCOUNTER_PRESETS,
  apply_encounter_preset,
  bearing_deg,
  retarget_escape_moves,
)
from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  mdp,
)
from safe_mimic.tasks.env_cfg import (
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
)
from safe_mimic.video_tools import (
  blind_lidar,
  caption,
  ffmpeg_writer,
  hstack_videos,
  opencv_lookat_wxyz,
  orbit_camera_position,
  pick_median_env,
)

DEFAULT_MOTION = Path("artifacts/motions/lafan1_dance1_subject1_demo_motion.npz")


def build(
  args: argparse.Namespace, *, checkpoint: Path, task_id: str
) -> tuple[Any, Any, Any]:
  cfg = load_env_cfg(task_id, play=True)
  retarget_escape_moves(cfg, args.motion_file)
  cfg.commands["motion"].motion_file = str(args.motion_file)
  cfg.commands["motion"].sampling_mode = "start"
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.scene.env_spacing = 60.0
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
  crowd["show_mesh"] = True
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["show_mesh"] = True
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["print_velocity"] = False
  blind_task = task_id in (
    LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
    LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
  )
  for sensor in cfg.scene.sensors or ():
    # The blind actor never sees the LiDAR, so never draw its hit points either
    # (the sensor itself must stay: the critic's privileged term reads it).
    sensor.debug_vis = (
      bool(args.lidar_points) and sensor.name == "lidar_360" and not blind_task
    )
  for group in cfg.observations.values():
    group.enable_corruption = False
  if args.drop_tracking_termination:
    cfg.terminations.pop("ee_body_pos", None)
  raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  agent_cfg = load_rl_cfg(task_id)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task_id)(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
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


class ViserRecorder:
  """A viser server + headless Chromium client that renders frames on demand."""

  # Software GL (SwiftShader) is the safe default; ``--gpu`` swaps in EGL on the
  # machine's GPU, which is an order of magnitude faster when the driver allows.
  CHROMIUM_ARGS_SOFTWARE = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ]
  CHROMIUM_ARGS_GPU = [
    "--use-gl=angle",
    "--use-angle=gl-egl",
    "--enable-gpu",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
  ]
  chromium_args: list[str] = CHROMIUM_ARGS_SOFTWARE
  settle_s: float = 0.0
  _playwright: Any = None

  def __init__(
    self, raw_env: Any, *, env_idx: int, port: int, width: int, height: int
  ) -> None:
    sim = raw_env.sim
    self.server = viser.ViserServer(
      port=port, label="safe-mimic recorder", verbose=False
    )
    self.scene = MjlabViserScene(
      server=self.server,
      mj_model=sim.mj_model,
      num_envs=raw_env.num_envs,
      sim_model=sim.model,
      expanded_fields=sim.expanded_fields,
    )
    self.scene.env_idx = env_idx
    self.scene.debug_visualization_enabled = True
    # All batched worlds share the same origin in this task, so drawing every
    # world piles 64 robots and crowds on top of each other. Show only ours.
    self.scene.show_only_selected = True
    self.scene.show_all_envs = False
    self.overlays = ViserDebugOverlays(raw_env, self.scene)
    self.raw_env = raw_env
    self.width, self.height = width, height
    from playwright.sync_api import sync_playwright

    # One playwright driver per process: starting a second one after the first
    # was never stopped (teardown is deliberately non-graceful) raises.
    if ViserRecorder._playwright is None:
      ViserRecorder._playwright = sync_playwright().start()
    self._pw = ViserRecorder._playwright
    self._browser = self._pw.chromium.launch(
      headless=True, args=list(self.chromium_args)
    )
    self._page = self._browser.new_page(viewport={"width": width, "height": height})
    self._page.goto(f"http://localhost:{port}")
    deadline = time.time() + 60.0
    while not self.server.get_clients() and time.time() < deadline:
      time.sleep(0.2)
    clients = self.server.get_clients()
    if not clients:
      raise RuntimeError(
        "no viser client connected (headless Chromium failed to attach)"
      )
    self.client = next(iter(clients.values()))
    time.sleep(1.0)  # let the scene's meshes arrive at the client

  def frame(
    self, *, camera_pos: np.ndarray, camera_wxyz: np.ndarray, fov_deg: float
  ) -> np.ndarray:
    self.overlays.queue()
    with self.server.atomic():
      self.scene.update(self.raw_env.sim.data, self.scene.env_idx)
    self.server.flush()
    if self.settle_s > 0.0:
      time.sleep(self.settle_s)  # let the client apply the scene/overlay messages
    image = self.client.get_render(
      height=self.height,
      width=self.width,
      wxyz=camera_wxyz,
      position=camera_pos,
      fov=math.radians(fov_deg),
      transport_format="png",
      timeout=60.0,
    )
    return np.asarray(image)[..., :3]

  def close(self, timeout_s: float = 10.0) -> None:
    """Tear down without letting anything hang the recording.

    Graceful ``browser.close()`` / ``playwright.stop()`` have both been seen to
    spin forever after a long session, so the headless Chromium is killed by
    process name and viser's ``stop`` runs on a daemon thread with a timeout.
    """
    import subprocess
    import threading

    subprocess.run(["pkill", "-f", "chromium_headless_shel[l]"], check=False)
    worker = threading.Thread(target=self.server.stop, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
      print("[WARN] viser server stop did not finish; continuing without it")


def record(
  args: argparse.Namespace,
  *,
  env_idx: int,
  checkpoint: Path,
  task_id: str,
  blinded: bool,
  label: str,
  output: Path,
  port: int,
) -> dict[str, Any]:
  raw_env, env, policy = build(args, checkpoint=checkpoint, task_id=task_id)
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
  recorder = ViserRecorder(
    raw_env, env_idx=env_idx, port=port, width=args.width, height=args.height
  )
  fps = round(1.0 / raw_env.step_dt / args.frame_stride)
  writer = ffmpeg_writer(output, args.width, args.height, fps)
  steps = int(round(args.horizon_s / raw_env.step_dt))
  min_clearance = float("inf")
  terminations: list[dict[str, Any]] = []
  banner_until = -1.0
  smoothed_target: np.ndarray | None = None
  try:
    with torch.inference_mode():
      for step in range(steps):
        t = step * raw_env.step_dt
        clearance = mdp.capsule_link_surface_clearances(
          robot.data.body_link_pos_w[:, link_ids_t],
          human.data.geom_pos_w,
          human.data.geom_quat_w,
          raw_env.sim.model.geom_size[:, human_geoms],
          link_radius_m=command.cfg.link_filter.link_radius_m,
        ).amin(dim=(-2, -1))[env_idx]
        min_clearance = min(min_clearance, float(clearance))
        if step % args.frame_stride == 0:
          target = robot.data.root_link_pos_w[env_idx].cpu().numpy().astype(np.float64)
          target[2] = 0.9
          # Follow the pelvis with a short exponential lag so the camera glides
          # instead of stepping; snap after a reset (large jump).
          if smoothed_target is None or np.linalg.norm(target - smoothed_target) > 2.0:
            smoothed_target = target.copy()
          else:
            smoothed_target = 0.85 * smoothed_target + 0.15 * target
          target = smoothed_target
          eye = orbit_camera_position(
            target,
            distance=args.distance,
            azimuth_deg=args.azimuth,
            elevation_deg=args.elevation,
          )
          frame = recorder.frame(
            camera_pos=eye,
            camera_wxyz=opencv_lookat_wxyz(eye, target),
            fov_deg=args.fov_deg,
          )
          lines = [
            label,
            f"t = {t:5.2f} s   "
            f"nearest human-to-link clearance {float(clearance):5.2f} m",
            "encounter: "
            f"{scenario['speed_mps']:.2f} m/s from {scenario['distance_m']:.1f} m, "
            f"bearing {scenario['bearing_deg']:+.0f} deg",
          ]
          hits = sum("collision" in x["cause"] for x in terminations)
          lines.append(
            f"episodes ended so far: {len(terminations)}  (collisions: {hits})"
          )
          if t <= banner_until and terminations:
            lines.append(
              f"EPISODE ENDED: {terminations[-1]['cause']}  -> reset, next sequence"
            )
          scale = 1.0
          if args.minimal_captions:
            lines, scale = [label], args.title_scale
          writer.stdin.write(
            caption(frame, lines, scale=scale).tobytes()  # type: ignore[union-attr]
          )
        if blinded:
          saved = blind_lidar(observations)
          actions = policy(observations)
          observations["lidar"][..., :-1].copy_(saved)
        else:
          actions = policy(observations)
        observations, _, dones, _ = env.step(actions)
        if bool(dones[env_idx]):
          manager = raw_env.termination_manager
          causes = [
            n for n in manager.active_terms if bool(manager.get_term(n)[env_idx])
          ]
          terminations.append(
            {"t_s": t + raw_env.step_dt, "cause": ",".join(causes) or "reset"}
          )
          banner_until = t + 1.5
          smoothed_target = None  # snap the camera to the re-spawned robot
  finally:
    writer.stdin.close()  # type: ignore[union-attr]
    writer.wait()
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
  output.with_suffix(".json").write_text(json.dumps(result, indent=2))
  recorder.close()
  env.close()
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
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID,
    ),
  )
  parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
  parser.add_argument(
    "--encounter-preset", choices=tuple(ENCOUNTER_PRESETS), default="slow"
  )
  parser.add_argument("--seed", type=int, default=11)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument(
    "--env-idx", type=int, default=None, help="skip the median picker"
  )
  parser.add_argument("--max-bearing-deg", type=float, default=60.0)
  parser.add_argument("--horizon-s", type=float, default=25.0)
  parser.add_argument(
    "--frame-stride", type=int, default=2, help="record every n-th sim step"
  )
  parser.add_argument("--no-crowd", action="store_true")
  parser.add_argument("--force-crowd", action="store_true")
  parser.add_argument(
    "--dense-crowd",
    action="store_true",
    help="pack the outer crowd ring to capacity at EVERY reset (no density "
    "randomisation), so the scene stays the same kind of scene across episodes",
  )
  parser.add_argument("--lidar-points", action="store_true")
  parser.add_argument("--drop-tracking-termination", action="store_true")
  parser.add_argument("--baseline-checkpoint", type=Path, default=None)
  parser.add_argument("--baseline-task-id", default=None)
  parser.add_argument(
    "--baseline-label", default="blind-TRAINED baseline (never had LiDAR)"
  )
  parser.add_argument("--sighted-label", default="LiDAR on")
  parser.add_argument(
    "--minimal-captions",
    action="store_true",
    help="caption each panel with its label only (no clock, clearance, or tally)",
  )
  parser.add_argument(
    "--title-scale",
    type=float,
    default=5.0,
    help="Font/padding multiplier for the panel title drawn by --minimal-captions.",
  )
  parser.add_argument(
    "--eval-blind-panel", action="store_true", help="add the eval-blinded sighted panel"
  )
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=540)
  parser.add_argument("--distance", type=float, default=4.5)
  parser.add_argument("--azimuth", type=float, default=135.0)
  parser.add_argument("--elevation", type=float, default=-30.0)
  parser.add_argument("--fov-deg", type=float, default=50.0)
  parser.add_argument("--port", type=int, default=8123)
  parser.add_argument(
    "--gpu", action="store_true", help="render the viser client on the GPU (EGL)"
  )
  parser.add_argument(
    "--client-settle-s",
    type=float,
    default=0.1,
    help="pause between pushing the scene update and requesting the frame, so the "
    "client has applied the overlays (LiDAR points) before rendering",
  )
  parser.add_argument(
    "--reuse-lidar-on",
    type=Path,
    default=None,
    help="skip recording the sighted panel and use this existing video instead",
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, default=Path("artifacts/videos/viser"))
  args = parser.parse_args()
  configure_torch_backends()
  if args.gpu:
    ViserRecorder.chromium_args = ViserRecorder.CHROMIUM_ARGS_GPU
  ViserRecorder.settle_s = args.client_settle_s

  if args.env_idx is None:
    raw_env, env, _ = build(args, checkpoint=args.checkpoint, task_id=args.task_id)
    env.reset()
    table = scenario_table(raw_env)
    env.close()
    del env, raw_env
    torch.cuda.empty_cache()
    env_idx = pick_median_env(
      table, max_bearing_deg=args.max_bearing_deg, exclude=set()
    )
  else:
    env_idx = args.env_idx
  tag = f"seed{args.seed}_env{env_idx}"
  panels: list[dict[str, Any]] = []
  if args.reuse_lidar_on is not None:
    panels.append(
      {
        "output": str(args.reuse_lidar_on),
        "blinded": False,
        "outcome": "reused",
        "terminations": [],
        "min_clearance_m": float("nan"),
        "scenario": {},
      }
    )
  else:
    panels.append(
      record(
        args,
        env_idx=env_idx,
        checkpoint=args.checkpoint,
        task_id=args.task_id,
        blinded=False,
        label=args.sighted_label,
        output=args.output_dir / f"viser_{tag}_lidar_on.mp4",
        port=args.port,
      )
    )
  if args.eval_blind_panel:
    panels.append(
      record(
        args,
        env_idx=env_idx,
        checkpoint=args.checkpoint,
        task_id=args.task_id,
        blinded=True,
        label="LiDAR blinded: every ray reads 'no return'",
        output=args.output_dir / f"viser_{tag}_lidar_blind.mp4",
        port=args.port + 1,
      )
    )
  if args.baseline_checkpoint is not None:
    panels.append(
      record(
        args,
        env_idx=env_idx,
        checkpoint=args.baseline_checkpoint,
        task_id=args.baseline_task_id or args.task_id,
        blinded=False,
        label=args.baseline_label,
        output=args.output_dir / f"viser_{tag}_baseline.mp4",
        port=args.port + 2,
      )
    )
  for panel in panels:
    print(
      f"   {panel['output']}: {panel['outcome']} "
      f"(min clearance {panel['min_clearance_m']:.2f} m)"
    )
  side = args.output_dir / f"viser_{tag}_side_by_side.mp4"
  hstack_videos([p["output"] for p in panels], side)
  results = {
    "env_idx": env_idx,
    "seed": args.seed,
    "panels": panels,
    "side_by_side": str(side),
  }
  (args.output_dir / f"viser_{tag}.json").write_text(json.dumps(results, indent=2))
  print(json.dumps({k: v for k, v in results.items() if k != "panels"}, indent=2))


if __name__ == "__main__":
  main()
