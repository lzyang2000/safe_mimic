#!/usr/bin/env python
"""Play a co-adjust checkpoint with the ADJUSTER's command drawn as the ghost.

mjlab's play ghost is posed from the motion command's properties, which in
the filtered tasks is the privileged teacher reference. The adjuster's
residuals only exist inside the actor network, so this script wraps the
inference policy: every step it reads the actor's own joint prediction, hands
``raw command + prediction`` to the command's ghost override, then acts.
Blue ghost = adjuster (what the deployed actor actually tracks); with
``--ghost both`` the teacher's green ghost is drawn too.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer

from safe_mimic.evaluation import ENCOUNTER_PRESETS, apply_encounter_preset
from safe_mimic.play_panel import SafeMimicPlayViewer, install_termination_printer
from safe_mimic.rl.perceptive_lidar import PerceptiveLidarActor
from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
)
from safe_mimic.tasks.env_cfg import PRIMARY_HUMAN_EVENT_NAME
from safe_mimic.tasks.kinematic_replay_command import PlanarFilteredReplayMotionCommand

DEFAULT_MOTION = Path("artifacts/motions/lafan1_dance1_subject1_demo_motion.npz")


class AdjusterGhostPolicy:
  """Inference policy wrapper that publishes the adjuster's joint command."""

  def __init__(
    self,
    policy: PerceptiveLidarActor,
    command: PlanarFilteredReplayMotionCommand,
    *,
    show_teacher: bool,
  ) -> None:
    self._policy = policy
    self._command = command
    self._show_teacher = show_teacher

  def __call__(self, obs) -> torch.Tensor:
    prediction = self._policy.predict_avoidance(obs)
    residual = prediction[:, self._policy.avoidance_planar_dim :]
    self._command.set_ghost_override(residual, show_teacher=self._show_teacher)
    return self._policy(obs)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--task-id",
    default=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
    choices=(
      LIDAR_AUXILIARY_COADJUST_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_FKC_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_FKC2_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
      LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
    ),
  )
  parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--ghost",
    choices=("adjuster", "both", "teacher"),
    default="adjuster",
    help="adjuster: blue adjuster ghost only; both: plus the teacher's green "
    "ghost; teacher: stock behavior.",
  )
  parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  parser.add_argument(
    "--encounter-preset",
    choices=("task", *ENCOUNTER_PRESETS),
    default="task",
    help="'task' keeps the task's own play encounters; 'standard' / 'slow' apply the "
    "evaluation presets' encounters (slow: spawn >= 1.8 m, approach <= 0.75 m/s). "
    "The episode stays effectively infinite in play either way.",
  )
  parser.add_argument(
    "--no-terminations",
    action="store_true",
    help="drop every termination (episodes never end)",
  )
  parser.add_argument(
    "--no-tracking-termination",
    action="store_true",
    help="drop only ee_body_pos (wrist/ankle height vs the corrected reference); "
    "collisions and falls still end the episode",
  )
  parser.add_argument(
    "--print-human-velocity",
    action="store_true",
    help="also print the periodic [primary-human] velocity lines (off: only "
    "[TERM] episode-end lines are printed while running)",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  cfg = load_env_cfg(args.task_id, play=True)
  cfg.commands["motion"].motion_file = str(args.motion_file)
  cfg.scene.num_envs = args.num_envs
  if args.encounter_preset != "task":
    # Encounters follow the preset; the episode stays effectively infinite so
    # play only resets on a failure termination (user direction 2026-09-06).
    apply_encounter_preset(cfg, args.encounter_preset, keep_episode_length=True)
  if args.no_terminations:
    cfg.terminations = {}
  elif args.no_tracking_termination:
    cfg.terminations.pop("ee_body_pos", None)
  # The play cfg turns on the walking human's periodic velocity print together
  # with its mesh; keep the run-time console to the [TERM] lines by default.
  cfg.events[PRIMARY_HUMAN_EVENT_NAME].params["print_velocity"] = bool(
    args.print_human_velocity
  )
  agent_cfg = load_rl_cfg(args.task_id)

  raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task_id)
  if runner_cls is None:
    raise RuntimeError("task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)
  if not isinstance(policy, PerceptiveLidarActor):
    raise RuntimeError("adjuster ghost requires the perceptive LiDAR actor")
  if not getattr(policy, "adjust_command_with_joint_prediction", False):
    raise RuntimeError("checkpoint is not a co-adjust (command-adjusting) actor")
  command = raw_env.command_manager.get_term("motion")
  if not isinstance(command, PlanarFilteredReplayMotionCommand):
    raise RuntimeError("task does not use the filtered replay command")

  if args.ghost == "teacher":
    viewer_policy = policy
  else:
    viewer_policy = AdjusterGhostPolicy(
      policy, command, show_teacher=args.ghost == "both"
    )
  print(
    f"[INFO] ghost mode: {args.ghost} (blue = adjuster command, "
    "green = privileged teacher reference)"
  )

  if not args.no_terminations:
    # Print why each episode ended (collision, wrist trip, fall, ...) with the
    # time since that env's last reset; works for both viewers.
    install_termination_printer(env, raw_env)

  resolved = args.viewer
  if resolved == "auto":
    import os

    resolved = (
      "native"
      if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      else "viser"
    )
  if resolved == "native":
    NativeMujocoViewer(env, viewer_policy).run()
  else:
    SafeMimicPlayViewer(env, viewer_policy).run()


if __name__ == "__main__":
  main()
