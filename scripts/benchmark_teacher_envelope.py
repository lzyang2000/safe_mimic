"""Measure the privileged reference filter's own collision envelope.

The robot is kinematically slaved to the filtered reference: the motion
command writes the CBF-filtered pose directly into the simulator each step,
with gravity, contacts, and actuation disabled. No policy or checkpoint is
involved. The resulting collision rate over the reaction-envelope encounter
distribution is the upper bound any student policy can reach by tracking the
teacher perfectly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends

from safe_mimic.tasks import mdp
from safe_mimic.tasks.env_cfg import (
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_ENTITY_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg,
)
from safe_mimic.tasks.kinematic_replay_command import (
  PlanarFilteredReplayMotionCommandCfg,
)

COLLISION_CLEARANCE_M = 0.1
PLANAR_SATURATION_FRACTION_OF_CAP = 0.97


def _bin_summary(
  values: torch.Tensor,
  collision: torch.Tensor,
  edges: tuple[float, ...],
  *,
  precision: int = 2,
) -> list[dict[str, float | int | str]]:
  rows: list[dict[str, float | int | str]] = []
  for lower, upper in zip(edges[:-1], edges[1:], strict=True):
    selected = torch.isfinite(values) & (values >= lower) & (values < upper)
    episodes = int(selected.count_nonzero())
    collisions = int((selected & collision).count_nonzero())
    upper_text = f"{upper:.{precision}f}" if math.isfinite(upper) else "inf"
    rows.append(
      {
        "range": f"[{lower:.{precision}f}, {upper_text})",
        "episodes": episodes,
        "collisions": collisions,
        "collision_rate": collisions / episodes if episodes else float("nan"),
      }
    )
  return rows


def _grid_summary(
  distance: torch.Tensor,
  speed: torch.Tensor,
  collision: torch.Tensor,
  distance_edges: tuple[float, ...],
  speed_edges: tuple[float, ...],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for distance_lower, distance_upper in zip(
    distance_edges[:-1], distance_edges[1:], strict=True
  ):
    distance_selected = (distance >= distance_lower) & (distance < distance_upper)
    cells = []
    for speed_lower, speed_upper in zip(speed_edges[:-1], speed_edges[1:], strict=True):
      selected = distance_selected & (speed >= speed_lower) & (speed < speed_upper)
      episodes = int(selected.count_nonzero())
      collisions = int((selected & collision).count_nonzero())
      cells.append(
        {
          "speed_range_mps": f"[{speed_lower:.2f}, {speed_upper:.2f})",
          "episodes": episodes,
          "collisions": collisions,
          "collision_rate": collisions / episodes if episodes else float("nan"),
        }
      )
    rows.append(
      {
        "distance_range_m": f"[{distance_lower:.2f}, {distance_upper:.2f})",
        "speed_cells": cells,
      }
    )
  return rows


def _print_table(title: str, rows: list[dict[str, float | int | str]]) -> None:
  print(f"\n{title}")
  print("range                 episodes  collisions  collision_rate")
  for row in rows:
    rate = float(row["collision_rate"])
    rate_text = f"{100.0 * rate:6.1f}%" if math.isfinite(rate) else "   n/a "
    print(
      f"{str(row['range']):<21} {int(row['episodes']):8d} "
      f"{int(row['collisions']):11d}  {rate_text}"
    )


def _minimum_clearance(
  raw_env: ManagerBasedRlEnv,
  *,
  robot_link_ids: torch.Tensor,
  human_geom_ids: torch.Tensor,
  link_radius_m: float,
) -> torch.Tensor:
  robot = raw_env.scene["robot"]
  human = raw_env.scene[PRIMARY_HUMAN_ENTITY_NAME]
  clearances = mdp.capsule_link_surface_clearances(
    robot.data.body_link_pos_w[:, robot_link_ids],
    human.data.geom_pos_w,
    human.data.geom_quat_w,
    raw_env.sim.model.geom_size[:, human_geom_ids],
    link_radius_m=link_radius_m,
  )
  return clearances.amin(dim=(-2, -1))


def _mean_and_p90(values: torch.Tensor) -> dict[str, float]:
  quantile = torch.quantile(values, torch.tensor(0.9, device=values.device))
  return {"mean": float(values.mean()), "p90": float(quantile)}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--seed", type=int, default=31)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--min-radius-m", type=float, default=0.75)
  parser.add_argument("--max-radius-m", type=float, default=4.0)
  parser.add_argument("--min-intercept-s", type=float, default=0.5)
  parser.add_argument("--max-intercept-s", type=float, default=4.0)
  parser.add_argument("--episode-length-s", type=float, default=6.0)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if not args.motion_file.is_file():
    parser.error(f"motion file not found: {args.motion_file}")
  if args.num_envs < 1:
    parser.error("--num-envs must be positive")
  configure_torch_backends()

  cfg = unitree_g1_lidar_auxiliary_avoidance_tracking_env_cfg(play=True)
  # Pin frame-0 starts for comparability with existing artifacts.
  cfg.commands["motion"].sampling_mode = "start"
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.episode_length_s = args.episode_length_s
  cfg.commands["motion"].motion_file = str(args.motion_file)
  cfg.events.pop("push_robot", None)
  crowd_event = cfg.events[HUMAN_MOTION_EVENT_NAME]
  crowd_event.params["obstacle_free_probability"] = 0.0
  crowd_event.params["min_count"] = 0
  crowd_event.params["max_count"] = 0
  crowd_event.params["randomize_density"] = False
  primary_event = cfg.events[PRIMARY_HUMAN_EVENT_NAME]
  primary_event.params["min_initial_spawn_radius_m"] = args.min_radius_m
  primary_event.params["max_initial_spawn_radius_m"] = args.max_radius_m
  primary_event.params["min_intersection_delay_s"] = args.min_intercept_s
  primary_event.params["max_intersection_delay_s"] = args.max_intercept_s
  primary_event.params["encounter_sampling"] = "independent"
  primary_event.params["show_mesh"] = False
  primary_event.params["print_velocity"] = False
  crowd_event.params["show_mesh"] = False
  for sensor in cfg.scene.sensors or ():
    sensor.debug_vis = False
  for group in cfg.observations.values():
    group.enable_corruption = False
  cfg.observations["lidar"].terms["directional_range_rate"].params["noise_cfg"] = None

  motion_cfg = cfg.commands["motion"]
  if not isinstance(motion_cfg, PlanarFilteredReplayMotionCommandCfg):
    raise TypeError("teacher envelope requires the filtered replay command")
  motion_cfg.write_reference_to_sim = True
  motion_cfg.align_reference_to_robot_each_step = False
  cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)
  cfg.sim.mujoco.disableflags = tuple(
    dict.fromkeys((*cfg.sim.mujoco.disableflags, "contact", "actuation"))
  )
  cfg.terminations.clear()

  raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  raw_env.reset()

  robot = raw_env.scene["robot"]
  human = raw_env.scene[PRIMARY_HUMAN_ENTITY_NAME]
  command = raw_env.command_manager.get_term("motion")
  link_names = tuple(command.cfg.link_filter.body_names)
  link_ids, resolved_names = robot.find_bodies(link_names, preserve_order=True)
  if tuple(resolved_names) != link_names:
    raise RuntimeError("robot link order mismatch")
  link_ids_t = torch.tensor(link_ids, dtype=torch.long, device=args.device)
  human_geom_ids = human.indexing.geom_ids.to(dtype=torch.long)
  event_term = raw_env.event_manager.get_term_cfg(PRIMARY_HUMAN_EVENT_NAME).func
  if not hasattr(event_term, "sampler"):
    raise RuntimeError("primary human event does not expose its sampler")
  sampler = event_term.sampler

  now_s = float(raw_env.common_step_counter) * raw_env.step_dt
  scheduled_ttc_s = sampler.global_intersection_times_s.clone() - now_s
  initial_human_root_w = sampler._poses.root_positions_w.clone()  # noqa: SLF001
  initial_robot_root_w = robot.data.root_link_pos_w.clone()
  initial_root_distance_m = torch.linalg.vector_norm(
    initial_human_root_w[:, :2] - initial_robot_root_w[:, :2], dim=-1
  )
  initial_clearance_m = _minimum_clearance(
    raw_env,
    robot_link_ids=link_ids_t,
    human_geom_ids=human_geom_ids,
    link_radius_m=command.cfg.link_filter.link_radius_m,
  )
  nominal_approach_speed_mps = initial_root_distance_m / scheduled_ttc_s.clamp_min(
    raw_env.step_dt
  )

  num_envs = args.num_envs
  minimum_clearance_m = initial_clearance_m.clone()
  first_collision_time_s = torch.full((num_envs,), torch.nan, device=args.device)
  planar_intervention_max_mps = torch.zeros(num_envs, device=args.device)
  planar_cbf_violation_max_mps = torch.zeros(num_envs, device=args.device)
  link_cbf_violation_max_mps = torch.zeros(num_envs, device=args.device)
  planar_saturated_steps = torch.zeros(num_envs, device=args.device)
  saturation_threshold_mps = (
    PLANAR_SATURATION_FRACTION_OF_CAP
    * motion_cfg.planar_filter.max_intervention_speed_mps
  )
  zero_actions = torch.zeros(
    (num_envs, raw_env.action_manager.total_action_dim), device=args.device
  )
  step_count = int(round(args.episode_length_s / raw_env.step_dt))

  def measure_clearance(elapsed_s: float) -> None:
    nonlocal minimum_clearance_m
    clearance = _minimum_clearance(
      raw_env,
      robot_link_ids=link_ids_t,
      human_geom_ids=human_geom_ids,
      link_radius_m=command.cfg.link_filter.link_radius_m,
    )
    minimum_clearance_m = torch.minimum(minimum_clearance_m, clearance)
    collided = torch.isnan(first_collision_time_s) & (clearance < COLLISION_CLEARANCE_M)
    first_collision_time_s[collided] = elapsed_s

  with torch.inference_mode():
    for step in range(step_count):
      measure_clearance(step * raw_env.step_dt)
      raw_env.step(zero_actions)
      planar_intervention = command.metrics["filter_intervention_speed_mps"]
      planar_intervention_max_mps = torch.maximum(
        planar_intervention_max_mps, planar_intervention
      )
      planar_saturated_steps += (
        planar_intervention >= saturation_threshold_mps
      ).float()
      planar_cbf_violation_max_mps = torch.maximum(
        planar_cbf_violation_max_mps,
        command.metrics["filter_cbf_violation_mps"],
      )
      link_cbf_violation_max_mps = torch.maximum(
        link_cbf_violation_max_mps,
        command.metrics["link_filter_cbf_violation_mps"],
      )
    measure_clearance(step_count * raw_env.step_dt)

  collision = minimum_clearance_m < COLLISION_CLEARANCE_M
  planar_saturation_fraction = planar_saturated_steps / float(step_count)

  tensors = {
    "scheduled_intercept_ttc_s": scheduled_ttc_s,
    "initial_root_distance_m": initial_root_distance_m,
    "initial_clearance_m": initial_clearance_m,
    "nominal_approach_speed_mps": nominal_approach_speed_mps,
    "collision": collision,
    "first_collision_time_s": first_collision_time_s,
    "minimum_clearance_m": minimum_clearance_m,
    "planar_intervention_max_mps": planar_intervention_max_mps,
    "planar_cbf_violation_max_mps": planar_cbf_violation_max_mps,
    "link_cbf_violation_max_mps": link_cbf_violation_max_mps,
    "planar_saturation_fraction": planar_saturation_fraction,
  }
  cpu = {name: tensor.cpu() for name, tensor in tensors.items()}
  collision_cpu = cpu["collision"]
  ttc_rows = _bin_summary(
    cpu["scheduled_intercept_ttc_s"],
    collision_cpu,
    (0.0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, float("inf")),
  )
  distance_rows = _bin_summary(
    cpu["initial_root_distance_m"],
    collision_cpu,
    (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, float("inf")),
  )
  speed_rows = _bin_summary(
    cpu["nominal_approach_speed_mps"],
    collision_cpu,
    (0.0, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, float("inf")),
  )
  clearance_rows = _bin_summary(
    cpu["initial_clearance_m"],
    collision_cpu,
    (-float("inf"), 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float("inf")),
  )
  distance_speed_grid = _grid_summary(
    cpu["initial_root_distance_m"],
    cpu["nominal_approach_speed_mps"],
    collision_cpu,
    (0.0, 1.5, 2.0, 2.5, 3.0, 3.5, float("inf")),
    (0.0, 0.75, 1.0, 1.5, 2.0, 3.0, float("inf")),
  )

  summary = {
    "episodes": num_envs,
    "collisions": int(collision_cpu.count_nonzero()),
    "collision_rate": float(collision_cpu.float().mean()),
    "collision_clearance_m": COLLISION_CLEARANCE_M,
    "mean_scheduled_ttc_s": float(cpu["scheduled_intercept_ttc_s"].mean()),
    "mean_initial_root_distance_m": float(cpu["initial_root_distance_m"].mean()),
    "mean_nominal_approach_speed_mps": float(cpu["nominal_approach_speed_mps"].mean()),
    "teacher_saturation": {
      "planar_intervention_max_mps": _mean_and_p90(cpu["planar_intervention_max_mps"]),
      "planar_cbf_violation_max_mps": _mean_and_p90(
        cpu["planar_cbf_violation_max_mps"]
      ),
      "link_cbf_violation_max_mps": _mean_and_p90(cpu["link_cbf_violation_max_mps"]),
      "planar_saturation_fraction": _mean_and_p90(cpu["planar_saturation_fraction"]),
      "planar_saturation_threshold_mps": float(saturation_threshold_mps),
    },
  }
  payload = {
    "motion_file": str(args.motion_file.resolve()),
    "seed": args.seed,
    "device": args.device,
    "configuration": {
      "num_envs": num_envs,
      "min_radius_m": args.min_radius_m,
      "max_radius_m": args.max_radius_m,
      "min_intercept_s": args.min_intercept_s,
      "max_intercept_s": args.max_intercept_s,
      "episode_length_s": args.episode_length_s,
      "human_runtime": "online",
      "crowd_count": 0,
      "collision_clearance_m": COLLISION_CLEARANCE_M,
      "robot_runtime": "kinematic-filtered-reference",
    },
    "summary": summary,
    "scheduled_ttc_bins": ttc_rows,
    "initial_distance_bins": distance_rows,
    "nominal_speed_bins": speed_rows,
    "initial_clearance_bins": clearance_rows,
    "distance_speed_grid": distance_speed_grid,
    "cases": [
      {
        name: bool(values[index])
        if values.dtype == torch.bool
        else None
        if not math.isfinite(float(values[index]))
        else float(values[index])
        for name, values in cpu.items()
      }
      for index in range(num_envs)
    ],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True))
  _print_table("Scheduled intercept TTC", ttc_rows)
  _print_table("Initial root distance", distance_rows)
  _print_table("Nominal average approach speed", speed_rows)
  _print_table("Initial robot-link clearance", clearance_rows)
  print(f"\nWROTE {args.output}")

  raw_env.close()
  gc.collect()
  if args.device.startswith("cuda"):
    torch.cuda.empty_cache()


if __name__ == "__main__":
  main()
