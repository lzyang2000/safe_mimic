"""Root-tracking diagnostic for the unified filtered-reference tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from safe_mimic.evaluation import ENCOUNTER_PRESETS, apply_encounter_preset
from safe_mimic.tasks import (
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID,
  LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
  UNIFIED_ROOT_LEAD_M,
)
from safe_mimic.tasks.env_cfg import (
  DEFAULT_G1_BALLET_MANIFEST,
  DEFAULT_G1_BALLET_MIRROR_MANIFEST,
  HUMAN_MOTION_EVENT_NAME,
  PRIMARY_HUMAN_EVENT_NAME,
  unitree_g1_lidar_unified_reference_tracking_env_cfg,
)
from safe_mimic.tasks.reference_filter import planar_capsule_geometry

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--task-id", default=LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_TASK_ID)
p.add_argument("--motion-file", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=512)
p.add_argument("--seed", type=int, default=31)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--device", default="cuda:0")
p.add_argument(
  "--encounter-preset", choices=tuple(ENCOUNTER_PRESETS), default="standard"
)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
configure_torch_backends()
ballet_slow = (
  a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_SLOW_TASK_ID
)
ballet_lag = (
  a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_LAG_TASK_ID
)
ballet_blind = (
  a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_TASK_ID
)
ballet_blind_nominal = (
  a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOMINAL_TASK_ID
)
ballet_blind_nohumans = (
  a.task_id
  == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_BLIND_NOHUMANS_TASK_ID
)
ballet_moves = (
  a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID
)
ballet = (
  ballet_slow
  or ballet_lag
  or ballet_blind
  or ballet_blind_nominal
  or ballet_blind_nohumans
  or ballet_moves
  or a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID
)
dense = a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_DENSE_TASK_ID
slow = (
  dense
  or ballet_slow
  or a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_SLOW_TASK_ID
)
leash = (
  slow or ballet or a.task_id == LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_TASK_ID
)
cfg = unitree_g1_lidar_unified_reference_tracking_env_cfg(
  play=True,
  active_joint_reward=a.task_id != LIDAR_AUXILIARY_COADJUST_UNIFIED_TASK_ID,
  root_lead_m=UNIFIED_ROOT_LEAD_M if leash else None,
  planar_filter_at_robot_root=leash,
  slow_regime=slow,
  dense_encounters=dense,
  lag_aware_ee_termination=ballet_lag,
  blind_actor=ballet_blind,
  motion_manifest=(
    str(DEFAULT_G1_BALLET_MIRROR_MANIFEST)
    if ballet_moves
    else str(DEFAULT_G1_BALLET_MANIFEST)
    if ballet
    else None
  ),
  escape_moves=ballet_moves,
  filter_gated_ee_termination=slow and not dense,
)
cfg.commands["motion"].sampling_mode = "start"
cfg.seed = a.seed
cfg.scene.num_envs = a.num_envs
cfg.episode_length_s = 6.0
cfg.commands["motion"].motion_file = str(a.motion_file)
cfg.events.pop("push_robot", None)
ce = cfg.events[HUMAN_MOTION_EVENT_NAME]
ce.params.update(
  obstacle_free_probability=0.0,
  min_count=0,
  max_count=0,
  randomize_density=False,
  show_mesh=False,
)
pe = cfg.events[PRIMARY_HUMAN_EVENT_NAME]
pe.params.update(
  min_initial_spawn_radius_m=0.75,
  max_initial_spawn_radius_m=4.0,
  min_intersection_delay_s=0.5,
  max_intersection_delay_s=4.0,
  show_mesh=False,
  print_velocity=False,
)
if a.encounter_preset != "standard":
  apply_encounter_preset(cfg, a.encounter_preset)
  if a.steps == 300:
    a.steps = 500  # cover the 10 s episodes of the slow preset
for s in cfg.scene.sensors or ():
  s.debug_vis = False
for g in cfg.observations.values():
  g.enable_corruption = False
cfg.observations["lidar"].terms["directional_range_rate"].params["noise_cfg"] = None
raw_env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
agent_cfg = load_rl_cfg(a.task_id)
env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(a.task_id)(env, asdict(agent_cfg), device=a.device)
runner.load(
  str(a.checkpoint), load_cfg={"actor": True}, strict=True, map_location=a.device
)
policy = runner.get_inference_policy(device=a.device)
obs, _ = env.reset()
robot = raw_env.scene["robot"]
command = raw_env.command_manager.get_term("motion")
dev = a.device
N = a.num_envs
pf = command.cfg.planar_filter
body_names = tuple(command.cfg.body_names)
ankle_ids = torch.tensor(
  [body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")],
  device=dev,
)
wrist_ids = torch.tensor(
  [body_names.index(n) for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")],
  device=dev,
)
edges = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 1e9], device=dev)
nb = len(edges) - 1


def z(*s):
  return torch.zeros(s, device=dev, dtype=torch.float64)


cnt = z(nb)
rpos = z(nb)
grad = z(nb)
vtoward = z(nb)
vfilt_toward = z(nb)
speed = z(nb)
ankle_err = z(nb)
wrist_err = z(nb)
bodypos_rew = z(nb)
esc = z(nb)
esc_cnt = z(nb)
refspeed = z(nb)
danger = z(1)
danger_ghost_safe = z(1)
danger_ghost_inactive = z(1)
danger_recovering = z(1)
total = z(1)
coll_offset_sum = z(1)
coll_cnt = z(1)
coll_bins = z(nb)
term_names = list(raw_env.termination_manager.active_terms)
step_age = torch.zeros(N, device=dev, dtype=torch.long)
with torch.inference_mode():
  for _step in range(a.steps):
    valid = step_age > 0
    anchor_t = command.anchor_pos_w
    anchor_r = command.robot_anchor_pos_w
    e3 = anchor_t - anchor_r
    off = e3[:, :2]
    offn = off.norm(dim=-1)
    r = torch.exp(-(e3.square().sum(-1)) / 0.3**2)
    g = r * 2.0 * e3.norm(dim=-1) / 0.3**2
    v_filt = command.filtered_root_velocity_xy_w
    v_raw = command._raw_body_lin_vel_w()[:, 0, :2]
    v_rob = robot.data.root_link_lin_vel_w[:, :2]
    u = off / offn.clamp_min(1e-6)[:, None]
    vt = (v_rob * u).sum(-1)
    vft = (v_filt * u).sum(-1)
    d = v_filt - v_raw
    active = d.norm(dim=-1) >= 0.1
    er = ((v_rob - v_raw) * d).sum(-1) / d.square().sum(-1).clamp_min(1e-8)
    bp = command.body_pos_relative_w
    rbp = command.robot_body_pos_w
    ae = (bp[:, ankle_ids] - rbp[:, ankle_ids]).norm(dim=-1).mean(-1)
    we = (bp[:, wrist_ids] - rbp[:, wrist_ids]).norm(dim=-1).mean(-1)
    bpr = torch.exp(-((bp - rbp).square().sum(-1)).mean(-1) / 0.09)
    # clearance of ghost vs robot to primary human
    centers, quats, sizes = command._obstacle_tensors()
    ghost_pos = anchor_r.clone()
    ghost_pos[:, :2] = command._filtered_root_xy_w
    _, c_ghost, act_g = planar_capsule_geometry(
      ghost_pos,
      centers,
      quats,
      sizes,
      robot_radius_m=pf.robot_radius_m,
      vertical_gate_m=pf.vertical_gate_m,
    )
    _, c_rob, act_r = planar_capsule_geometry(
      robot.data.root_link_pos_w,
      centers,
      quats,
      sizes,
      robot_radius_m=pf.robot_radius_m,
      vertical_gate_m=pf.vertical_gate_m,
    )
    c_ghost = torch.where(act_g, c_ghost, torch.full_like(c_ghost, 1e9)).amin(-1)
    c_rob = torch.where(act_r, c_rob, torch.full_like(c_rob, 1e9)).amin(-1)
    in_danger = valid & (c_rob < pf.safe_clearance_m)
    recovering = ((v_filt - v_raw) * u).sum(
      -1
    ) < -0.05  # reference heading back toward robot
    b = torch.bucketize(offn, edges, right=True) - 1
    for i in range(nb):
      m = valid & (b == i)
      mf = m.double()
      cnt[i] += mf.sum()
      rpos[i] += (r * mf).sum()
      grad[i] += (g * mf).sum()
      vtoward[i] += (vt * mf).sum()
      vfilt_toward[i] += (vft * mf).sum()
      speed[i] += (v_rob.norm(dim=-1) * mf).sum()
      ankle_err[i] += (ae * mf).sum()
      wrist_err[i] += (we * mf).sum()
      bodypos_rew[i] += (bpr * mf).sum()
      refspeed[i] += (v_filt.norm(dim=-1) * mf).sum()
      ma = (m & active).double()
      esc[i] += (er * ma).sum()
      esc_cnt[i] += ma.sum()
    total += valid.double().sum()
    danger += in_danger.double().sum()
    danger_ghost_safe += (in_danger & (c_ghost >= pf.safe_clearance_m)).double().sum()
    danger_ghost_inactive += (
      (in_danger & (c_ghost >= pf.activation_clearance_m)).double().sum()
    )
    danger_recovering += (in_danger & recovering).double().sum()
    actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    coll = raw_env.termination_manager.get_term("primary_human_collision") & valid
    coll_cnt += coll.double().sum()
    coll_offset_sum += (offn * coll.double()).sum()
    for i in range(nb):
      coll_bins[i] += (coll & (b == i)).double().sum()
    step_age = torch.where(dones.bool(), torch.zeros_like(step_age), step_age + 1)


def L(t):
  return [round(float(x), 4) for x in t]


out = {
  "checkpoint": str(a.checkpoint),
  "num_envs": N,
  "steps": a.steps,
  "encounter_preset": a.encounter_preset,
  "valid_frames": float(total),
  "offset_bins_m": [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, "inf"],
  "frame_fraction": L(cnt / total),
  "root_pos_reward_mean": L(rpos / cnt.clamp_min(1)),
  "root_pos_reward_grad_per_m": L(grad / cnt.clamp_min(1)),
  "robot_speed_toward_ghost_mps": L(vtoward / cnt.clamp_min(1)),
  "filtered_speed_toward_ghost_dir_mps": L(vfilt_toward / cnt.clamp_min(1)),
  "robot_speed_mps": L(speed / cnt.clamp_min(1)),
  "filtered_ref_speed_mps": L(refspeed / cnt.clamp_min(1)),
  "ankle_relpos_err_m": L(ankle_err / cnt.clamp_min(1)),
  "wrist_relpos_err_m": L(wrist_err / cnt.clamp_min(1)),
  "body_pos_reward_mean": L(bodypos_rew / cnt.clamp_min(1)),
  "escape_ratio_active": L(esc / esc_cnt.clamp_min(1)),
  "active_frames_per_bin": L(esc_cnt),
  "danger": {
    "frames": float(danger),
    "frac_of_valid": float(danger / total),
    "ghost_clear_ge_safe_frac": float(danger_ghost_safe / danger.clamp_min(1)),
    "ghost_clear_ge_activation_frac": float(
      danger_ghost_inactive / danger.clamp_min(1)
    ),
    "reference_recovering_toward_robot_frac": float(
      danger_recovering / danger.clamp_min(1)
    ),
  },
  "collisions": {
    "count": float(coll_cnt),
    "mean_offset_at_collision_m": float(coll_offset_sum / coll_cnt.clamp_min(1)),
    "offset_bin_counts": L(coll_bins),
  },
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
