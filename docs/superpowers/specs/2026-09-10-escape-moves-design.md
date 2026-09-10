# Escape moves: avoid people with ballet steps instead of a shuffle

Date: 2026-09-10. Branch: `escape-moves`. Base: main `8af4e69`.

## Goal

Today the privileged planar CBF filter escapes an approaching human by
displacing the reference root sideways at up to ~1 m/s while the current clip
keeps playing. The robot therefore "shuffles" out of the way. The ballet
library already contains travelling steps (glissades, chaînés, ballet turns,
travelling turns) that move the root 0.3 m in about 0.25 s when entered at
their travel phase, at peak speeds of 1.2-1.6 m/s. This feature lets the
reference generator answer a predicted collision by switching the RAW clip to
a travelling ballet move whose travel direction matches the escape direction,
then resume the interrupted clip where it left off. The CBF filter stays on
top as the fallback, so safety is unchanged when a move is not enough.

Approach A from the design discussion: the move switch lives in the deployed
reference generator and is driven by the same planar signal the actor's
avoidance head already predicts. No new network head, no discrete loss, and
the actor keeps seeing no privileged information.

Non-goals: heading offsets at entry (the mirrored library gives directional
coverage instead), choreography-aware phrase boundaries, multi-move plans,
changes to rewards or terminations.

## Data: mirrored library

`src/safe_mimic/motions/mirror_g1.py`

- `G1MirrorSpec.from_model(model)` derives from the G1 MuJoCo model
  (`mjlab` `G1_XML`): the joint permutation (`left_*` <-> `right_*`, others
  fixed), the per-joint sign (+1 for hinge axes along body y, i.e. pitch; -1
  for roll/yaw axes), and the body permutation (`left_*_link` <->
  `right_*_link`, others fixed). Joint order = model hinge order, body order =
  model bodies minus `world`, which is the tracker-NPZ convention.
- `mirror_motion_arrays(arrays, spec)` reflects a tracker NPZ across the
  sagittal (x-z) plane: `joint_pos/joint_vel[:, perm] * sign`; body arrays
  re-indexed by the body permutation; positions `y -> -y`; quaternions
  `(w, x, y, z) -> (w, -x, y, -z)`; linear velocity `y -> -y`; angular
  velocity `(wx, wy, wz) -> (-wx, wy, -wz)`. `fps` untouched.
- Properties tested: mirroring twice is the identity; MuJoCo forward
  kinematics of the mirrored root pose and joint angles reproduces the
  mirrored body cloud to within 5 mm on a real ballet clip (and the original
  clip reproduces its own cloud, which pins the joint/body ordering).

`scripts/mirror_g1_motion_library.py --manifest <trimmed ballet.yaml>
--output-root <dir>/npz_50hz --output-manifest <dir>/ballet.yaml`
copies every original NPZ and writes `<stem>__mirror.npz` beside it, then
writes a manifest with both entries (mirrors inherit weight, split and
metadata plus `mirrored: true`). Output:
`artifacts/bones-seed/datasets/g1_ballet_v1_trim1s_mirror/` (696 clips).
Existing dataset directories are never modified.

## Travel index

`src/safe_mimic/motions/escape_move_index.py`

For each clip: `root_xy = body_pos_w[:, 0, :2]`, anchor yaw from the anchor
body's quaternion (`torso_link`, the replay command's anchor).

- Fastest window: `W = round(0.6 s * fps)` frames; `i* = argmax_i
  |root_xy[i+W] - root_xy[i]|`. `travel_m = |disp(i*)|`.
- `candidate = travel_m >= 0.3`.
- `entry_frame = max(0, i* - round(0.1 s * fps))` (a short lead so the step
  reads as a step).
- `exit_frame`: first frame after `i* + W` where root planar speed drops below
  0.3 m/s, clamped to `clip_len - 1` (strictly inside the clip so the planner's
  resume check runs before the clip-end chaining); never earlier than `i* + W`
  unless the clamp forces it.
- `direction_b`: `disp(i*)` rotated by `-yaw(anchor at entry_frame)`,
  normalized. `speed_mps = travel_m / 0.6 s`.

`scripts/build_escape_move_index.py --manifest <mirror ballet.yaml> --output
<dir>/escape_moves.json` writes
`{"fps", "anchor_body", "window_s", "clips": [{"file", "entry_frame",
"exit_frame", "direction_b", "speed_mps", "travel_m", "candidate"}]}` in
manifest order. Tests use a synthetic clip with a known travel segment.

## Planner

`src/safe_mimic/tasks/escape_moves.py` holds the pure logic; the filtered
replay command owns the per-env state and calls it.

### Config

`EscapeMoveCfg` (dataclass, attached as
`PlanarFilteredReplayMotionCommandCfg.escape_moves: EscapeMoveCfg | None =
None`; `None` = feature off, byte-identical behaviour):

| field | default | meaning |
|---|---|---|
| `index_file` | required | travel index JSON for the command's manifest |
| `trigger_speed_mps` | 0.25 | planar trigger norm threshold |
| `trigger_steps` | 3 | consecutive steps above threshold before switching |
| `min_alignment` | 0.7 | minimum cos(move direction, escape direction) |
| `speed_cap_mps` | 1.0 | speed term saturates here |
| `pose_distance_weight` | 0.5 | lambda on RMS joint distance at entry (rad) |
| `blend_s` | 0.3 | linear blend duration at entry and at resume |
| `cooldown_s` | 1.0 | no new trigger after a resume |
| `trigger_source` | `"teacher"` | `"teacher"` or `"actor"` |

### Signals

- Teacher trigger vector (world xy): the planar filter's
  `result.intervention_w` from the PREVIOUS step (stored on the command).
  Escape direction in the body frame = that vector rotated by the inverse
  robot anchor yaw.
- Actor trigger vector: `command.set_actor_escape_hint(hint_b)` receives the
  actor's 2-D planar prediction (body frame, already `filtered - raw` planar
  velocity by supervision) each step from the play/eval loop. Same threshold.

### State machine (per env)

`mode` in {NOMINAL, ESCAPING}; `trigger_count`; `cooldown_steps`;
`saved_start, saved_end, saved_frame`; `exit_frame`; `blend_frame` (global
frame index of the frozen pre-switch pose, -1 = none); `blend_steps_left`.

Each `_update_command`, after `_advance_time_steps` and before the raw
quantities are read:

1. NOMINAL: `trigger_count = trigger_count + 1 if |v| >= thr else 0`. If
   `trigger_count >= trigger_steps` and `cooldown_steps == 0`: select a move
   (below). If one qualifies: save `(clip_start, clip_end, time_steps)`, set
   `clip_start/end` to the move clip's bounds, `time_steps = entry_frame`,
   `exit_frame`, `blend_frame = saved_frame`, `blend_steps_left =
   blend_steps`, `mode = ESCAPING`, `trigger_count = 0`. If none qualifies,
   reset `trigger_count` and wait (the CBF keeps handling it).
2. ESCAPING: if `time_steps >= exit_frame`: restore the saved bounds,
   `time_steps = saved_frame`, `blend_frame = exit_frame - 1`,
   `blend_steps_left = blend_steps`, `mode = NOMINAL`, `cooldown_steps =
   cooldown`. (`_advance_time_steps` chaining cannot fire inside a move
   because `exit_frame < clip end`, so the resume check always wins.)
3. Decrement `cooldown_steps` and `blend_steps_left`; clear `blend_frame`
   when the blend ends.

Env reset (`_resample_command`) clears all planner state for those envs.

### Selection

Candidates: index rows with `candidate = True`. For env `e` with body-frame
escape direction `d`:

`score_c = cos(direction_b_c, d) * min(speed_c, speed_cap) - lambda *
RMS(entry_joint_pos_c - raw_joint_pos_e)`, over candidates with
`cos >= min_alignment`; pick argmax. `entry_joint_pos` is gathered from the
library at each candidate's global entry frame once at construction.
Manifest rows map to library clip ids by resolved NPZ path.

### Blend

Both raw getters (`_raw_body_pos_w/quat/lin_vel/ang_vel`, `_raw_joint_pos/
vel`) return `lerp(aligned(blend_frame), aligned(time_steps), alpha)` for envs
with an active blend, `alpha = 1 - blend_steps_left / blend_steps`. Each frame
is aligned to the robot independently (its own yaw delta and the robot's root
xy), so the blend is a pose interpolation in the robot's frame with no heading
jump. Quaternions use normalized lerp with sign alignment. The filtered state
(root target, joint residual) is NOT reset at a switch: the raw root is glued
to the robot and the raw joints change continuously, so residuals stay valid.

### Metrics

`escape_move_active` (1 while ESCAPING), `escape_move_count` (switches so far
in the episode, reset on resample), `escape_intervention_speed_mps`
(planar intervention norm while ESCAPING, 0 otherwise).

## Actor / deployment

No network change. Training and standard evaluation use
`trigger_source="teacher"`. `scripts/play_adjuster_ghost.py
--escape-trigger actor` sets `trigger_source="actor"` and, each step, calls
`command.set_actor_escape_hint(policy.predict_avoidance(obs)[:, :2])` before
stepping, so the switch is driven by the head's compass exactly as it would be
on the robot. Eval scripts default to teacher; the flag is the deployability
check.

## Task / cfg

`unitree_g1_lidar_unified_reference_tracking_env_cfg(..., escape_moves=False)`.
With `True`, the command gets `EscapeMoveCfg(index_file=<manifest dir>/
escape_moves.json)`; the caller passes the mirrored manifest as
`motion_manifest`. `env_cfg.py` gains `DEFAULT_G1_BALLET_MIRROR_MANIFEST` and
`DEFAULT_G1_BALLET_ESCAPE_INDEX`.

Task `SafeMimic-Tracking-LiveAligned-Crowd-Human-Unitree-G1-Lidar-Auxiliary-
CoAdjust-Unified-Joint-Leash-Ballet-Moves`
(`LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_MOVES_TASK_ID`): Leash +
active joint reward + mirrored ballet manifest + escape moves; rl cfg identical
to the other Leash-Ballet tasks (experiment name suffix
`..._leash_ballet_moves`). Avoidance variant
`coadjust-unified-joint-leash-ballet-moves`. Routed through the seven eval and
video scripts like the Blind-NoHumans id.

`benchmark_policy_reaction_envelope.py` adds per-episode `escape_moves`
(count) and `escape_resolved` (an escape move happened and the planar
intervention while escaping stayed below 0.2 m/s), and
`summary.fair_regime.escape_resolved_rate`.

## Tests

- mirror: involution; FK consistency on a real clip (original and mirrored);
  joint/body permutation spot checks (left/right swap, pitch sign kept).
- index: synthetic clip with a known 1 m travel segment yields the expected
  entry/exit/direction/speed; a static clip is not a candidate.
- planner pure functions: trigger counting, candidate scoring (alignment gate,
  speed cap, pose penalty), blend alpha schedule.
- command state machine on the `_update_stub` style fake: trigger -> switch
  (bounds, frame, blend), exit -> resume at the saved frame with a blend,
  cooldown blocks re-trigger, reset clears state, teacher vs actor source.
- blend continuity: raw joint targets across a switch move by less than
  `2 * max step delta`.
- cfg: `escape_moves` default `None`; existing Leash-Ballet cfg unchanged
  field by field; Moves task registered with the mirrored manifest and the
  index; play cfg too.

## Rollout

1. Mirror + index the trimmed library (data step, ~minutes).
2. Implement per the plan, TDD, small commits.
3. Smoke: 16 envs, play cfg, 300 zero-action steps; confirm switches happen,
   blends are continuous, resume occurs, metrics populate.
4. Train from scratch (parent launches in tmux): `uv run train <Moves task>
   --env.scene.num-envs 4096 --agent.run-name
   coadjust_unified_joint_leash_ballet_moves_scratch_4096`.
5. Gates on the slow preset (dance clip and ballet reference) versus
   ballet@30k; side-by-side video.

## Success criteria

- No safety loss on the slow preset versus ballet@30k (2.9 % dance,
  2.2 % ballet reference safety failures).
- Fewer wrist trips (tracking terminations below 2.1 % / 13.8 %).
- Majority of fair-regime encounters resolved by an escape move with planar
  intervention under 0.2 m/s (`escape_resolved_rate > 0.5`).
- The escape in the video reads as a ballet step, not a slide.
