"""Left/right mirroring of G1 tracker NPZ clips (reflection across the x-z plane).

The permutations and signs are derived from the MuJoCo model rather than
hand-written: ``left_*`` and ``right_*`` joints and bodies swap, a hinge about
the body y axis (pitch) keeps its sign under the reflection, roll (x) and yaw
(z) hinges reverse. Joint order is the model's hinge order and body order the
model's bodies minus ``world``, which is the tracker-NPZ convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np

TIME_MAJOR_BODY_KEYS = (
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def mirror_name(name: str) -> str:
  """Swap the ``left_`` / ``right_`` prefix of a joint or body name."""
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name


@dataclass(frozen=True)
class G1MirrorSpec:
  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]
  joint_perm: np.ndarray
  """``mirrored[:, j] = original[:, joint_perm[j]] * joint_sign[j]``."""

  joint_sign: np.ndarray
  body_perm: np.ndarray
  """``mirrored[:, b] = reflect(original[:, body_perm[b]])``."""

  @classmethod
  def from_model(cls, model: mujoco.MjModel) -> G1MirrorSpec:
    joints = [
      j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    joint_names = tuple(
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints
    )
    body_names = tuple(
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
      for b in range(1, model.nbody)
    )
    joint_perm = np.array(
      [joint_names.index(mirror_name(n)) for n in joint_names], dtype=np.int64
    )
    joint_sign = np.array(
      [1.0 if abs(float(model.jnt_axis[j][1])) > 0.5 else -1.0 for j in joints]
    )
    body_perm = np.array(
      [body_names.index(mirror_name(n)) for n in body_names], dtype=np.int64
    )
    return cls(joint_names, body_names, joint_perm, joint_sign, body_perm)


def mirror_motion_arrays(
  arrays: Mapping[str, np.ndarray], spec: G1MirrorSpec
) -> dict[str, np.ndarray]:
  """Reflect every array of a tracker NPZ across the sagittal plane."""
  out: dict[str, np.ndarray] = {}
  for key, raw in arrays.items():
    value = np.asarray(raw)
    if key in ("joint_pos", "joint_vel"):
      out[key] = (value[:, spec.joint_perm] * spec.joint_sign).astype(value.dtype)
    elif key in TIME_MAJOR_BODY_KEYS:
      swapped = value[:, spec.body_perm].copy()
      if key == "body_quat_w":
        swapped[..., 1] *= -1.0  # (w, x, y, z) -> (w, -x, y, -z)
        swapped[..., 3] *= -1.0
      elif key == "body_ang_vel_w":
        swapped[..., 0] *= -1.0  # pseudo-vector: x and z flip
        swapped[..., 2] *= -1.0
      else:
        swapped[..., 1] *= -1.0
      out[key] = swapped
    else:
      out[key] = value.copy()
  return out


__all__ = [
  "G1MirrorSpec",
  "TIME_MAJOR_BODY_KEYS",
  "mirror_motion_arrays",
  "mirror_name",
]
