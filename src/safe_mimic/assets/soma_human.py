"""MJLab debug asset for the BONES-SEED SOMA mesh."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg


def get_soma_debug_mesh_spec(mesh_path: Path | str) -> mujoco.MjSpec:
  """Load the converted SOMA OBJ as a non-colliding MJLab debug entity.

  The source USD is a skinned mesh, which MuJoCo does not animate natively.
  This static entity is for overlay/debug views. Runtime obstacle geometry should
  use the fitted capsules instead.
  """

  mesh_path = Path(mesh_path).resolve()
  xml = (
    "<mujoco>"
    "<asset>"
    f'<mesh name="soma_debug_mesh" file="{mesh_path}"/>'
    "</asset>"
    "<worldbody>"
    '<body name="soma_debug_mesh_body">'
    '<geom name="soma_debug_mesh" type="mesh" mesh="soma_debug_mesh" '
    'rgba="0.8 0.63 0.48 0.6" group="2" '
    'contype="0" conaffinity="0" density="0"/>'
    "</body>"
    "</worldbody>"
    "</mujoco>"
  )
  return mujoco.MjSpec.from_string(xml)


def get_soma_debug_mesh_cfg(mesh_path: Path | str) -> EntityCfg:
  """Return an MJLab entity config for the static non-colliding SOMA mesh."""

  from mjlab.entity import EntityCfg

  return EntityCfg(spec_fn=partial(get_soma_debug_mesh_spec, mesh_path))
