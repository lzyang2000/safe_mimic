"""Load and skin the ASCII USD mesh bundled with BONES-SEED SOMA."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safe_mimic.motions.soma_bvh import BvhMotion


@dataclass(frozen=True)
class SomaMeshSkin:
  """SOMA mesh, topology, and linear-blend skinning data."""

  points_cm: np.ndarray
  triangles: np.ndarray
  joint_names: tuple[str, ...]
  joint_paths: tuple[tuple[str, ...], ...]
  joint_indices: np.ndarray
  joint_weights: np.ndarray
  bind_transforms: np.ndarray
  geom_bind_transform: np.ndarray

  def skin_frame_mujoco(
    self,
    motion: BvhMotion,
    frame_index: int,
    body_scale_xyz: np.ndarray | None = None,
  ) -> np.ndarray:
    """Skin one frame and return MuJoCo X/Y/Z-up vertices in meters."""

    motion_indices = {name: index for index, name in enumerate(motion.joint_names)}
    joint_map = np.asarray(
      [motion_indices[name] for name in self.joint_names], dtype=np.int64
    )
    positions_cm = motion.positions_m[frame_index, joint_map] * 100.0
    rotations = motion.rotations_world[frame_index, joint_map]

    current = np.zeros_like(self.bind_transforms)
    current[:, :3, :3] = np.swapaxes(rotations, -1, -2)
    current[:, 3, :3] = positions_cm
    current[:, 3, 3] = 1.0
    skin_matrices = np.linalg.inv(self.bind_transforms) @ current

    points = np.column_stack((self.points_cm, np.ones(len(self.points_cm))))
    skinned = np.zeros((len(points), 3), dtype=np.float64)
    for influence in range(self.joint_indices.shape[1]):
      weights = self.joint_weights[:, influence]
      active = weights > 0.0
      if not np.any(active):
        continue
      matrices = skin_matrices[self.joint_indices[active, influence]]
      transformed = np.einsum("ni,nij->nj", points[active], matrices)
      skinned[active] += transformed[:, :3] * weights[active, None]

    mujoco_points = np.stack(
      (skinned[:, 0], skinned[:, 2], skinned[:, 1]), axis=-1
    ) * 0.01
    if body_scale_xyz is not None:
      mujoco_points *= body_scale_xyz
    return mujoco_points

  def write_mujoco_obj(self, path: Path | str) -> None:
    """Write the bind-pose mesh as a Z-up OBJ that MJLab can load."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.stack(
      (self.points_cm[:, 0], self.points_cm[:, 2], self.points_cm[:, 1]),
      axis=-1,
    ) * 0.01
    with path.open("w") as output:
      output.write("# BONES-SEED SOMA debug mesh; visualization only\n")
      for point in points:
        output.write(f"v {point[0]:.7g} {point[1]:.7g} {point[2]:.7g}\n")
      for triangle in self.triangles + 1:
        output.write(f"f {triangle[0]} {triangle[1]} {triangle[2]}\n")


@dataclass(frozen=True)
class SomaViserSkin:
  """Runtime skinning payload reduced to the compact motion joint set."""

  vertices_m: np.ndarray
  triangles: np.ndarray
  joint_names: tuple[str, ...]
  bind_positions_m: np.ndarray
  bind_quaternions_wxyz: np.ndarray
  bind_inverse_matrices: np.ndarray
  skin_weights: np.ndarray

  def skin_vertices_mujoco(
    self,
    joint_positions_m: np.ndarray,
    joint_quaternions_wxyz: np.ndarray,
    *,
    body_scale_xyz: np.ndarray | None = None,
    placement_yaw: float = 0.0,
    translation_w: np.ndarray | None = None,
  ) -> np.ndarray:
    """Skin one compact pose and return world-space MuJoCo vertices.

    This CPU path is only used by the one-environment Viser debugger. Updating
    a regular mesh avoids Viser's skinned-mesh node transform being applied to
    both the bones and the mesh, which otherwise offsets a globally placed rig.
    """
    positions = np.asarray(joint_positions_m, dtype=np.float64)
    quaternions = np.asarray(joint_quaternions_wxyz, dtype=np.float64)
    joint_count = len(self.joint_names)
    if positions.shape != (joint_count, 3):
      raise ValueError("joint_positions_m must have shape [joint_count, 3]")
    if quaternions.shape != (joint_count, 4):
      raise ValueError(
        "joint_quaternions_wxyz must have shape [joint_count, 4]"
      )

    current = np.zeros((joint_count, 4, 4), dtype=np.float64)
    current[:, :3, :3] = _quaternion_wxyz_to_matrix(quaternions)
    current[:, :3, 3] = positions
    current[:, 3, 3] = 1.0
    skin_matrices = current @ self.bind_inverse_matrices
    points = np.column_stack(
      (self.vertices_m.astype(np.float64), np.ones(len(self.vertices_m)))
    )
    skinned = np.zeros((len(points), 3), dtype=np.float64)
    for joint_index in range(joint_count):
      weights = self.skin_weights[:, joint_index]
      active = weights > 0.0
      if not np.any(active):
        continue
      transformed = np.einsum(
        "nij,nj->ni", skin_matrices[joint_index][None], points[active]
      )
      skinned[active] += transformed[:, :3] * weights[active, None]

    if body_scale_xyz is not None:
      skinned *= np.asarray(body_scale_xyz, dtype=np.float64)
    cosine = np.cos(placement_yaw)
    sine = np.sin(placement_yaw)
    x = cosine * skinned[:, 0] - sine * skinned[:, 1]
    y = sine * skinned[:, 0] + cosine * skinned[:, 1]
    skinned[:, 0] = x
    skinned[:, 1] = y
    if translation_w is not None:
      skinned += np.asarray(translation_w, dtype=np.float64)
    return skinned.astype(np.float32)


def cluster_soma_viser_skin(
  skin: SomaViserSkin, voxel_size_m: float
) -> SomaViserSkin:
  """Reduce a debug skin by merging nearby bind-pose vertices.

  This only changes the Viser mesh. Physics and LiDAR continue to use the full
  capsule representation. Averaging both bind vertices and skin weights keeps
  the clustered mesh articulated while substantially reducing websocket and
  browser rendering load for dense crowds.
  """

  if voxel_size_m <= 0.0:
    raise ValueError("voxel_size_m must be positive")
  origin = skin.vertices_m.min(axis=0)
  cells = np.floor(
    (skin.vertices_m - origin) / voxel_size_m + 0.5
  ).astype(np.int64)
  _, inverse = np.unique(cells, axis=0, return_inverse=True)
  cluster_count = int(inverse.max()) + 1
  counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)

  vertices = np.zeros((cluster_count, 3), dtype=np.float64)
  np.add.at(vertices, inverse, skin.vertices_m)
  vertices /= counts[:, None]

  weights = np.zeros(
    (cluster_count, skin.skin_weights.shape[1]), dtype=np.float64
  )
  np.add.at(weights, inverse, skin.skin_weights)
  weights /= counts[:, None]
  weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

  triangles = inverse[skin.triangles]
  nondegenerate = (
    (triangles[:, 0] != triangles[:, 1])
    & (triangles[:, 1] != triangles[:, 2])
    & (triangles[:, 0] != triangles[:, 2])
  )
  triangles = triangles[nondegenerate]
  canonical = np.sort(triangles, axis=1)
  _, unique_indices = np.unique(canonical, axis=0, return_index=True)
  triangles = triangles[np.sort(unique_indices)]

  return SomaViserSkin(
    vertices_m=vertices.astype(np.float32),
    triangles=triangles.astype(np.uint32),
    joint_names=skin.joint_names,
    bind_positions_m=skin.bind_positions_m,
    bind_quaternions_wxyz=skin.bind_quaternions_wxyz,
    bind_inverse_matrices=skin.bind_inverse_matrices,
    skin_weights=weights.astype(np.float32),
  )


def _quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
  """Convert normalized wxyz quaternions to column-vector rotation matrices."""
  quaternions = np.asarray(quaternions, dtype=np.float64)
  quaternions /= np.maximum(
    np.linalg.norm(quaternions, axis=-1, keepdims=True), 1e-12
  )
  w, x, y, z = np.moveaxis(quaternions, -1, 0)
  matrices = np.empty((*quaternions.shape[:-1], 3, 3), dtype=np.float64)
  matrices[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  matrices[..., 0, 1] = 2.0 * (x * y - z * w)
  matrices[..., 0, 2] = 2.0 * (x * z + y * w)
  matrices[..., 1, 0] = 2.0 * (x * y + z * w)
  matrices[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  matrices[..., 1, 2] = 2.0 * (y * z - x * w)
  matrices[..., 2, 0] = 2.0 * (x * z - y * w)
  matrices[..., 2, 1] = 2.0 * (y * z + x * w)
  matrices[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
  return matrices


def _matrix_to_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
  """Convert proper rotation matrices to normalized wxyz quaternions."""
  matrices = np.asarray(matrices, dtype=np.float64)
  quaternions = np.empty((*matrices.shape[:-2], 4), dtype=np.float64)
  for index in np.ndindex(matrices.shape[:-2]):
    matrix = matrices[index]
    trace = float(np.trace(matrix))
    if trace > 0.0:
      scale = np.sqrt(trace + 1.0) * 2.0
      quaternion = np.asarray(
        (
          0.25 * scale,
          (matrix[2, 1] - matrix[1, 2]) / scale,
          (matrix[0, 2] - matrix[2, 0]) / scale,
          (matrix[1, 0] - matrix[0, 1]) / scale,
        )
      )
    else:
      diagonal = np.diag(matrix)
      axis = int(np.argmax(diagonal))
      first = axis
      second = (axis + 1) % 3
      third = (axis + 2) % 3
      scale = np.sqrt(
        1.0 + matrix[first, first] - matrix[second, second] - matrix[third, third]
      ) * 2.0
      quaternion = np.empty(4, dtype=np.float64)
      quaternion[first + 1] = 0.25 * scale
      quaternion[0] = (matrix[third, second] - matrix[second, third]) / scale
      quaternion[second + 1] = (
        matrix[second, first] + matrix[first, second]
      ) / scale
      quaternion[third + 1] = (
        matrix[third, first] + matrix[first, third]
      ) / scale
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    quaternions[index] = quaternion if quaternion[0] >= 0.0 else -quaternion
  return quaternions


def prepare_soma_viser_skin(
  skin: SomaMeshSkin,
  runtime_joint_names: tuple[str, ...],
) -> SomaViserSkin:
  """Collapse the full SOMA rig onto joints retained by the CUDA motion bank.

  Fingers, facial joints, and terminal joints are not animated by the compact
  bank. Assigning their weights to the nearest retained ancestor is exact while
  those omitted joints remain at their bind-pose offsets.
  """
  runtime_indices = {name: index for index, name in enumerate(runtime_joint_names)}
  if len(runtime_indices) != len(runtime_joint_names):
    raise ValueError("runtime joint names must be unique")
  mesh_indices = {name: index for index, name in enumerate(skin.joint_names)}
  missing = set(runtime_joint_names) - mesh_indices.keys()
  if missing:
    raise ValueError(f"runtime joints missing from SOMA mesh: {sorted(missing)}")

  full_to_runtime = np.empty(len(skin.joint_names), dtype=np.int64)
  for index, path in enumerate(skin.joint_paths):
    retained = next((name for name in reversed(path) if name in runtime_indices), None)
    if retained is None:
      raise ValueError(f"SOMA joint path has no retained ancestor: {'/'.join(path)}")
    full_to_runtime[index] = runtime_indices[retained]

  weights = np.zeros(
    (len(skin.points_cm), len(runtime_joint_names)), dtype=np.float32
  )
  vertices = np.arange(len(skin.points_cm))
  for influence in range(skin.joint_indices.shape[1]):
    mapped = full_to_runtime[skin.joint_indices[:, influence]]
    np.add.at(
      weights,
      (vertices, mapped),
      skin.joint_weights[:, influence].astype(np.float32),
    )
  if weights.shape[1] > 4:
    strongest = np.argpartition(weights, -4, axis=1)[:, -4:]
    limited = np.zeros_like(weights)
    rows = np.arange(len(weights))[:, None]
    limited[rows, strongest] = weights[rows, strongest]
    limited /= np.maximum(limited.sum(axis=1, keepdims=True), 1e-12)
    weights = limited

  basis = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
  )
  selected = np.asarray(
    [mesh_indices[name] for name in runtime_joint_names], dtype=np.int64
  )
  bind = skin.bind_transforms[selected]
  bind_positions_m = np.einsum("ij,nj->ni", basis, bind[:, 3, :3]) * 0.01
  bind_rotations_bvh = np.swapaxes(bind[:, :3, :3], -1, -2)
  bind_rotations_mujoco = np.einsum(
    "ij,njk,lk->nil", basis, bind_rotations_bvh, basis
  )
  vertices_m = np.einsum("ij,nj->ni", basis, skin.points_cm) * 0.01
  bind_quaternions = _matrix_to_quaternion_wxyz(bind_rotations_mujoco)
  bind_matrices = np.zeros((len(selected), 4, 4), dtype=np.float64)
  bind_matrices[:, :3, :3] = bind_rotations_mujoco
  bind_matrices[:, :3, 3] = bind_positions_m
  bind_matrices[:, 3, 3] = 1.0
  return SomaViserSkin(
    vertices_m=vertices_m.astype(np.float32),
    triangles=skin.triangles.astype(np.uint32),
    joint_names=runtime_joint_names,
    bind_positions_m=bind_positions_m.astype(np.float32),
    bind_quaternions_wxyz=bind_quaternions.astype(np.float32),
    bind_inverse_matrices=np.linalg.inv(bind_matrices),
    skin_weights=weights,
  )


def _read_usd_properties(path: Path) -> dict[str, object]:
  wanted = {
    "uniform matrix4d[] bindTransforms": "bind_transforms",
    "uniform token[] joints": "joint_names",
    "int[] faceVertexCounts": "face_counts",
    "int[] faceVertexIndices": "face_indices",
    "point3f[] points": "points",
    "matrix4d primvars:skel:geomBindTransform": "geom_bind_transform",
    "int[] primvars:skel:jointIndices": "joint_indices",
    "float[] primvars:skel:jointWeights": "joint_weights",
  }
  properties: dict[str, object] = {}
  with path.open() as source:
    for line in source:
      stripped = line.strip()
      for prefix, name in wanted.items():
        if name in properties or not stripped.startswith(prefix + " ="):
          continue
        value = stripped.split("=", maxsplit=1)[1].strip()
        if value.startswith("[") and "] (" in value:
          value = value.rsplit("] (", maxsplit=1)[0] + "]"
        properties[name] = ast.literal_eval(value)
        break
  missing = set(wanted.values()) - properties.keys()
  if missing:
    raise ValueError(f"Missing SOMA USD properties: {sorted(missing)}")
  return properties


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
  triangles: list[tuple[int, int, int]] = []
  cursor = 0
  for count in face_counts:
    face = face_indices[cursor : cursor + count]
    triangles.extend(
      (int(face[0]), int(face[i]), int(face[i + 1]))
      for i in range(1, count - 1)
    )
    cursor += int(count)
  return np.asarray(triangles, dtype=np.int32)


def load_soma_mesh_skin(path: Path | str) -> SomaMeshSkin:
  """Parse the mesh and skinning arrays from SOMA's ASCII USD file."""

  properties = _read_usd_properties(Path(path))
  points = np.asarray(properties["points"], dtype=np.float64)
  face_counts = np.asarray(properties["face_counts"], dtype=np.int32)
  face_indices = np.asarray(properties["face_indices"], dtype=np.int32)
  joint_indices = np.asarray(properties["joint_indices"], dtype=np.int32)
  joint_weights = np.asarray(properties["joint_weights"], dtype=np.float64)
  influences = len(joint_indices) // len(points)
  joint_indices = joint_indices.reshape(len(points), influences)
  joint_weights = joint_weights.reshape(len(points), influences)
  joint_paths = tuple(
    tuple(str(name).split("/")) for name in properties["joint_names"]
  )
  joint_names = tuple(path[-1] for path in joint_paths)
  return SomaMeshSkin(
    points_cm=points,
    triangles=_triangulate(face_counts, face_indices),
    joint_names=joint_names,
    joint_paths=joint_paths,
    joint_indices=joint_indices,
    joint_weights=joint_weights,
    bind_transforms=np.asarray(properties["bind_transforms"], dtype=np.float64),
    geom_bind_transform=np.asarray(
      properties["geom_bind_transform"], dtype=np.float64
    ),
  )
