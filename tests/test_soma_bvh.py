from pathlib import Path

import numpy as np

from safe_mimic.motions.soma_bvh import load_bvh_samples, load_bvh_window


def test_bvh_translation_and_rotation_sampling(tmp_path: Path) -> None:
  path = tmp_path / "tiny.bvh"
  path.write_text(
    """HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Hips
  {
    OFFSET 0 100 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Knee
    {
      OFFSET 10 0 0
      CHANNELS 3 Zrotation Yrotation Xrotation
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.01
0 0 0 0 0 0 0 100 0 0 0 0 0 0 0
0 0 0 0 0 0 10 100 0 90 0 0 0 0 0
"""
  )
  motion = load_bvh_samples(path, sample_count=2)
  hips = motion.joint_names.index("Hips")
  knee = motion.joint_names.index("Knee")

  assert np.allclose(motion.positions_m[:, hips], ((0, 1, 0), (0.1, 1, 0)))
  assert np.allclose(motion.positions_m[0, knee], (0.1, 1, 0))
  assert np.allclose(motion.positions_m[1, knee], (0.1, 1.1, 0))


def test_bvh_window_preserves_source_time(tmp_path: Path) -> None:
  path = tmp_path / "window.bvh"
  frames = "\n".join(f"{x} 0 0 0 0 0" for x in range(6))
  path.write_text(
    f"""HIERARCHY
ROOT Root
{{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
}}
MOTION
Frames: 6
Frame Time: 0.1
{frames}
"""
  )

  motion = load_bvh_window(path, start_time_s=0.1, duration_s=0.3, output_fps=10)

  assert np.array_equal(motion.frame_indices, (1, 2, 3))
  assert np.allclose(motion.positions_m[:, 0, 0], (0.01, 0.02, 0.03))
