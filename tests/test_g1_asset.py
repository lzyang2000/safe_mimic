import math

import pytest

from safe_mimic.assets.g1 import (
  MID360_PITCH_RAD,
  MID360_POSITION_M,
  MID360_SITE_NAME,
  get_g1_with_mid360_spec,
)


def test_mid360_site_matches_urdf_extrinsic() -> None:
  model = get_g1_with_mid360_spec().compile()
  site_id = model.site(MID360_SITE_NAME).id

  assert model.body(model.site_bodyid[site_id]).name == "torso_link"
  assert model.site_pos[site_id] == pytest.approx(MID360_POSITION_M)
  assert model.site_quat[site_id] == pytest.approx(
    (math.cos(MID360_PITCH_RAD / 2.0), 0.0, math.sin(MID360_PITCH_RAD / 2.0), 0.0)
  )
