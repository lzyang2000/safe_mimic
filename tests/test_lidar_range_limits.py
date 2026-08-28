import torch

from safe_mimic.sensing.held_scan import _apply_range_limits_


def test_range_limits_turn_near_and_far_returns_into_misses() -> None:
  distances = torch.tensor([[-1.0, 0.29, 0.3, 5.0, 5.01]])
  normals = torch.ones(1, 5, 3)
  origins = torch.tensor([[[1.0, 2.0, 3.0]]]).expand(1, 5, 3)
  hits = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)

  _apply_range_limits_(
    distances,
    normals,
    hits,
    origins,
    min_distance=0.3,
    max_distance=5.0,
  )

  torch.testing.assert_close(
    distances, torch.tensor([[-1.0, -1.0, 0.3, 5.0, -1.0]])
  )
  assert torch.equal(normals[0, 2:4], torch.ones(2, 3))
  assert torch.count_nonzero(normals[0, (0, 1, 4)]) == 0
  assert torch.equal(hits[0, (0, 1, 4)], origins[0, (0, 1, 4)])
