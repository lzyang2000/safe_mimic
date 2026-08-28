from safe_mimic.motions.g1_size_sample import (
  grouped_distribution,
  stratified_payload_sample,
  tracker_float32_payload_bytes,
  tracker_frame_count,
)


def test_tracker_payload_matches_converter_schema() -> None:
  assert tracker_frame_count(121) == 51
  expected_values = 29 + 29 + 30 * 3 + 30 * 4 + 30 * 3 + 30 * 3
  assert tracker_float32_payload_bytes(51) == 51 * expected_values * 4 + 8


def test_stratified_sample_is_deterministic_and_preserves_strata() -> None:
  records = [
    {
      "csv_path": f"{package}/{index}.csv",
      "package": package,
      "category": "shared",
      "tracker_payload_bytes": 100,
      "tracker_frame_count": 1,
    }
    for package in ("a", "b")
    for index in range(10)
  ]

  selected = stratified_payload_sample(records, 1000, seed="test")
  repeated = stratified_payload_sample(list(reversed(records)), 1000, seed="test")

  assert {record["csv_path"] for record in selected} == {
    record["csv_path"] for record in repeated
  }
  assert len(selected) == 10
  distribution = grouped_distribution(selected, "package")
  assert distribution["a"]["clips"] == 5
  assert distribution["b"]["clips"] == 5
