"""Play-time cap on the primary human's scheduled approach speed.

The online composer imposes the approach speed through the schedule: the
human is spawned on a ring of radius ``spawn_radius`` and its source walk is
time-warped so it reaches the robot after ``intersection_delay_s``. The
realized approach speed is therefore ``radius / delay``, which is also the
definition the TTC encounter sampler uses. Capping the speed means stretching
the delay, not selecting slower source clips (every clip is re-warped).
"""

from types import SimpleNamespace

import torch

from safe_mimic.tasks.human_capsule_event import (
  HumanCapsuleMotion,
  advance_robot_speed_cap,
  limit_intersection_delay_to_speed,
)


def test_delay_is_unchanged_when_the_approach_is_already_slower_than_the_cap() -> None:
  delay = limit_intersection_delay_to_speed(
    delay_s=torch.tensor([2.0]),
    spawn_radius_m=torch.tensor([1.0]),
    max_speed_mps=torch.tensor([1.0]),
  )
  torch.testing.assert_close(delay, torch.tensor([2.0]))


def test_delay_stretches_until_the_realized_speed_equals_the_cap() -> None:
  radius = torch.tensor([3.0])
  cap = torch.tensor([0.6])
  delay = limit_intersection_delay_to_speed(
    delay_s=torch.tensor([1.0]), spawn_radius_m=radius, max_speed_mps=cap
  )
  torch.testing.assert_close(delay, torch.tensor([5.0]))
  torch.testing.assert_close(radius / delay, cap)


def test_each_environment_is_limited_independently() -> None:
  delay = limit_intersection_delay_to_speed(
    delay_s=torch.tensor([1.0, 4.0, 1.0]),
    spawn_radius_m=torch.tensor([2.0, 2.0, 0.0]),
    max_speed_mps=torch.tensor([0.5, 0.5, 0.5]),
  )
  # Env 0 is too fast and stretches; env 1 already complies; env 2 spawns on
  # top of the robot and needs no travel at all.
  torch.testing.assert_close(delay, torch.tensor([4.0, 4.0, 1.0]))


def test_inputs_are_not_mutated() -> None:
  delay_s = torch.tensor([1.0])
  radius = torch.tensor([3.0])
  limit_intersection_delay_to_speed(
    delay_s=delay_s, spawn_radius_m=radius, max_speed_mps=torch.tensor([0.6])
  )
  torch.testing.assert_close(delay_s, torch.tensor([1.0]))
  torch.testing.assert_close(radius, torch.tensor([3.0]))


def test_speed_cap_rises_immediately_to_a_new_peak() -> None:
  peak, cap = advance_robot_speed_cap(
    torch.tensor([0.2]), torch.tensor([0.8]), decay=0.99, floor=0.3
  )
  torch.testing.assert_close(peak, torch.tensor([0.8]))
  torch.testing.assert_close(cap, torch.tensor([0.8]))


def test_speed_cap_decays_instead_of_following_a_momentary_stop() -> None:
  peak, cap = advance_robot_speed_cap(
    torch.tensor([0.8]), torch.tensor([0.0]), decay=0.5, floor=0.3
  )
  torch.testing.assert_close(peak, torch.tensor([0.4]))
  torch.testing.assert_close(cap, torch.tensor([0.4]))


def test_speed_cap_never_falls_below_the_floor() -> None:
  # A standing robot would otherwise cap the human at zero and the encounter
  # would never happen.
  peak, cap = advance_robot_speed_cap(
    torch.tensor([0.05]), torch.tensor([0.0]), decay=0.5, floor=0.3
  )
  torch.testing.assert_close(peak, torch.tensor([0.025]))
  torch.testing.assert_close(cap, torch.tensor([0.3]))


def _event_stub(*, enabled: bool, peak: list[float]) -> HumanCapsuleMotion:
  event = HumanCapsuleMotion.__new__(HumanCapsuleMotion)
  event.limit_approach_speed_to_robot = enabled
  event.approach_speed_floor_mps = 0.3
  event._robot_speed_peak_mps = torch.tensor(peak)
  event._env = SimpleNamespace(step_dt=0.02)
  return event


def test_scheduled_delay_is_untouched_while_the_limit_is_off() -> None:
  event = _event_stub(enabled=False, peak=[0.6])
  steps = torch.tensor([50])
  limited = event._speed_limited_delay_steps(
    torch.tensor([0]), steps, torch.tensor([3.0])
  )
  torch.testing.assert_close(limited, steps)


def test_scheduled_delay_stretches_to_the_robot_speed_when_the_limit_is_on() -> None:
  event = _event_stub(enabled=True, peak=[0.6])
  # 3 m in 1 s is 3 m/s; the robot peaked at 0.6 m/s, so 5 s = 250 steps.
  limited = event._speed_limited_delay_steps(
    torch.tensor([0]), torch.tensor([50]), torch.tensor([3.0])
  )
  torch.testing.assert_close(limited, torch.tensor([250]))


def test_scheduled_delay_uses_the_floor_for_a_standing_robot() -> None:
  event = _event_stub(enabled=True, peak=[0.0])
  # Floor 0.3 m/s over 3 m is 10 s = 500 steps.
  limited = event._speed_limited_delay_steps(
    torch.tensor([0]), torch.tensor([50]), torch.tensor([3.0])
  )
  torch.testing.assert_close(limited, torch.tensor([500]))


def test_scheduled_delay_is_per_environment_and_at_least_one_step() -> None:
  event = _event_stub(enabled=True, peak=[0.6, 5.0])
  limited = event._speed_limited_delay_steps(
    torch.tensor([0, 1]), torch.tensor([50, 1]), torch.tensor([3.0, 0.0])
  )
  torch.testing.assert_close(limited, torch.tensor([250, 1]))
