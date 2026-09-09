"""Viser play-panel checkbox that caps the human's approach speed."""

from types import SimpleNamespace

import pytest

from safe_mimic.play_panel import (
  HUMAN_SPEED_LIMIT_HINT,
  HUMAN_SPEED_LIMIT_LABEL,
  add_human_speed_limit_checkbox,
  primary_human_event,
)


class _FakeCheckbox:
  def __init__(self, label: str, initial_value: bool, hint: str | None) -> None:
    self.label = label
    self.value = initial_value
    self.hint = hint
    self._callbacks: list = []

  def on_update(self, func):
    self._callbacks.append(func)
    return func

  def fire(self) -> None:
    for func in self._callbacks:
      func(self)


class _FakeGui:
  def __init__(self) -> None:
    self.checkboxes: list[_FakeCheckbox] = []

  def add_checkbox(
    self, label: str, initial_value: bool = False, *, hint: str | None = None
  ) -> _FakeCheckbox:
    checkbox = _FakeCheckbox(label, initial_value, hint)
    self.checkboxes.append(checkbox)
    return checkbox


def test_checkbox_starts_off_and_leaves_the_event_unlimited() -> None:
  event = SimpleNamespace(limit_approach_speed_to_robot=True)
  gui = _FakeGui()
  checkbox = add_human_speed_limit_checkbox(gui, event)
  assert checkbox.label == HUMAN_SPEED_LIMIT_LABEL
  assert checkbox.hint == HUMAN_SPEED_LIMIT_HINT
  assert checkbox.value is False
  assert event.limit_approach_speed_to_robot is False


def test_toggling_the_checkbox_limits_and_unlimits_the_event() -> None:
  event = SimpleNamespace(limit_approach_speed_to_robot=False)
  gui = _FakeGui()
  checkbox = add_human_speed_limit_checkbox(gui, event)
  checkbox.value = True
  checkbox.fire()
  assert event.limit_approach_speed_to_robot is True
  checkbox.value = False
  checkbox.fire()
  assert event.limit_approach_speed_to_robot is False


def test_primary_human_event_is_read_from_the_event_manager() -> None:
  event = SimpleNamespace(limit_approach_speed_to_robot=False)
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      event_manager=SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(func=event)
      )
    )
  )
  assert primary_human_event(env) is event


def test_primary_human_event_reports_a_task_without_one() -> None:
  def _missing(name: str):
    raise ValueError(name)

  env = SimpleNamespace(
    unwrapped=SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=_missing))
  )
  with pytest.raises(LookupError, match="primary human"):
    primary_human_event(env)
