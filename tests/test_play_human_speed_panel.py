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


# --- termination reason printer (2026-09-09) ---------------------------------

import torch  # noqa: E402

from safe_mimic.play_panel import install_termination_printer  # noqa: E402


class _FakeManager:
  def __init__(self, flags: dict[str, list[bool]]) -> None:
    self.flags = flags

  @property
  def active_terms(self) -> list[str]:
    return list(self.flags)

  def get_term(self, name: str) -> torch.Tensor:
    return torch.tensor(self.flags[name])


class _FakeEnv:
  def __init__(self, dones_per_step: list[list[int]]) -> None:
    self._dones = iter(dones_per_step)
    self.calls = 0

  def step(self, actions):
    self.calls += 1
    return "obs", "rew", torch.tensor(next(self._dones)), {}


def test_termination_printer_reports_cause_time_and_env() -> None:
  env = _FakeEnv([[0, 0], [1, 0], [0, 1]])
  manager = _FakeManager(
    {
      "time_out": [False, False],
      "primary_human_collision": [True, False],
      "ee_body_pos": [False, True],
    }
  )
  raw_env = SimpleNamespace(termination_manager=manager, step_dt=0.02)
  lines: list[str] = []
  install_termination_printer(env, raw_env, log=lines.append)
  for _ in range(3):
    result = env.step(None)
  assert env.calls == 3 and result[2].tolist() == [0, 1]  # passthrough
  assert lines == [
    "[TERM] env 0 ended after 0.04 s: primary_human_collision",
    "[TERM] env 1 ended after 0.06 s: ee_body_pos",
  ]


def test_termination_printer_resets_the_per_env_clock() -> None:
  env = _FakeEnv([[1, 0], [0, 0], [1, 0]])
  manager = _FakeManager({"anchor_pos": [True, False]})
  raw_env = SimpleNamespace(termination_manager=manager, step_dt=0.5)
  lines: list[str] = []
  install_termination_printer(env, raw_env, log=lines.append)
  for _ in range(3):
    env.step(None)
  assert lines == [
    "[TERM] env 0 ended after 0.50 s: anchor_pos",
    "[TERM] env 0 ended after 1.00 s: anchor_pos",
  ]


def test_termination_printer_labels_a_reset_without_active_term() -> None:
  env = _FakeEnv([[1]])
  manager = _FakeManager({"time_out": [False]})
  raw_env = SimpleNamespace(termination_manager=manager, step_dt=0.02)
  lines: list[str] = []
  install_termination_printer(env, raw_env, log=lines.append)
  env.step(None)
  assert lines == [
    "[TERM] env 0 ended after 0.02 s: reset (no active termination term)"
  ]


def test_avoidance_play_cfg_keeps_the_console_quiet() -> None:
  """Play prints only the [TERM] lines: no periodic human-velocity print."""
  from mjlab.tasks.registry import load_env_cfg

  from safe_mimic.tasks import (
    LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID,
  )
  from safe_mimic.tasks.env_cfg import PRIMARY_HUMAN_EVENT_NAME

  cfg = load_env_cfg(
    LIDAR_AUXILIARY_COADJUST_UNIFIED_JOINT_LEASH_BALLET_TASK_ID, play=True
  )
  primary = cfg.events[PRIMARY_HUMAN_EVENT_NAME].params
  assert primary["show_mesh"] is True
  assert primary["print_velocity"] is False


# --- actor escape hint (2026-09-10) -------------------------------------------

from safe_mimic.play_panel import install_actor_escape_hint  # noqa: E402


def test_actor_escape_hint_feeds_the_planar_prediction_before_each_step() -> None:
  class Env:
    def __init__(self) -> None:
      self.steps: list = []

    def get_observations(self):
      return "obs"

    def step(self, actions):
      self.steps.append(actions)
      return ("obs", "rew", torch.zeros(2), {})

  class Policy:
    avoidance_planar_dim = 2

    def predict_avoidance(self, obs):
      assert obs == "obs"
      return torch.tensor([[0.3, -0.1, 9.0], [0.0, 0.5, 9.0]])

  class Command:
    def __init__(self) -> None:
      self.hints: list = []

    def set_actor_escape_hint(self, hint):
      self.hints.append(hint.clone())

  env, command = Env(), Command()
  install_actor_escape_hint(env, command, Policy())
  env.step("a")
  assert env.steps == ["a"]
  torch.testing.assert_close(command.hints[0], torch.tensor([[0.3, -0.1], [0.0, 0.5]]))
