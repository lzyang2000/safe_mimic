"""Viser play-panel controls specific to Safe Mimic tasks.

`mjlab`'s :class:`ViserPlayViewer` owns the stock panel; these helpers add
Safe Mimic controls on top of it without touching the dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from mjlab.viewer import ViserPlayViewer
from typing_extensions import override

from safe_mimic.tasks.env_cfg import PRIMARY_HUMAN_EVENT_NAME

HUMAN_SPEED_LIMIT_LABEL = "Human approach <= robot speed"
HUMAN_SPEED_LIMIT_HINT = (
  "Stretch each scheduled encounter so the primary human never approaches "
  "faster than the robot's recent peak planar speed. The spawn ring, crossing "
  "angle and intersection point are unchanged; only the walk slows down."
)


class _SpeedLimitedEvent(Protocol):
  limit_approach_speed_to_robot: bool


def primary_human_event(env: Any) -> _SpeedLimitedEvent:
  """Return the primary human's event-term instance for a play environment."""
  manager = env.unwrapped.event_manager
  try:
    term_cfg = manager.get_term_cfg(PRIMARY_HUMAN_EVENT_NAME)
  except (ValueError, KeyError) as error:
    raise LookupError(
      f"this task has no primary human event {PRIMARY_HUMAN_EVENT_NAME!r}"
    ) from error
  return term_cfg.func


def add_human_speed_limit_checkbox(
  gui: Any, event: _SpeedLimitedEvent, *, initial: bool = False
) -> Any:
  """Add the approach-speed checkbox and bind it to the human event.

  The event owns the flag, so the checkbox only mirrors it; the cap applies
  from the next scheduled encounter onward.
  """
  checkbox = gui.add_checkbox(
    HUMAN_SPEED_LIMIT_LABEL, initial_value=initial, hint=HUMAN_SPEED_LIMIT_HINT
  )
  event.limit_approach_speed_to_robot = bool(initial)

  def _on_update(_event: Any) -> None:
    event.limit_approach_speed_to_robot = bool(checkbox.value)

  checkbox.on_update(_on_update)
  return checkbox


def install_termination_printer(
  env: Any, raw_env: Any, log: Callable[[str], None] = print
) -> None:
  """Wrap ``env.step`` so every episode end prints its cause and duration.

  Works for any viewer (native or viser) because both call ``env.step``.
  The cause is read from the raw env's termination manager right after the
  step; mjlab keeps the per-term flags until the next ``compute``.
  """
  inner_step = env.step
  clock: torch.Tensor | None = None

  def step(actions: Any) -> tuple[Any, ...]:
    nonlocal clock
    result = inner_step(actions)
    dones = torch.as_tensor(result[2]).reshape(-1).bool()
    if clock is None:
      clock = torch.zeros(dones.shape[0], dtype=torch.float64)
    clock += float(raw_env.step_dt)
    if bool(dones.any()):
      manager = raw_env.termination_manager
      for env_idx in torch.nonzero(dones).flatten().tolist():
        causes = [
          name for name in manager.active_terms if bool(manager.get_term(name)[env_idx])
        ]
        cause = ", ".join(causes) or "reset (no active termination term)"
        log(f"[TERM] env {env_idx} ended after {float(clock[env_idx]):.2f} s: {cause}")
        clock[env_idx] = 0.0
    return result

  env.step = step


def install_actor_escape_hint(env: Any, command: Any, policy: Any) -> None:
  """Feed the actor's planar prediction to the escape-move planner each step.

  Deployment path for escape moves: before every ``env.step`` the current
  observations are run through the actor's avoidance head and the first
  ``avoidance_planar_dim`` values (body-frame planar correction, m/s) are handed
  to ``command.set_actor_escape_hint``. The command must have been configured
  with ``trigger_source="actor"``.
  """
  inner_step = env.step
  planar_dim = int(getattr(policy, "avoidance_planar_dim", 2))

  def step(actions: Any) -> tuple[Any, ...]:
    with torch.no_grad():
      prediction = policy.predict_avoidance(env.get_observations())
    command.set_actor_escape_hint(prediction[..., :planar_dim])
    return inner_step(actions)

  env.step = step


class SafeMimicPlayViewer(ViserPlayViewer):
  """Play viewer whose panel carries the Safe Mimic human-speed control."""

  @override
  def setup(self) -> None:
    super().setup()
    try:
      event = primary_human_event(self.env)
    except LookupError:
      return
    with self._server.gui.add_folder("Safe Mimic"):
      add_human_speed_limit_checkbox(self._server.gui, event)


__all__ = [
  "HUMAN_SPEED_LIMIT_HINT",
  "install_actor_escape_hint",
  "install_termination_printer",
  "HUMAN_SPEED_LIMIT_LABEL",
  "SafeMimicPlayViewer",
  "add_human_speed_limit_checkbox",
  "primary_human_event",
]
