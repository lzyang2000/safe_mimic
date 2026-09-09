"""Viser play-panel controls specific to Safe Mimic tasks.

`mjlab`'s :class:`ViserPlayViewer` owns the stock panel; these helpers add
Safe Mimic controls on top of it without touching the dependency.
"""

from __future__ import annotations

from typing import Any, Protocol

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
  "HUMAN_SPEED_LIMIT_LABEL",
  "SafeMimicPlayViewer",
  "add_human_speed_limit_checkbox",
  "primary_human_event",
]
