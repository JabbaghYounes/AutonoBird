"""
scripts/autonomy/gesture_action_map.py — gesture → orchestrator action mapping

Turns the recogniser's `Gesture` output into method calls on the
`Orchestrator`. Includes a cooldown timer so a sustained gesture doesn't
trigger a flurry of duplicate commands, and a guarded mapping so a NONE
gesture is silently ignored (rather than mapping to "do nothing"
explicitly — keeps the dispatch logic simple at the call site).

Designed to be used as a subscriber that the operator (or the
orchestrator itself) drives once per perception frame after the
classifier has updated:

    classifier = GestureClassifier()
    action_map = GestureActionMap(orchestrator)
    while running:
        frame = perception.latest()
        if frame is not None:
            g = classifier.update(frame)
            action_map.dispatch(g)

Mapping (kept deliberately tight for the safety-critical first version):

    STOP    →  orchestrator.command_hold()    (planner pauses, vehicle holds)
    LAND    →  orchestrator.command_land()    (planner stops, vehicle LAND mode)
    COME    →  orchestrator.command_resume()  (planner resumes — operator-tracked
                                                follow is a future extension)
    RECEDE  →  orchestrator.command_hold()    (treated as hold for now; reverse
                                                cruise is the future extension)
    NONE    →  ignored

Cooldown prevents firing the same action twice within `cooldown_s`. Even
with the classifier's temporal smoothing, a sustained gesture would
otherwise dispatch on every frame after the first.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from gesture_classifier import Gesture


class GestureActionMap:
    """Dispatch table from recognised gestures to orchestrator commands."""

    def __init__(
        self,
        orchestrator,                       # type: ignore[no-untyped-def]
        cooldown_s: float = 2.0,
        on_dispatch: Optional[Callable[[Gesture], None]] = None,
    ):
        """
        orchestrator: anything exposing the command_hold / command_land /
            command_resume methods. Decoupled from the concrete
            Orchestrator class so tests can pass in a mock.
        cooldown_s: minimum interval between dispatches. Counted per the
            wall clock, not per gesture.
        on_dispatch: optional callback fired each time an action is
            dispatched (e.g. logging, LED feedback). Receives the
            Gesture that was dispatched.
        """
        self.orch = orchestrator
        self.cooldown_s = float(cooldown_s)
        self.on_dispatch = on_dispatch
        self._last_t: float = 0.0
        self._last_gesture: Gesture = Gesture.NONE

    def dispatch(self, gesture: Gesture) -> bool:
        """Map the gesture to a command, respecting cooldown.

        Returns True if an action was dispatched, False otherwise.
        """
        if gesture == Gesture.NONE:
            return False
        now = time.time()
        if now - self._last_t < self.cooldown_s:
            return False

        ok = self._dispatch_one(gesture)
        if ok:
            self._last_t = now
            self._last_gesture = gesture
            if self.on_dispatch is not None:
                try:
                    self.on_dispatch(gesture)
                except Exception:
                    pass
        return ok

    def _dispatch_one(self, gesture: Gesture) -> bool:
        """The actual mapping. Returns True if the orchestrator accepted it."""
        try:
            if gesture == Gesture.STOP:
                self.orch.command_hold()
                return True
            if gesture == Gesture.LAND:
                self.orch.command_land()
                return True
            if gesture == Gesture.COME:
                self.orch.command_resume()
                return True
            if gesture == Gesture.RECEDE:
                # Until a "reverse cruise" mode exists in the planner,
                # treat RECEDE as a HOLD — operator-initiated stop while
                # the next direction is being figured out.
                self.orch.command_hold()
                return True
        except Exception as e:
            # Don't let one bad dispatch crash the whole autonomy loop.
            print(f"[gesture-action] dispatch of {gesture.name} failed: {e}")
            return False
        return False

    # ---- introspection ---- #

    @property
    def last_dispatched(self) -> Gesture:
        return self._last_gesture

    @property
    def seconds_since_last(self) -> float:
        if self._last_t == 0.0:
            return float("inf")
        return time.time() - self._last_t
