"""
scripts/autonomy/state_machine.py

High-level flight state machine for AutonoBird.

A passive observer that consumes the bridge's vehicle state + event stream
and infers one of a small set of high-level flight states. Other autonomy
modules (planner, orchestrator), the LED status driver, and the voice /
gesture command mappers subscribe to state changes to react.

The FSM is observation-only: it does NOT issue MAVLink commands. Callers
that want to drive the vehicle use the bridge's Vehicle API directly and
let the FSM follow. This keeps the FSM testable in isolation and avoids
the chicken-and-egg "FSM commands MAVLink commands change state which
affects FSM" feedback loop.

States (an enum, not strings — explicit transitions, no typos):

    DISCONNECTED         -- vehicle not connected (transport down)
    NO_FIX               -- connected, awaiting GPS 3D fix
    PREARMED             -- GPS ok, not armed, ready to arm
    ARMED_ON_GROUND      -- armed, alt < 0.5 m, not climbing
    ASCENDING            -- armed, climbing (alt-rate > 0.2 m/s)
    AIRBORNE             -- armed, alt >= 0.5 m, |alt-rate| < 0.2 m/s
    DESCENDING           -- armed, descending (alt-rate < -0.2 m/s)
    DISARMED_POSTFLIGHT  -- disarmed after a flight session, alt < 0.5 m
    FAULT                -- battery / EKF / crash failsafe; terminal until reset()

Transitions are inferred from a combination of:

- vehicle.state (the snapshot from the bridge: armed, alt_rel, gps_fix)
- vehicle.drain_events() / events queue (mode changes, STATUSTEXT, item_reached)

The FSM polls at `poll_hz` (default 10 Hz) — fine-grained enough to catch
the takeoff transient (sub-second) without hammering the GIL. Vehicle
telemetry is the actual rate-limit at ~4–7 Hz.

Threading: start() launches a daemon thread. stop() joins it.
Subscribers are called synchronously from the FSM thread; keep them
short or push work to your own queue.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

# ----------------------------------------------------------------------- #
# Cross-subsystem import: pull bridge.Vehicle from ../flight-controller   #
# ----------------------------------------------------------------------- #
_AUTONOMY_DIR = Path(__file__).parent
_BRIDGE_DIR = (_AUTONOMY_DIR / ".." / "flight-controller").resolve()
if str(_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_DIR))

from bridge import Vehicle, VehicleEvent  # noqa: E402


# ----------------------------------------------------------------------- #
# State definitions                                                       #
# ----------------------------------------------------------------------- #


class FlightState(Enum):
    DISCONNECTED = auto()
    NO_FIX = auto()
    PREARMED = auto()
    ARMED_ON_GROUND = auto()
    ASCENDING = auto()
    AIRBORNE = auto()
    DESCENDING = auto()
    DISARMED_POSTFLIGHT = auto()
    FAULT = auto()

    def __str__(self) -> str:
        return self.name


# Keywords in STATUSTEXT that trigger a FAULT transition. ArduPilot emits
# these on its serious failsafes; we treat them as terminal until the
# operator explicitly resets the FSM.
_FAULT_KEYWORDS = (
    "battery failsafe",
    "ekf failsafe",
    "crash: disarming",
    "vibration compensation",
    "gps glitch",
    "radio failsafe",
)

# Below this altitude the vehicle is treated as "on ground" for the
# purpose of distinguishing armed-on-ground vs airborne.
_GROUND_ALT_M = 0.5

# Climb-rate thresholds (m/s, +up). Hysteresis avoids flapping between
# ASCENDING/AIRBORNE/DESCENDING when the autopilot's filtered climb-rate
# oscillates around zero near hover.
#
# To ENTER ASCENDING/DESCENDING the rate must exceed the wide threshold;
# to LEAVE back to AIRBORNE the rate must fall below the narrow one. This
# is a Schmitt-trigger pattern. Together with the autopilot's own
# pre-filtered climb signal (VFR_HUD's `climb` field, not a poll-based
# derivative), this kills the strobing observed at 0.2 m/s.
_CLIMB_RATE_ENTER_MS = 0.5
_CLIMB_RATE_EXIT_MS = 0.15


@dataclass
class Transition:
    """One state change in the FSM's history."""

    t: float
    from_state: FlightState
    to_state: FlightState
    reason: Optional[str] = None

    def __repr__(self) -> str:
        r = f" reason={self.reason!r}" if self.reason else ""
        return f"{self.from_state} -> {self.to_state}{r}"


Subscriber = Callable[[FlightState, FlightState, Optional[str]], None]


# ----------------------------------------------------------------------- #
# State machine                                                           #
# ----------------------------------------------------------------------- #


class StateMachine:
    """Passive flight-state inferrer driven off the bridge's `Vehicle`."""

    def __init__(self, vehicle: Vehicle, poll_hz: float = 10.0):
        if poll_hz <= 0:
            raise ValueError("poll_hz must be > 0")
        self.vehicle = vehicle
        self.poll_period = 1.0 / poll_hz

        self._state = FlightState.DISCONNECTED
        self._history: list[Transition] = []
        self._fault_reason: Optional[str] = None

        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the observer thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._loop, name="fsm-observer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the observer thread. Idempotent."""
        self._stop_flag.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "StateMachine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Public state access                                                #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> FlightState:
        with self._lock:
            return self._state

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    @property
    def history(self) -> list[Transition]:
        with self._lock:
            return list(self._history)

    def subscribe(self, callback: Subscriber) -> None:
        """Register a callback invoked on every state change.

        Signature: callback(old: FlightState, new: FlightState, reason: str|None)
        Callbacks run synchronously on the FSM thread — keep them short
        or hand off to your own queue.
        """
        with self._lock:
            self._subscribers.append(callback)

    def reset(self) -> None:
        """Re-evaluate state from current vehicle telemetry.

        Clears FAULT and forces a fresh inference. Use after the operator
        has acknowledged the fault condition and verified it has cleared.
        """
        with self._lock:
            self._fault_reason = None
            self._state = FlightState.DISCONNECTED

    # ------------------------------------------------------------------ #
    # Internal loop                                                      #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                self._tick()
            except Exception as e:  # never let an exception kill the FSM
                self._fail_open(f"FSM tick error: {e}")
            self._stop_flag.wait(self.poll_period)

    def _tick(self) -> None:
        # Drain events first — these have priority over polled inference,
        # because mode changes / STATUSTEXT are authoritative.
        for evt in self.vehicle.drain_events():
            self._handle_event(evt)

        if self._state == FlightState.FAULT:
            return  # terminal — operator must call reset()

        self._infer_from_state()

    def _handle_event(self, evt: VehicleEvent) -> None:
        if evt.kind == "status":
            text = (evt.payload or "").lower()
            for kw in _FAULT_KEYWORDS:
                if kw in text:
                    self._transition(FlightState.FAULT, reason=evt.payload)
                    return
        # mode / arm events are reflected in vehicle.state and picked up by
        # _infer_from_state — no special handling needed here.

    def _infer_from_state(self) -> None:
        s = self.vehicle.state

        if not s.connected:
            self._transition(FlightState.DISCONNECTED)
            return

        gps_ok = s.gps_fix is not None and s.gps_fix >= 3
        if not gps_ok:
            self._transition(FlightState.NO_FIX)
            return

        alt = s.alt_rel if s.alt_rel is not None else 0.0
        # Prefer the autopilot's filtered VFR_HUD climb (m/s, +up). The
        # bridge already populates state.climb_rate from VFR_HUD; this is
        # smoother than differentiating polled altitude ourselves.
        climb = s.climb_rate
        on_ground = alt < _GROUND_ALT_M

        if s.armed:
            if on_ground:
                # Either just armed or just landed but still armed (rare).
                # If we came from a flight state, this is a touchdown
                # transient before the autopilot disarms.
                self._transition(FlightState.ARMED_ON_GROUND)
                return

            if climb is None:
                # No climb signal yet (VFR_HUD not received). Treat as
                # AIRBORNE rather than flapping.
                self._transition(FlightState.AIRBORNE)
                return

            # Schmitt-trigger hysteresis on climb_rate to suppress flap
            # at the AIRBORNE boundary. The enter and exit thresholds are
            # different; the current FSM state determines which one applies.
            if self._state == FlightState.ASCENDING:
                # Already ascending. Only drop to AIRBORNE when rate falls
                # below the narrow exit threshold.
                if climb < _CLIMB_RATE_EXIT_MS:
                    self._transition(FlightState.AIRBORNE)
                # else stay in ASCENDING
            elif self._state == FlightState.DESCENDING:
                # Already descending. Only drop to AIRBORNE when rate
                # rises above the negative narrow exit threshold.
                if climb > -_CLIMB_RATE_EXIT_MS:
                    self._transition(FlightState.AIRBORNE)
                # else stay in DESCENDING
            else:
                # Currently AIRBORNE / ARMED_ON_GROUND / other. Need
                # sustained climb beyond the wide enter threshold to
                # leave the AIRBORNE band.
                if climb > _CLIMB_RATE_ENTER_MS:
                    self._transition(FlightState.ASCENDING)
                elif climb < -_CLIMB_RATE_ENTER_MS:
                    self._transition(FlightState.DESCENDING)
                else:
                    self._transition(FlightState.AIRBORNE)
        else:
            # Disarmed
            if self._state in (
                FlightState.ASCENDING,
                FlightState.AIRBORNE,
                FlightState.DESCENDING,
                FlightState.ARMED_ON_GROUND,
            ):
                self._transition(FlightState.DISARMED_POSTFLIGHT)
            else:
                self._transition(FlightState.PREARMED)

    def _transition(
        self, new_state: FlightState, reason: Optional[str] = None
    ) -> None:
        with self._lock:
            old = self._state
            if old == new_state:
                return
            self._state = new_state
            if new_state == FlightState.FAULT and reason is not None:
                self._fault_reason = reason
            self._history.append(Transition(time.time(), old, new_state, reason))
            subscribers = list(self._subscribers)

        # Fire callbacks outside the lock to avoid deadlock if a callback
        # ever calls back into the FSM.
        for cb in subscribers:
            try:
                cb(old, new_state, reason)
            except Exception:
                # Subscribers must not be able to take down the FSM.
                pass

    def _fail_open(self, reason: str) -> None:
        """Internal fault: something in the FSM itself broke."""
        self._transition(FlightState.FAULT, reason=reason)
