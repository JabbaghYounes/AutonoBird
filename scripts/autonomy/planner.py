"""
scripts/autonomy/planner.py

Reactive obstacle-avoidance planner.

The planner runs in a background thread at a fixed rate (default 10 Hz),
reading the latest `PerceptionFrame` from a `PerceptionSource` and
emitting velocity setpoints to the autopilot via the bridge's
`Vehicle.send_velocity_ned()` API.

This is a **reactive** controller, not a path planner with a global goal.
It cruises in a commanded forward direction; when perception reports an
obstacle within `obstacle_threshold_m`, it switches to a sidestep mode
until the obstacle is no longer in range. Hysteresis avoids flapping at
the obstacle-threshold boundary.

A future global planner (A* / RRT*) would sit above this one — consume
a waypoint goal, plan a coarse route, and feed leg-by-leg cruise
directions to the reactive layer. That's recorded in the engineering
backlog under "Path planner consuming depth + detections".

Coordinate-frame assumption (initial implementation):
- The planner emits velocities in the **local NED frame**: vx = north,
  vy = east, vz = down (positive = downward).
- For this assumption to be physically meaningful, the drone must be
  yawed approximately to the cruise direction. The ArduPilot SITL
  default home (Jerrabomberra) starts with `hdg=353` (≈north), so cruise
  in the +north direction is correct out of the box. For arbitrary yaw,
  a body-frame velocity setpoint or a yaw-rotation of the NED vector
  is needed — recorded as a follow-up.

Modes:
    IDLE       — planner not running; emits nothing
    CRUISING   — forward velocity at `cruise_speed_ms`
    AVOIDING   — lateral sidestep, direction chosen by obstacle position
    LANDING    — terminal: emits zero velocity, ready to be stopped

State transitions (per tick):
    IDLE     --start()-->        CRUISING
    CRUISING --obstacle <T_obs--> AVOIDING
    AVOIDING --no obstacle <T_clear--> CRUISING
    any      --pause()/stop()-->  IDLE
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

# bridge.Vehicle is imported lazily by callers — the planner's type
# annotations use a string so this module doesn't have to do the
# sys.path dance itself. Test code (test_avoidance.py) constructs the
# Vehicle and passes it in.

from perception_source import Detection, PerceptionFrame, PerceptionSource


# ---------------------------------------------------------------------- #
# State + telemetry data classes                                         #
# ---------------------------------------------------------------------- #


class PlannerMode(Enum):
    IDLE = auto()
    CRUISING = auto()
    AVOIDING = auto()
    LANDING = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class PlannerTelemetry:
    """What the planner is doing right now. Snapshot for observers."""

    mode: PlannerMode
    vx: float
    vy: float
    vz: float
    closest_obstacle_m: Optional[float]
    closest_obstacle_centroid_x: Optional[float]
    ticks: int  # number of control cycles executed

    def __repr__(self) -> str:
        co = (
            f"{self.closest_obstacle_m:.2f}m@{self.closest_obstacle_centroid_x:+.2f}"
            if self.closest_obstacle_m is not None
            else "-"
        )
        return (
            f"Planner({self.mode} v=({self.vx:+.2f},{self.vy:+.2f},{self.vz:+.2f}) "
            f"obs={co} ticks={self.ticks})"
        )


# Subscriber callback signature
ModeChangeCallback = Callable[[PlannerMode, PlannerMode], None]


# ---------------------------------------------------------------------- #
# Planner                                                                #
# ---------------------------------------------------------------------- #


class Planner:
    """Reactive avoider. Bridges PerceptionSource -> Vehicle.send_velocity_ned."""

    def __init__(
        self,
        vehicle: "object",  # bridge.Vehicle, kept loose to avoid the import here
        perception: PerceptionSource,
        cruise_speed_ms: float = 1.0,
        avoidance_speed_ms: float = 1.0,
        obstacle_threshold_m: float = 1.5,
        clear_threshold_m: float = 2.0,
        rate_hz: float = 10.0,
        min_obstacle_confidence: float = 0.3,
    ):
        if clear_threshold_m <= obstacle_threshold_m:
            raise ValueError(
                "clear_threshold_m must be > obstacle_threshold_m (hysteresis)"
            )
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")

        self.vehicle = vehicle
        self.perception = perception

        self.cruise_speed_ms = float(cruise_speed_ms)
        self.avoidance_speed_ms = float(avoidance_speed_ms)
        self.obstacle_threshold_m = float(obstacle_threshold_m)
        self.clear_threshold_m = float(clear_threshold_m)
        self.min_obstacle_confidence = float(min_obstacle_confidence)
        self._period = 1.0 / rate_hz

        self._mode = PlannerMode.IDLE
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Last commanded velocity (for telemetry / debug).
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_vz = 0.0
        self._ticks = 0
        self._last_obstacle: Optional[Detection] = None
        # Avoidance direction sign for vy: +1 = east, -1 = west.
        # Latched on entry to AVOIDING to prevent left-right oscillation
        # when an obstacle's centroid hovers near 0.
        self._avoidance_sign: int = +1

        self._subscribers: list[ModeChangeCallback] = []

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Begin the control loop. Transitions IDLE -> CRUISING."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._set_mode(PlannerMode.CRUISING)
        self._thread = threading.Thread(
            target=self._loop, name="planner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the control loop. Idempotent."""
        self._stop_flag.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._set_mode(PlannerMode.IDLE)

    def pause(self) -> None:
        """Pause emission (vehicle keeps its last commanded velocity briefly,
        then ArduCopter falls back to no-target).
        """
        self._set_mode(PlannerMode.IDLE)

    def resume(self) -> None:
        """Resume CRUISING after a pause()."""
        if self._mode == PlannerMode.IDLE:
            self._set_mode(PlannerMode.CRUISING)

    def land(self) -> None:
        """Mark LANDING; planner will emit zero velocity until stopped.

        The actual mode change (e.g. switching to ArduPilot LAND/RTL) is
        the orchestrator's responsibility — the planner just stops pushing
        offboard targets.
        """
        self._set_mode(PlannerMode.LANDING)

    def subscribe(self, cb: ModeChangeCallback) -> None:
        with self._lock:
            self._subscribers.append(cb)

    def __enter__(self) -> "Planner":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # State access                                                       #
    # ------------------------------------------------------------------ #

    @property
    def mode(self) -> PlannerMode:
        with self._lock:
            return self._mode

    @property
    def telemetry(self) -> PlannerTelemetry:
        with self._lock:
            obs_depth = self._last_obstacle.depth_m if self._last_obstacle else None
            obs_cx = (
                self._last_obstacle.bbox_centroid_norm[0]
                if self._last_obstacle
                else None
            )
            return PlannerTelemetry(
                mode=self._mode,
                vx=self._last_vx,
                vy=self._last_vy,
                vz=self._last_vz,
                closest_obstacle_m=obs_depth,
                closest_obstacle_centroid_x=obs_cx,
                ticks=self._ticks,
            )

    # ------------------------------------------------------------------ #
    # Control loop                                                       #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                self._tick()
            except Exception:
                # Never let an exception kill the planner — the vehicle
                # falls back to ArduCopter's no-offboard-target behaviour
                # if we stop publishing, which is safer than crashing.
                pass
            self._stop_flag.wait(self._period)

    def _tick(self) -> None:
        frame = self.perception.latest()
        obstacle = self._select_obstacle(frame)

        # Update mode based on obstacle presence (hysteresis).
        if self._mode == PlannerMode.CRUISING:
            if obstacle is not None and obstacle.depth_m <= self.obstacle_threshold_m:
                # Latch sidestep direction. If obstacle is centred-ish
                # (|centroid_x| < 0.1), default to sidestep east (+vy).
                cx = obstacle.bbox_centroid_norm[0]
                if cx > 0.1:
                    self._avoidance_sign = -1   # obstacle on right -> sidestep left (west)
                elif cx < -0.1:
                    self._avoidance_sign = +1   # obstacle on left -> sidestep right (east)
                else:
                    self._avoidance_sign = +1   # centred -> arbitrary, default east
                self._last_obstacle = obstacle
                self._set_mode(PlannerMode.AVOIDING)
        elif self._mode == PlannerMode.AVOIDING:
            # Only return to CRUISING if no obstacle within the clear-band.
            if obstacle is None or obstacle.depth_m > self.clear_threshold_m:
                self._last_obstacle = None
                self._set_mode(PlannerMode.CRUISING)
            else:
                # Keep tracking the obstacle for telemetry.
                self._last_obstacle = obstacle

        # Emit velocity for the current mode.
        if self._mode == PlannerMode.CRUISING:
            self._command(self.cruise_speed_ms, 0.0, 0.0)
        elif self._mode == PlannerMode.AVOIDING:
            # No forward motion while avoiding — pure lateral. Could be
            # blended (e.g. half-forward + lateral) later.
            self._command(0.0, self._avoidance_sign * self.avoidance_speed_ms, 0.0)
        elif self._mode == PlannerMode.LANDING:
            self._command(0.0, 0.0, 0.0)
        else:  # IDLE
            return

        with self._lock:
            self._ticks += 1

    def _select_obstacle(self, frame: Optional[PerceptionFrame]) -> Optional[Detection]:
        """Pick the closest detection that passes the confidence filter."""
        if frame is None:
            return None
        candidates = [
            d for d in frame.detections if d.confidence >= self.min_obstacle_confidence
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda d: d.depth_m)

    def _command(self, vx: float, vy: float, vz: float) -> None:
        try:
            self.vehicle.send_velocity_ned(vx, vy, vz)  # type: ignore[attr-defined]
        except Exception:
            # Transport errors get swallowed for the same reason as
            # _loop's catch — keep the planner alive.
            return
        with self._lock:
            self._last_vx = vx
            self._last_vy = vy
            self._last_vz = vz

    def _set_mode(self, new_mode: PlannerMode) -> None:
        with self._lock:
            old = self._mode
            if old == new_mode:
                return
            self._mode = new_mode
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(old, new_mode)
            except Exception:
                pass


# ---------------------------------------------------------------------- #
# Standalone smoke-test                                                  #
# ---------------------------------------------------------------------- #


class _NullVehicle:
    """Vehicle stand-in for offline planner testing. Just logs commands."""

    def __init__(self) -> None:
        self.commands: list[tuple[float, float, float]] = []

    def send_velocity_ned(self, vx: float, vy: float, vz: float) -> None:
        self.commands.append((vx, vy, vz))


def _self_test() -> None:
    """No-vehicle, no-SITL test: confirm the planner reacts correctly to
    a scripted synthetic perception timeline. Prints transitions and a
    sample of issued commands."""
    from perception_source import (
        SyntheticEvent,
        SyntheticPerceptionSource,
        obstacle_offset,
    )

    events = [
        SyntheticEvent(0.0, []),                                      # clear
        SyntheticEvent(1.0, [obstacle_offset(0.8, centroid_x_norm=+0.4)]),  # obstacle on right
        SyntheticEvent(2.0, []),                                      # clear
        SyntheticEvent(3.0, [obstacle_offset(0.6, centroid_x_norm=-0.5)]),  # obstacle on left
        SyntheticEvent(4.0, []),                                      # clear
    ]
    src = SyntheticPerceptionSource(events, rate_hz=10.0)
    veh = _NullVehicle()
    plan = Planner(veh, src, rate_hz=10.0)

    def on_mode(old: PlannerMode, new: PlannerMode) -> None:
        print(f"  planner: {old} -> {new}")

    plan.subscribe(on_mode)

    print("Starting synthetic 5s replay ...")
    src.start()
    plan.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        t = plan.telemetry
        print(f"  +{time.time() - src._start_t:.2f}s  {t}")
        time.sleep(0.5)
    plan.stop()
    src.stop()
    print(f"Total commands sent: {len(veh.commands)}")
    sample = veh.commands[::max(1, len(veh.commands) // 10)]
    for c in sample:
        print(f"    v=({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})")


if __name__ == "__main__":
    _self_test()
