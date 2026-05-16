"""
scripts/autonomy/test_avoidance.py

Closed-loop autonomy test against ArduPilot SITL.

What this demonstrates:
    bridge.Vehicle  <->  ArduCopter SITL  (MAVLink)
         ^
         | velocity setpoints @ 10 Hz
         |
    planner.Planner  <--  perception_source.SyntheticPerceptionSource
                          (scripted obstacle injection)

End-to-end: a synthetic obstacle is injected at a fixed time, the planner
detects it and switches from CRUISING -> AVOIDING, the drone in SITL
laterally side-steps, the obstacle clears, the planner returns to CRUISING,
the drone resumes the original heading. RTL + land + disarm to close.

Pre-flight:
    1. SITL up:    scripts/sitl/run_sitl.sh
    2. Autonomy venv active

Run:
    source ~/Documents/AutonoBird/scripts/autonomy/venv/bin/activate
    python ~/Documents/AutonoBird/scripts/autonomy/test_avoidance.py

This closes the dissertation's T5 row at the SITL stage (§ 6.4 / § 6.6).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Cross-subsystem bridge import.
_AUTONOMY_DIR = Path(__file__).parent
_BRIDGE_DIR = (_AUTONOMY_DIR / ".." / "flight-controller").resolve()
if str(_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_DIR))

from bridge import BridgeError, Vehicle, load_config as load_bridge_config  # noqa: E402

from perception_source import (  # noqa: E402
    SyntheticEvent,
    SyntheticPerceptionSource,
    obstacle_ahead,
)
from planner import Planner, PlannerMode  # noqa: E402
from state_machine import FlightState, StateMachine  # noqa: E402


# ---------------------------------------------------------------------- #
# Test parameters                                                        #
# ---------------------------------------------------------------------- #

# Mission profile (seconds, from planner.start()):
#   0 ..  4   pre-flight settle (planner not yet running)
#   4 .. 12   CRUISING north
#  12 .. 20   obstacle injected, expect AVOIDING (sidestep east)
#  20 .. 28   obstacle clears, resume CRUISING
#  28         stop planner + RTL + land

CRUISE_ALT_M = 5.0
PLANNER_RATE_HZ = 10.0
CRUISE_SPEED_MS = 1.5
AVOIDANCE_SPEED_MS = 1.5
OBSTACLE_THRESHOLD_M = 1.5
CLEAR_THRESHOLD_M = 2.0

OBSTACLE_INJECT_T = 8.0    # seconds after planner.start()
OBSTACLE_CLEAR_T = 16.0
TOTAL_DURATION_S = 24.0

ARM_RETRIES = 4
ARM_RETRY_DELAY_S = 5.0


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon pairs in metres."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class PhaseSample:
    """One position sample, tagged with the planner mode active at the time."""

    t_offset: float       # seconds since planner.start()
    planner_mode: PlannerMode
    lat: float
    lon: float
    alt_rel: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"[+{self.t_offset:5.1f}s {self.planner_mode.name:9s}] "
            f"lat={self.lat:.6f} lon={self.lon:.6f} alt={self.alt_rel:.2f}m"
        )


@dataclass
class TestSummary:
    """Aggregated results from one closed-loop run."""

    samples: list[PhaseSample] = field(default_factory=list)
    mode_changes: list[tuple[float, PlannerMode, PlannerMode]] = field(default_factory=list)
    initial_lat: float = 0.0
    initial_lon: float = 0.0


def _phase_displacement(
    summary: TestSummary, mode: PlannerMode
) -> tuple[float, float]:
    """Net displacement (north_m, east_m) during a given planner mode.

    Computed from the first and last sample tagged with that mode.
    Returns (0, 0) if no sample is found in that mode.
    """
    rows = [s for s in summary.samples if s.planner_mode == mode]
    if len(rows) < 2:
        return 0.0, 0.0
    first, last = rows[0], rows[-1]
    north_m = haversine_m(first.lat, first.lon, last.lat, first.lon)
    if last.lat < first.lat:
        north_m = -north_m
    east_m = haversine_m(first.lat, first.lon, first.lat, last.lon)
    if last.lon < first.lon:
        east_m = -east_m
    return north_m, east_m


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def arm_with_retry(v: Vehicle) -> None:
    """Arm with retry; pre-arm 'Need Position Estimate' is a transient.

    The autopilot can take 10-30 s after first heartbeat for EKF to land a
    position estimate. We retry instead of failing the whole test.
    """
    last_err: Exception | None = None
    for i in range(ARM_RETRIES):
        try:
            v.arm()
            return
        except BridgeError as e:
            last_err = e
            log(f"  arm attempt {i + 1}/{ARM_RETRIES} failed: {e}")
            time.sleep(ARM_RETRY_DELAY_S)
    raise BridgeError(f"arm() failed after {ARM_RETRIES} attempts: {last_err}")


def run() -> int:
    cfg = load_bridge_config()
    log(f"Connecting bridge to {cfg['connection_uri']} ...")
    v = Vehicle(
        connection_uri=cfg["connection_uri"],
        source_system=cfg.get("source_system", 255),
        heartbeat_timeout=cfg.get("heartbeat_timeout", 30.0),
    )

    try:
        v.connect()
    except BridgeError as e:
        log(f"FAIL: bridge connect: {e}")
        log("Is SITL running? Try `scripts/sitl/run_sitl.sh` in another terminal.")
        return 1

    # FSM is observation-only here — useful for the log + a sanity check.
    fsm = StateMachine(v, poll_hz=10.0)
    fsm_transitions: list[tuple[float, FlightState]] = []

    def on_fsm(old: FlightState, new: FlightState, reason) -> None:
        fsm_transitions.append((time.time(), new))
        log(f"FSM: {old} -> {new}" + (f"  ({reason})" if reason else ""))

    fsm.subscribe(on_fsm)
    fsm.start()

    summary = TestSummary()

    try:
        log("Waiting for GPS fix + PREARMED ...")
        deadline = time.time() + 30
        while time.time() < deadline and fsm.state != FlightState.PREARMED:
            time.sleep(0.2)
        if fsm.state != FlightState.PREARMED:
            log(f"FAIL: never reached PREARMED (stuck in {fsm.state})")
            return 1

        log("Switching to GUIDED + arming ...")
        v.set_mode("GUIDED")
        arm_with_retry(v)
        log(f"  armed={v.state.armed}")

        log(f"Taking off to {CRUISE_ALT_M} m ...")
        v.takeoff(CRUISE_ALT_M)
        log(f"  reached alt={v.state.alt_rel:.2f} m")

        # Brief settle.
        time.sleep(2.0)

        # ------------------------------------------------------------ #
        # Build perception timeline + planner                          #
        # ------------------------------------------------------------ #

        events = [
            SyntheticEvent(0.0, []),
            SyntheticEvent(OBSTACLE_INJECT_T, [obstacle_ahead(0.8)]),
            SyntheticEvent(OBSTACLE_CLEAR_T, []),
        ]
        log(
            f"Synthetic perception: clear -> obstacle at t={OBSTACLE_INJECT_T}s "
            f"-> clear at t={OBSTACLE_CLEAR_T}s"
        )

        src = SyntheticPerceptionSource(events, rate_hz=PLANNER_RATE_HZ)
        plan = Planner(
            vehicle=v,
            perception=src,
            cruise_speed_ms=CRUISE_SPEED_MS,
            avoidance_speed_ms=AVOIDANCE_SPEED_MS,
            obstacle_threshold_m=OBSTACLE_THRESHOLD_M,
            clear_threshold_m=CLEAR_THRESHOLD_M,
            rate_hz=PLANNER_RATE_HZ,
        )

        def on_planner(old: PlannerMode, new: PlannerMode) -> None:
            offset = time.time() - planner_t0
            summary.mode_changes.append((offset, old, new))
            log(f"Planner: {old} -> {new}  (+{offset:.2f}s)")

        plan.subscribe(on_planner)

        # ------------------------------------------------------------ #
        # Run the closed loop                                          #
        # ------------------------------------------------------------ #

        s = v.state
        summary.initial_lat = s.lat or 0.0
        summary.initial_lon = s.lon or 0.0
        log(f"Initial position: lat={summary.initial_lat:.6f} lon={summary.initial_lon:.6f}")

        log(f"Running closed loop for {TOTAL_DURATION_S} s ...")
        src.start()
        planner_t0 = time.time()
        plan.start()

        # Sample telemetry every 0.5 s during the run.
        sample_period = 0.5
        next_sample = planner_t0
        while time.time() < planner_t0 + TOTAL_DURATION_S:
            now = time.time()
            if now >= next_sample:
                state = v.state
                if state.lat is not None and state.lon is not None and state.alt_rel is not None:
                    summary.samples.append(
                        PhaseSample(
                            t_offset=now - planner_t0,
                            planner_mode=plan.mode,
                            lat=state.lat,
                            lon=state.lon,
                            alt_rel=state.alt_rel,
                        )
                    )
                next_sample = now + sample_period
            time.sleep(0.05)

        log("Closed loop done. Stopping planner + perception source ...")
        plan.stop()
        src.stop()

        # ------------------------------------------------------------ #
        # Return to launch                                             #
        # ------------------------------------------------------------ #

        log("Switching to RTL ...")
        v.rtl()
        log("Waiting for disarm (up to 90 s) ...")
        deadline = time.time() + 90
        while time.time() < deadline and v.state.armed:
            time.sleep(0.5)
        if v.state.armed:
            log("WARN: did not disarm within 90 s (test result still valid)")
        else:
            log(f"  disarmed. final state: mode={v.state.mode} alt={v.state.alt_rel}")

        # ------------------------------------------------------------ #
        # Analysis                                                     #
        # ------------------------------------------------------------ #

        print()
        log("=" * 64)
        log("Per-mode displacement summary")
        log("=" * 64)
        for mode in (PlannerMode.CRUISING, PlannerMode.AVOIDING):
            north_m, east_m = _phase_displacement(summary, mode)
            log(
                f"  {mode.name:9s}  north={north_m:+6.2f}m  east={east_m:+6.2f}m  "
                f"|disp|={math.hypot(north_m, east_m):5.2f}m"
            )

        # Validation criteria.
        cruise_north, _ = _phase_displacement(summary, PlannerMode.CRUISING)
        _, avoid_east = _phase_displacement(summary, PlannerMode.AVOIDING)

        cruise_ok = cruise_north > 1.5   # moved at least 1.5m north during cruise
        avoid_ok = abs(avoid_east) > 1.0  # moved at least 1m laterally during avoid

        # The planner should have seen at least: IDLE->CRUISING, CRUISING->AVOIDING,
        # AVOIDING->CRUISING (back), CRUISING->IDLE (on stop)
        modes_visited = {mc[2] for mc in summary.mode_changes}
        planner_ok = {
            PlannerMode.CRUISING,
            PlannerMode.AVOIDING,
        }.issubset(modes_visited)

        print()
        log(f"  CRUISING net-north > 1.5 m   :  {'PASS' if cruise_ok else 'FAIL'}  ({cruise_north:+.2f})")
        log(f"  AVOIDING net-east  > 1.0 m   :  {'PASS' if avoid_ok else 'FAIL'}  ({avoid_east:+.2f})")
        log(f"  Planner visited CRUISING+AVOIDING: {'PASS' if planner_ok else 'FAIL'}")
        if not (cruise_ok and avoid_ok and planner_ok):
            log("FAIL: closed-loop avoidance test did not meet acceptance criteria.")
            return 1
        log("PASS: closed-loop avoidance demonstrated end-to-end.")
        return 0

    except BridgeError as e:
        log(f"FAIL: bridge error: {e}")
        return 1
    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130
    finally:
        log("Stopping FSM + disconnecting bridge ...")
        fsm.stop()
        v.disconnect()


if __name__ == "__main__":
    sys.exit(run())
