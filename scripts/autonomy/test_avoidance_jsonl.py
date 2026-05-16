"""
scripts/autonomy/test_avoidance_jsonl.py

Closed-loop avoidance test driven by a recorded (or synthetic) JSONL
detection file, rather than the in-process SyntheticPerceptionSource
used by test_avoidance.py.

Same flight pattern, same validation criteria; the difference is the
perception transport — the planner consumes the same data shape, but
it arrives from a file written by `scripts/perception/depth_detect.py`
(real run) or by `make_synthetic_jsonl.py` (offline scaffolding).

This validates the dissertation's T5 collision-avoidance scenario with
**real-perception input** (or with a synthetic JSONL that exercises
the same file-tail pipeline). It is the closest SITL-track demo to the
real-flight scenario.

Pre-flight:
    1. SITL is up:  scripts/sitl/run_sitl.sh
    2. JSONL file is available. Two ways:
        a. Real capture on the Pi rig:
             python scripts/perception/depth_detect.py --jsonl /tmp/session.jsonl --no-gui
             (then copy the file to the dev workstation if needed)
        b. Synthetic:
             python scripts/autonomy/make_synthetic_jsonl.py --out /tmp/session.jsonl

Run:
    source ~/Documents/AutonoBird/scripts/autonomy/venv/bin/activate
    python test_avoidance_jsonl.py --jsonl /tmp/session.jsonl
"""

from __future__ import annotations

import argparse
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

from perception_source import DepthDetectSource  # noqa: E402
from planner import Planner, PlannerMode  # noqa: E402
from state_machine import FlightState, StateMachine  # noqa: E402


# Test parameters — kept identical to test_avoidance.py so results are
# comparable between the synthetic-source and JSONL-source paths.
CRUISE_ALT_M = 5.0
PLANNER_RATE_HZ = 10.0
CRUISE_SPEED_MS = 1.5
AVOIDANCE_SPEED_MS = 1.5
OBSTACLE_THRESHOLD_M = 1.5
CLEAR_THRESHOLD_M = 2.0

TOTAL_DURATION_S = 24.0

ARM_RETRIES = 4
ARM_RETRY_DELAY_S = 5.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class PhaseSample:
    t_offset: float
    planner_mode: PlannerMode
    lat: float
    lon: float
    alt_rel: float


@dataclass
class TestSummary:
    samples: list[PhaseSample] = field(default_factory=list)
    mode_changes: list[tuple[float, PlannerMode, PlannerMode]] = field(default_factory=list)


def _phase_displacement(
    summary: TestSummary, mode: PlannerMode
) -> tuple[float, float]:
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


def arm_with_retry(v: Vehicle) -> None:
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    ap.add_argument(
        "--jsonl", required=True,
        help="Path to JSONL file from depth_detect.py --jsonl or make_synthetic_jsonl.py",
    )
    ap.add_argument(
        "--replay-speed", type=float, default=1.0,
        help="JSONL replay speed (default 1.0 = real-time). 0 = as-fast-as-possible.",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl).expanduser()
    if not jsonl_path.is_file():
        log(f"FAIL: JSONL file not found: {jsonl_path}")
        log("Generate one with:  python make_synthetic_jsonl.py --out /tmp/session.jsonl")
        return 1
    log(f"Using JSONL: {jsonl_path} (replay_speed={args.replay_speed})")

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

    fsm = StateMachine(v, poll_hz=10.0)

    def on_fsm(old: FlightState, new: FlightState, reason) -> None:
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

        log(f"Taking off to {CRUISE_ALT_M} m ...")
        v.takeoff(CRUISE_ALT_M)
        log(f"  reached alt={v.state.alt_rel:.2f} m")
        time.sleep(2.0)

        # Build perception source from the JSONL.
        src = DepthDetectSource(
            jsonl_path=str(jsonl_path),
            tail_from_end=False,
            replay_speed=args.replay_speed,
        )
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

        s = v.state
        initial_lat = s.lat or 0.0
        initial_lon = s.lon or 0.0
        log(f"Initial position: lat={initial_lat:.6f} lon={initial_lon:.6f}")

        log(f"Running closed loop for {TOTAL_DURATION_S} s (replay paced by JSONL) ...")
        src.start()
        planner_t0 = time.time()
        plan.start()

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

        # Analysis (same shape as test_avoidance.py for cross-comparison)
        print()
        log("=" * 64)
        log("Per-mode displacement summary (JSONL replay)")
        log("=" * 64)
        for mode in (PlannerMode.CRUISING, PlannerMode.AVOIDING):
            north_m, east_m = _phase_displacement(summary, mode)
            log(
                f"  {mode.name:9s}  north={north_m:+6.2f}m  east={east_m:+6.2f}m  "
                f"|disp|={math.hypot(north_m, east_m):5.2f}m"
            )

        cruise_north, _ = _phase_displacement(summary, PlannerMode.CRUISING)
        _, avoid_east = _phase_displacement(summary, PlannerMode.AVOIDING)

        cruise_ok = cruise_north > 1.5
        avoid_ok = abs(avoid_east) > 1.0
        modes_visited = {mc[2] for mc in summary.mode_changes}
        planner_ok = {PlannerMode.CRUISING, PlannerMode.AVOIDING}.issubset(modes_visited)

        print()
        log(f"  CRUISING net-north > 1.5 m   :  {'PASS' if cruise_ok else 'FAIL'}  ({cruise_north:+.2f})")
        log(f"  AVOIDING net-east  > 1.0 m   :  {'PASS' if avoid_ok else 'FAIL'}  ({avoid_east:+.2f})")
        log(f"  Planner visited CRUISING+AVOIDING: {'PASS' if planner_ok else 'FAIL'}")

        if planner_ok and cruise_ok and avoid_ok:
            log("PASS: JSONL-driven closed-loop avoidance demonstrated end-to-end.")
            return 0

        # If only CRUISING was visited (no obstacle ever close enough),
        # report that clearly — the user may have given a JSONL file that
        # never had a sub-threshold detection.
        if not planner_ok:
            log(
                "NOTE: planner never entered AVOIDING — check the JSONL contains "
                "detections with depth_m <= obstacle_threshold_m."
            )
        log("FAIL: acceptance criteria not met (see notes above).")
        return 1

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
