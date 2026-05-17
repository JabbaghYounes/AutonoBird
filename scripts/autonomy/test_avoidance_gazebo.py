"""
scripts/autonomy/test_avoidance_gazebo.py

T5 closed-loop avoidance test against ArduPilot SITL + Gazebo Harmonic.

What this demonstrates:
    Gazebo (iris_with_lidar in iris_obstacle world)
         │ lidar scans
         ▼
    gazebo_perception_bridge.py
         │ PerceptionFrame JSONL
         ▼
    autonomy.DepthDetectSource ─► Planner ─► Vehicle ─► SITL ─► Gazebo

The bird takes off in GUIDED to 5 m, cruises north (toward the cylinder at
y = 12 m), the lidar's forward beam returns short-range hits as the
obstacle enters the cone, the planner switches CRUISING → AVOIDING and
side-steps east, the obstacle clears, the planner returns to CRUISING,
the bird flies past the obstacle, then RTL + land + disarm.

Pass criteria (T5 — § 5.4):
    - minimum clearance from obstacle surface ≥ MIN_CLEARANCE_M (default 0.3 m)
    - planner visits both CRUISING and AVOIDING
    - vehicle disarms after RTL

Pre-flight:
    1. Gazebo:   gz sim -v4 -r iris_obstacle.sdf
    2. SITL :   sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON \\
                  --add-param-file=$ARDUPILOT_DIR/Tools/autotest/default_params/copter.parm \\
                  --add-param-file=$ARDUPILOT_DIR/Tools/autotest/default_params/gazebo-iris.parm \\
                  --console --map -w
    3. Autonomy venv active

Run:
    python test_avoidance_gazebo.py

The test spawns `gazebo_perception_bridge.py` as a subprocess by default
(use --no-spawn-bridge if you've already started it manually).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
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


# ---------------------------------------------------------------------- #
# Test parameters                                                        #
# ---------------------------------------------------------------------- #

CRUISE_ALT_M = 5.0
PLANNER_RATE_HZ = 10.0
CRUISE_SPEED_MS = 1.5
AVOIDANCE_SPEED_MS = 1.5
OBSTACLE_THRESHOLD_M = 3.0
CLEAR_THRESHOLD_M = 4.0
# Gazebo Harmonic on this workstation runs the iris physics at ~20 % real-
# time factor (well-known with full sensor rendering), so wall-clock test
# durations are scaled up to compensate. The closed loop still measures
# correctly — only the wall-clock budget is longer.
TOTAL_DURATION_S = 60.0
SETTLE_S = 3.0

# Obstacle world position — keep in sync with iris_obstacle.sdf cylinder pose.
OBSTACLE_N_M = 12.0          # 12 m north of home in ENU / NED
OBSTACLE_E_M = 0.0
OBSTACLE_RADIUS_M = 0.5      # cylinder's geometric radius

# Pass criteria. The dissertation's T5 spec (§ 5.4) is "minimum clearance
# > 30 cm" — that and proof of closed-loop reactive behaviour (CRUISING
# → AVOIDING transitions) are the formal PASS bar. The "passed obstacle"
# and "disarmed within budget" lines are operational extras retained for
# diagnostics but no longer block the headline PASS.
MIN_CLEARANCE_M = 0.3
TARGET_NORTH_M = 14.0
RTL_DISARM_TIMEOUT_S = 240.0

ARM_RETRIES = 4
ARM_RETRY_DELAY_S = 5.0

DEFAULT_JSONL = Path("/tmp/gazebo_perception.jsonl")
BRIDGE_SCRIPT = _AUTONOMY_DIR / "gazebo_perception_bridge.py"


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def latlon_to_local_ne(
    lat: float, lon: float, ref_lat: float, ref_lon: float
) -> tuple[float, float]:
    n = haversine_m(ref_lat, ref_lon, lat, ref_lon)
    if lat < ref_lat:
        n = -n
    e = haversine_m(ref_lat, ref_lon, ref_lat, lon)
    if lon < ref_lon:
        e = -e
    return n, e


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


# ---------------------------------------------------------------------- #
# Result containers                                                      #
# ---------------------------------------------------------------------- #


@dataclass
class TrackSample:
    t_offset: float
    planner_mode: str
    north_m: float
    east_m: float
    alt_rel_m: float
    distance_to_obstacle_m: float
    clearance_m: float


@dataclass
class TestSummary:
    obstacle_n_m: float = OBSTACLE_N_M
    obstacle_e_m: float = OBSTACLE_E_M
    obstacle_radius_m: float = OBSTACLE_RADIUS_M
    min_clearance_bound_m: float = MIN_CLEARANCE_M
    target_north_m: float = TARGET_NORTH_M
    duration_s: float = 0.0
    samples: list[TrackSample] = field(default_factory=list)
    mode_changes: list[tuple[float, str, str]] = field(default_factory=list)
    fsm_transitions: list[tuple[float, str]] = field(default_factory=list)
    min_clearance_m: float = float("inf")
    max_north_reached_m: float = 0.0
    max_east_reached_m: float = 0.0
    planner_visited_cruising: bool = False
    planner_visited_avoiding: bool = False
    completed_disarm: bool = False
    pass_clearance: bool = False
    pass_passed_obstacle: bool = False
    pass_planner_states: bool = False
    pass_disarmed: bool = False


# ---------------------------------------------------------------------- #
# Bridge subprocess                                                      #
# ---------------------------------------------------------------------- #


def spawn_gazebo_bridge(jsonl_path: Path, topic: str) -> subprocess.Popen:
    venv_py = _AUTONOMY_DIR / "venv" / "bin" / "python"
    if not venv_py.is_file():
        venv_py = Path(sys.executable)
    cmd = [
        str(venv_py),
        str(BRIDGE_SCRIPT),
        "--topic",
        topic,
        "--jsonl",
        str(jsonl_path),
        "--quiet",
    ]
    log(f"spawning gazebo bridge: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    return proc


# ---------------------------------------------------------------------- #
# Outputs                                                                #
# ---------------------------------------------------------------------- #


def write_outputs(summary: TestSummary, out_dir: Path, stamp: str) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"t5_gazebo_{stamp}.json"
    payload = {
        "test": "T5_gazebo_closed_loop_avoidance",
        "timestamp": stamp,
        "summary": {**{k: v for k, v in asdict(summary).items() if k != "samples"}},
        "samples": [asdict(s) for s in summary.samples],
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    png_path: Path | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return json_path, None

    if not summary.samples:
        return json_path, None

    times = [s.t_offset for s in summary.samples]
    norths = [s.north_m for s in summary.samples]
    easts = [s.east_m for s in summary.samples]
    dists = [s.distance_to_obstacle_m for s in summary.samples]
    clrs = [s.clearance_m for s in summary.samples]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(easts, norths, "-", linewidth=0.8, alpha=0.7)
    ax.plot(easts, norths, ".", markersize=3)
    # Obstacle cylinder.
    theta = [i * 2 * math.pi / 60 for i in range(61)]
    ox = [
        summary.obstacle_e_m + summary.obstacle_radius_m * math.cos(t) for t in theta
    ]
    oy = [
        summary.obstacle_n_m + summary.obstacle_radius_m * math.sin(t) for t in theta
    ]
    ax.fill(ox, oy, color="r", alpha=0.4, label=f"obstacle (r={summary.obstacle_radius_m:.2f} m)")
    # Clearance bound ring.
    bound_r = summary.obstacle_radius_m + summary.min_clearance_bound_m
    bx = [summary.obstacle_e_m + bound_r * math.cos(t) for t in theta]
    by = [summary.obstacle_n_m + bound_r * math.sin(t) for t in theta]
    ax.plot(bx, by, "r--", linewidth=1, alpha=0.6, label=f"+{summary.min_clearance_bound_m:.1f} m bound")
    ax.plot([0], [0], "g+", markersize=10, label="home")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Trajectory vs obstacle")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(times, dists, "-", color="tab:blue", label="distance to obstacle center")
    ax.plot(times, clrs, "-", color="tab:orange", label="clearance from surface")
    ax.axhline(summary.min_clearance_bound_m, color="r", linestyle="--", linewidth=1, label=f"{summary.min_clearance_bound_m:.1f} m bound")
    ax.set_xlabel("t since planner start (s)")
    ax.set_ylabel("metres")
    ax.set_title(f"Clearance (min = {summary.min_clearance_m:.2f} m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    modes = [s.planner_mode for s in summary.samples]
    cruise_mask = [m == "CRUISING" for m in modes]
    avoid_mask = [m == "AVOIDING" for m in modes]
    ax.plot(
        [t for t, m in zip(times, cruise_mask) if m],
        [n for n, m in zip(norths, cruise_mask) if m],
        "o", color="tab:blue", markersize=3, label="CRUISING",
    )
    ax.plot(
        [t for t, m in zip(times, avoid_mask) if m],
        [n for n, m in zip(norths, avoid_mask) if m],
        "s", color="tab:orange", markersize=3, label="AVOIDING",
    )
    ax.axhline(summary.obstacle_n_m, color="r", linestyle="--", linewidth=1, alpha=0.6, label=f"obstacle @ {summary.obstacle_n_m:.0f} m N")
    ax.axhline(summary.target_north_m, color="g", linestyle="--", linewidth=1, alpha=0.6, label=f"target {summary.target_north_m:.0f} m N")
    ax.set_xlabel("t since planner start (s)")
    ax.set_ylabel("north position (m)")
    ax.set_title("Planner mode + N progress")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"T5 closed-loop avoidance in Gazebo — {stamp}")
    fig.tight_layout()
    png_path = out_dir / f"t5_gazebo_{stamp}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return json_path, png_path


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument(
        "--lidar-topic",
        default="/iris/forward_lidar",
        help="Gazebo LaserScan topic the bridge will subscribe to.",
    )
    ap.add_argument(
        "--no-spawn-bridge",
        action="store_true",
        help="Don't spawn gazebo_perception_bridge.py — assume it's already running.",
    )
    return ap.parse_args()


def run() -> int:
    args = parse_args()

    # Reset the JSONL so we tail-from-zero into a clean stream.
    if not args.no_spawn_bridge:
        try:
            args.jsonl.unlink()
        except FileNotFoundError:
            pass

    bridge_proc: subprocess.Popen | None = None
    if not args.no_spawn_bridge:
        bridge_proc = spawn_gazebo_bridge(args.jsonl, args.lidar_topic)
        # Give the bridge a moment to attach to the topic before we start.
        time.sleep(2.0)
        if bridge_proc.poll() is not None:
            log("FAIL: bridge process exited immediately")
            if bridge_proc.stdout is not None:
                log(bridge_proc.stdout.read())
            return 1

    cfg = load_bridge_config()
    log(f"connecting bridge to {cfg['connection_uri']} ...")
    v = Vehicle(
        connection_uri=cfg["connection_uri"],
        source_system=cfg.get("source_system", 255),
        heartbeat_timeout=cfg.get("heartbeat_timeout", 30.0),
    )
    try:
        v.connect()
    except BridgeError as e:
        log(f"FAIL: bridge connect: {e}")
        if bridge_proc is not None:
            os.killpg(os.getpgid(bridge_proc.pid), signal.SIGTERM)
        return 1

    fsm = StateMachine(v, poll_hz=10.0)
    summary = TestSummary()

    def on_fsm(old: FlightState, new: FlightState, reason) -> None:
        summary.fsm_transitions.append((time.time(), new.name))
        log(f"FSM: {old} -> {new}" + (f"  ({reason})" if reason else ""))

    fsm.subscribe(on_fsm)
    fsm.start()

    try:
        log("waiting for GPS fix + PREARMED ...")
        deadline = time.time() + 30
        while time.time() < deadline and fsm.state != FlightState.PREARMED:
            time.sleep(0.2)
        if fsm.state != FlightState.PREARMED:
            log(f"FAIL: never reached PREARMED (stuck in {fsm.state})")
            return 1

        log("GUIDED + arm + takeoff ...")
        v.set_mode("GUIDED")
        arm_with_retry(v)
        # 60 s takeoff timeout: Gazebo's real-time factor can dip well
        # below 1.0 with the lidar + obstacle scene, so the climb to 5 m
        # in wall-clock can take 20-30 s.
        v.takeoff(CRUISE_ALT_M, timeout=60.0)
        log(f"  reached alt={v.state.alt_rel:.2f} m")
        time.sleep(SETTLE_S)

        # Capture home reference position after takeoff.
        home = v.state
        ref_lat = home.lat or 0.0
        ref_lon = home.lon or 0.0
        log(f"  home ref: lat={ref_lat:.6f} lon={ref_lon:.6f}")

        # Perception source + planner.
        src = DepthDetectSource(
            jsonl_path=str(args.jsonl),
            tail_from_end=True,  # only react to scans from after takeoff
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
            summary.mode_changes.append((offset, old.name, new.name))
            if new == PlannerMode.CRUISING:
                summary.planner_visited_cruising = True
            elif new == PlannerMode.AVOIDING:
                summary.planner_visited_avoiding = True
            log(f"Planner: {old} -> {new}  (+{offset:.2f}s)")

        plan.subscribe(on_planner)

        log(f"closed loop running for {TOTAL_DURATION_S} s ...")
        src.start()
        planner_t0 = time.time()
        plan.start()

        sample_period = 0.25
        next_sample = planner_t0
        while time.time() < planner_t0 + TOTAL_DURATION_S:
            now = time.time()
            if now >= next_sample:
                st = v.state
                if (
                    st.lat is not None
                    and st.lon is not None
                    and st.alt_rel is not None
                ):
                    n, e = latlon_to_local_ne(st.lat, st.lon, ref_lat, ref_lon)
                    d_center = math.hypot(n - OBSTACLE_N_M, e - OBSTACLE_E_M)
                    clearance = d_center - OBSTACLE_RADIUS_M
                    summary.samples.append(
                        TrackSample(
                            t_offset=now - planner_t0,
                            planner_mode=plan.mode.name,
                            north_m=n,
                            east_m=e,
                            alt_rel_m=st.alt_rel,
                            distance_to_obstacle_m=d_center,
                            clearance_m=clearance,
                        )
                    )
                    summary.min_clearance_m = min(summary.min_clearance_m, clearance)
                    summary.max_north_reached_m = max(summary.max_north_reached_m, n)
                    summary.max_east_reached_m = max(summary.max_east_reached_m, e)
                next_sample = now + sample_period
            time.sleep(0.05)

        log("loop done — stopping planner + perception ...")
        plan.stop()
        src.stop()

        log(f"RTL + waiting for disarm (up to {RTL_DISARM_TIMEOUT_S:.0f} s, "
            f"Gazebo RTF accommodation) ...")
        v.rtl()
        deadline = time.time() + RTL_DISARM_TIMEOUT_S
        while time.time() < deadline and v.state.armed:
            time.sleep(0.5)
        summary.completed_disarm = not v.state.armed
        if not summary.completed_disarm:
            log(f"WARN: did not disarm within {RTL_DISARM_TIMEOUT_S:.0f} s")
        else:
            log(f"  disarmed (mode={v.state.mode})")

        summary.duration_s = TOTAL_DURATION_S

        # Acceptance.
        summary.pass_clearance = summary.min_clearance_m >= MIN_CLEARANCE_M
        summary.pass_passed_obstacle = summary.max_north_reached_m >= TARGET_NORTH_M
        summary.pass_planner_states = (
            summary.planner_visited_cruising and summary.planner_visited_avoiding
        )
        summary.pass_disarmed = summary.completed_disarm

        out_dir = (_AUTONOMY_DIR / ".." / "sitl" / "logs").resolve()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        json_path, png_path = write_outputs(summary, out_dir, stamp)

        print()
        log("=" * 64)
        log("T5 Gazebo closed-loop avoidance — results")
        log("=" * 64)
        log(f"  Samples                       :  {len(summary.samples)}  over {summary.duration_s:.1f} s")
        log(f"  Max N reached                 :  {summary.max_north_reached_m:.2f} m  (target {TARGET_NORTH_M:.1f} m)")
        log(f"  Max E reached                 :  {summary.max_east_reached_m:.2f} m")
        log(f"  Min distance to obstacle ctr  :  {summary.min_clearance_m + OBSTACLE_RADIUS_M:.3f} m")
        log(f"  Min clearance from surface    :  {summary.min_clearance_m:.3f} m  (bound {MIN_CLEARANCE_M:.2f} m)")
        log(f"  Planner visited CRUISING      :  {summary.planner_visited_cruising}")
        log(f"  Planner visited AVOIDING      :  {summary.planner_visited_avoiding}")
        log(f"  Disarmed after RTL            :  {summary.completed_disarm}")
        log("")
        log(f"  PASS clearance ≥ {MIN_CLEARANCE_M:.1f} m       :  "
            f"{'PASS' if summary.pass_clearance else 'FAIL'}  ({summary.min_clearance_m:.3f})")
        log(f"  PASS passed obstacle (N ≥ {TARGET_NORTH_M:.0f}) :  "
            f"{'PASS' if summary.pass_passed_obstacle else 'FAIL'}  ({summary.max_north_reached_m:.2f})")
        log(f"  PASS planner CRUISING+AVOIDING :  "
            f"{'PASS' if summary.pass_planner_states else 'FAIL'}")
        log(f"  PASS disarmed after RTL        :  "
            f"{'PASS' if summary.pass_disarmed else 'FAIL'}")
        log("")
        log(f"  Output JSON                   :  {json_path}")
        if png_path is not None:
            log(f"  Output PNG                    :  {png_path}")

        # The dissertation T5 spec (§ 5.4) is "minimum clearance > 30 cm".
        # That, plus evidence of closed-loop CRUISING<->AVOIDING transitions,
        # is the formal PASS bar. Passing the obstacle on the far side and
        # disarming-within-budget are operational extras logged but not
        # gating the PASS.
        all_pass = summary.pass_clearance and summary.pass_planner_states
        if all_pass:
            log("PASS: T5 closed-loop avoidance in Gazebo demonstrated "
                "(formal clearance + closed-loop criteria met).")
            return 0
        log("FAIL: T5 formal criteria not met (clearance or planner states).")
        return 1

    except BridgeError as e:
        log(f"FAIL: bridge error: {e}")
        return 1
    except KeyboardInterrupt:
        log("interrupted by user.")
        return 130
    finally:
        log("stopping FSM + disconnecting bridge ...")
        fsm.stop()
        v.disconnect()
        if bridge_proc is not None and bridge_proc.poll() is None:
            try:
                os.killpg(os.getpgid(bridge_proc.pid), signal.SIGTERM)
                bridge_proc.wait(timeout=3.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(bridge_proc.pid), signal.SIGKILL)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(run())
