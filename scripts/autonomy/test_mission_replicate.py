"""
scripts/autonomy/test_mission_replicate.py

T6 multi-run mission replication test against ArduPilot SITL.

Flies the same QGC WPL mission N times in sequence against one running SITL
instance, captures per-run extents (max-north, max-east displacement from
the takeoff position; peak altitude; mission duration), and aggregates to
mean ± σ. Promotes the dissertation's §6.4 FR5/T6 row from a single-run
"provisionally supported" demonstration to a quantitative
"supported over N runs" result.

Default mission: scripts/sitl/missions/box_50m.txt
Default N: 10
Default acceptance: each axis within MISSION_EXTENT_TOLERANCE_PCT of target.

Pre-flight:
    1. SITL up:    scripts/sitl/run_sitl.sh
    2. Autonomy venv active

Run:
    source ~/Documents/AutonoBird/scripts/autonomy/venv/bin/activate
    python ~/Documents/AutonoBird/scripts/autonomy/test_mission_replicate.py
    # or with overrides:
    python test_mission_replicate.py --runs 5 --mission ../sitl/missions/box_50m.txt

Outputs JSON + PNG at scripts/sitl/logs/t6_replicate_<timestamp>.{json,png}
"""

from __future__ import annotations

import argparse
import json
import math
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
from pymavlink import mavutil  # noqa: E402

from state_machine import FlightState, StateMachine  # noqa: E402


# ---------------------------------------------------------------------- #
# Defaults                                                               #
# ---------------------------------------------------------------------- #

DEFAULT_MISSION = (_AUTONOMY_DIR / ".." / "sitl" / "missions" / "box_50m.txt").resolve()
DEFAULT_RUNS = 10
SAMPLE_RATE_HZ = 5.0

# Acceptance: per-run extents within 5 % of target. Target is the largest
# extent across the parsed waypoints (50 m for box_50m).
MISSION_EXTENT_TOLERANCE_PCT = 5.0

# Per-run timeouts.
PER_RUN_TIMEOUT_S = 240.0      # whole mission incl. takeoff + RTL
DISARM_TIMEOUT_S = 90.0

ARM_RETRIES = 4
ARM_RETRY_DELAY_S = 5.0


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


# ---------------------------------------------------------------------- #
# QGC WPL parsing                                                        #
# ---------------------------------------------------------------------- #


def parse_qgc_wpl(path: Path) -> list[dict]:
    """Parse a QGC WPL 110 mission file into Vehicle.upload_mission items.

    QGC line layout (tab-separated):
        seq current frame command p1 p2 p3 p4 x(lat) y(lon) z(alt) autocontinue

    Lat/lon are stored as float degrees in the file; mission_item_int_send
    wants int32 = degrees * 1e7. We do that conversion here. The HOME row
    (seq 0) is preserved with current=1 as the file specifies.
    """
    lines = path.read_text().splitlines()
    if not lines or "QGC WPL" not in lines[0]:
        raise ValueError(f"{path}: not a QGC WPL file (missing header)")
    items: list[dict] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 12:
            raise ValueError(f"{path}: short row: {line!r}")
        seq, current, frame, command = (int(x) for x in parts[:4])
        p1, p2, p3, p4 = (float(x) for x in parts[4:8])
        x, y, z = float(parts[8]), float(parts[9]), float(parts[10])
        autocontinue = int(parts[11])
        items.append(
            {
                "seq": seq,
                "current": current,
                "frame": frame,
                "command": command,
                "autocontinue": autocontinue,
                "param1": p1,
                "param2": p2,
                "param3": p3,
                "param4": p4,
                "x": int(round(x * 1e7)),
                "y": int(round(y * 1e7)),
                "z": z,
            }
        )
    return items


def mission_target_extent_m(items: list[dict]) -> tuple[float, float, float]:
    """Compute (north_extent, east_extent, peak_alt) of the mission's
    NAV_WAYPOINT corners.

    Counts only frame=3 (MAV_FRAME_GLOBAL_RELATIVE_ALT) NAV_WAYPOINT rows
    so the HOME row (frame=0, z=MSL elevation) doesn't poison the alt
    estimate. Extents are relative to the first NAV_WAYPOINT.
    """
    nav = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
    rel_alt_frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    waypoints = [
        it for it in items
        if it["command"] == nav and it["frame"] == rel_alt_frame
    ]
    if not waypoints:
        return 0.0, 0.0, 0.0
    ref_lat = waypoints[0]["x"] / 1e7
    ref_lon = waypoints[0]["y"] / 1e7
    norths: list[float] = []
    easts: list[float] = []
    peak_alt = 0.0
    for it in waypoints:
        lat = it["x"] / 1e7
        lon = it["y"] / 1e7
        n, e = latlon_to_local_ne(lat, lon, ref_lat, ref_lon)
        norths.append(n)
        easts.append(e)
        peak_alt = max(peak_alt, it["z"])
    n_ext = max(norths) - min(norths)
    e_ext = max(easts) - min(easts)
    return n_ext, e_ext, peak_alt


# ---------------------------------------------------------------------- #
# Per-run flight                                                         #
# ---------------------------------------------------------------------- #


@dataclass
class RunResult:
    run_index: int
    duration_s: float
    items_reached: list[int]
    n_extent_m: float
    e_extent_m: float
    peak_alt_m: float
    ref_lat: float
    ref_lon: float
    n_extent_err_pct: float
    e_extent_err_pct: float
    alt_err_pct: float
    completed_mission: bool
    pass_extents: bool


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


def wait_for(fsm: StateMachine, target: FlightState, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fsm.state == target:
            return True
        time.sleep(0.2)
    return False


def fly_one(
    v: Vehicle,
    fsm: StateMachine,
    mission_items: list[dict],
    target: tuple[float, float, float],
    takeoff_alt_m: float,
    run_index: int,
) -> RunResult:
    n_target, e_target, alt_target = target

    log(f"--- Run {run_index} ---")

    # Wait for the autopilot to be ready to arm again. After the previous
    # disarm the FSM passes through DISARMED_POSTFLIGHT -> PREARMED; on
    # the first run it just starts in PREARMED.
    if not wait_for(fsm, FlightState.PREARMED, timeout_s=30.0):
        raise BridgeError(f"Run {run_index}: never reached PREARMED (state={fsm.state})")

    log("  Uploading mission ...")
    v.upload_mission(mission_items)

    log("  GUIDED + arm ...")
    v.set_mode("GUIDED")
    arm_with_retry(v)

    log(f"  Takeoff to {takeoff_alt_m} m ...")
    v.takeoff(takeoff_alt_m)

    # Reference position = first sample with a valid lat/lon, taken right
    # after takeoff completes. Used to express extents in local NE.
    ref_state = v.state
    ref_lat = ref_state.lat or 0.0
    ref_lon = ref_state.lon or 0.0
    log(f"  Reference (post-takeoff): lat={ref_lat:.6f} lon={ref_lon:.6f}")

    log("  Switching to AUTO (mission resumes from waypoint 2, takeoff item is auto-skipped) ...")
    v.set_mode("AUTO")

    # Drain queued events from prior phases (e.g. mode-change events).
    v.drain_events()
    items_reached: list[int] = []

    # ---- Sample loop until disarm ---- #
    sample_period = 1.0 / SAMPLE_RATE_HZ
    t0 = time.time()
    next_sample = t0
    peak_alt = 0.0
    n_min = n_max = 0.0
    e_min = e_max = 0.0
    completed = False
    while time.time() < t0 + PER_RUN_TIMEOUT_S:
        # Drain events for MISSION_ITEM_REACHED.
        for ev in v.drain_events():
            if ev.kind == "item_reached":
                items_reached.append(int(ev.payload))
                log(f"    item_reached: seq {ev.payload}")

        # Sample telemetry.
        now = time.time()
        if now >= next_sample:
            st = v.state
            if st.lat is not None and st.lon is not None and st.alt_rel is not None:
                n, e = latlon_to_local_ne(st.lat, st.lon, ref_lat, ref_lon)
                n_min = min(n_min, n)
                n_max = max(n_max, n)
                e_min = min(e_min, e)
                e_max = max(e_max, e)
                peak_alt = max(peak_alt, st.alt_rel)
            next_sample = now + sample_period

        # Exit when disarmed (RTL completed + landed).
        if not v.state.armed and (time.time() - t0) > 5.0:
            completed = True
            break

        time.sleep(0.05)

    duration = time.time() - t0
    log(f"  Run {run_index} finished in {duration:.1f} s, "
        f"items_reached={items_reached}, completed_disarm={completed}")

    n_extent = n_max - n_min
    e_extent = e_max - e_min
    n_err = 100.0 * (n_extent - n_target) / n_target if n_target else 0.0
    e_err = 100.0 * (e_extent - e_target) / e_target if e_target else 0.0
    a_err = 100.0 * (peak_alt - alt_target) / alt_target if alt_target else 0.0

    pass_extents = (
        abs(n_err) <= MISSION_EXTENT_TOLERANCE_PCT
        and abs(e_err) <= MISSION_EXTENT_TOLERANCE_PCT
        and abs(a_err) <= MISSION_EXTENT_TOLERANCE_PCT
    )

    return RunResult(
        run_index=run_index,
        duration_s=duration,
        items_reached=items_reached,
        n_extent_m=n_extent,
        e_extent_m=e_extent,
        peak_alt_m=peak_alt,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        n_extent_err_pct=n_err,
        e_extent_err_pct=e_err,
        alt_err_pct=a_err,
        completed_mission=completed,
        pass_extents=pass_extents,
    )


# ---------------------------------------------------------------------- #
# Aggregation + outputs                                                  #
# ---------------------------------------------------------------------- #


@dataclass
class ReplicateSummary:
    mission_path: str
    n_runs: int
    target_n_m: float
    target_e_m: float
    target_alt_m: float
    tolerance_pct: float
    runs: list[RunResult] = field(default_factory=list)
    n_extent_mean: float = 0.0
    n_extent_std: float = 0.0
    e_extent_mean: float = 0.0
    e_extent_std: float = 0.0
    alt_mean: float = 0.0
    alt_std: float = 0.0
    duration_mean: float = 0.0
    pass_rate: float = 0.0
    all_pass: bool = False


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def aggregate(runs: list[RunResult], summary: ReplicateSummary) -> ReplicateSummary:
    summary.runs = runs
    if not runs:
        return summary
    n_ext = [r.n_extent_m for r in runs]
    e_ext = [r.e_extent_m for r in runs]
    alts = [r.peak_alt_m for r in runs]
    durs = [r.duration_s for r in runs]
    summary.n_extent_mean, summary.n_extent_std = _mean_std(n_ext)
    summary.e_extent_mean, summary.e_extent_std = _mean_std(e_ext)
    summary.alt_mean, summary.alt_std = _mean_std(alts)
    summary.duration_mean, _ = _mean_std(durs)
    summary.pass_rate = 100.0 * sum(1 for r in runs if r.pass_extents) / len(runs)
    summary.all_pass = all(r.pass_extents and r.completed_mission for r in runs)
    return summary


def write_outputs(summary: ReplicateSummary, out_dir: Path, stamp: str) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"t6_replicate_{stamp}.json"
    payload = {
        "test": "T6_mission_replicate",
        "timestamp": stamp,
        "summary": {
            **{k: v for k, v in asdict(summary).items() if k != "runs"},
        },
        "runs": [asdict(r) for r in summary.runs],
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

    if not summary.runs:
        return json_path, None

    runs = summary.runs
    xs = [r.run_index for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar([x - 0.2 for x in xs], [r.n_extent_m for r in runs], width=0.4, label="N extent")
    ax.bar([x + 0.2 for x in xs], [r.e_extent_m for r in runs], width=0.4, label="E extent")
    ax.axhline(summary.target_n_m, color="r", linestyle="--", linewidth=1, label=f"target {summary.target_n_m:.0f} m")
    tol_lo = summary.target_n_m * (1 - summary.tolerance_pct / 100)
    tol_hi = summary.target_n_m * (1 + summary.tolerance_pct / 100)
    ax.axhspan(tol_lo, tol_hi, color="r", alpha=0.08, label=f"±{summary.tolerance_pct:.0f}%")
    ax.set_xlabel("run")
    ax.set_ylabel("extent (m)")
    ax.set_title("Per-run mission extents")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(xs, [r.peak_alt_m for r in runs], color="tab:green")
    ax.axhline(summary.target_alt_m, color="r", linestyle="--", linewidth=1, label=f"target {summary.target_alt_m:.0f} m")
    tol_lo = summary.target_alt_m * (1 - summary.tolerance_pct / 100)
    tol_hi = summary.target_alt_m * (1 + summary.tolerance_pct / 100)
    ax.axhspan(tol_lo, tol_hi, color="r", alpha=0.08, label=f"±{summary.tolerance_pct:.0f}%")
    ax.set_xlabel("run")
    ax.set_ylabel("peak alt (m)")
    ax.set_title("Per-run peak altitude")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(xs, [r.duration_s for r in runs], color="tab:purple")
    ax.set_xlabel("run")
    ax.set_ylabel("duration (s)")
    ax.set_title(f"Per-run duration  (mean={summary.duration_mean:.1f} s)")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"T6 mission replication — {summary.n_runs} runs, "
        f"N={summary.n_extent_mean:.2f}±{summary.n_extent_std:.2f} m, "
        f"E={summary.e_extent_mean:.2f}±{summary.e_extent_std:.2f} m, "
        f"alt={summary.alt_mean:.2f}±{summary.alt_std:.2f} m  ({stamp})"
    )
    fig.tight_layout()
    png_path = out_dir / f"t6_replicate_{stamp}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return json_path, png_path


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--mission", type=Path, default=DEFAULT_MISSION)
    ap.add_argument("--takeoff-alt", type=float, default=10.0)
    ap.add_argument("--tolerance-pct", type=float, default=MISSION_EXTENT_TOLERANCE_PCT)
    return ap.parse_args()


def run() -> int:
    args = parse_args()

    log(f"Mission file : {args.mission}")
    if not args.mission.is_file():
        log(f"FAIL: mission file not found: {args.mission}")
        return 1
    items = parse_qgc_wpl(args.mission)
    n_target, e_target, alt_target = mission_target_extent_m(items)
    log(f"Mission target extent : N={n_target:.1f} m, E={e_target:.1f} m, peak alt={alt_target:.1f} m")
    log(f"Runs={args.runs}, tolerance=±{args.tolerance_pct:.1f}%")

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
    fsm.start()

    summary = ReplicateSummary(
        mission_path=str(args.mission),
        n_runs=args.runs,
        target_n_m=n_target,
        target_e_m=e_target,
        target_alt_m=alt_target,
        tolerance_pct=args.tolerance_pct,
    )

    runs: list[RunResult] = []
    try:
        for i in range(1, args.runs + 1):
            try:
                r = fly_one(
                    v=v,
                    fsm=fsm,
                    mission_items=items,
                    target=(n_target, e_target, alt_target),
                    takeoff_alt_m=args.takeoff_alt,
                    run_index=i,
                )
            except BridgeError as e:
                log(f"  Run {i}: bridge error: {e}")
                r = RunResult(
                    run_index=i,
                    duration_s=0.0,
                    items_reached=[],
                    n_extent_m=0.0,
                    e_extent_m=0.0,
                    peak_alt_m=0.0,
                    ref_lat=0.0,
                    ref_lon=0.0,
                    n_extent_err_pct=-100.0,
                    e_extent_err_pct=-100.0,
                    alt_err_pct=-100.0,
                    completed_mission=False,
                    pass_extents=False,
                )
            runs.append(r)

        # ---- Aggregate ---- #
        aggregate(runs, summary)
        out_dir = (_AUTONOMY_DIR / ".." / "sitl" / "logs").resolve()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        json_path, png_path = write_outputs(summary, out_dir, stamp)

        print()
        log("=" * 64)
        log("T6 mission replication — results")
        log("=" * 64)
        log(f"  Runs                    :  {args.runs}")
        log(f"  Target N extent         :  {n_target:.2f} m")
        log(f"  Target E extent         :  {e_target:.2f} m")
        log(f"  Target peak alt         :  {alt_target:.2f} m")
        log(f"  Tolerance               :  ±{args.tolerance_pct:.1f}%")
        log("")
        log(f"  Mean N extent           :  {summary.n_extent_mean:.2f} ± {summary.n_extent_std:.2f} m")
        log(f"  Mean E extent           :  {summary.e_extent_mean:.2f} ± {summary.e_extent_std:.2f} m")
        log(f"  Mean peak alt           :  {summary.alt_mean:.2f} ± {summary.alt_std:.2f} m")
        log(f"  Mean run duration       :  {summary.duration_mean:.1f} s")
        log("")
        log("  Per-run summary (extent N/E/alt, |err|, completed):")
        for r in runs:
            ok = "PASS" if (r.pass_extents and r.completed_mission) else "FAIL"
            log(
                f"    run {r.run_index:2d}: "
                f"N={r.n_extent_m:5.2f}m ({r.n_extent_err_pct:+.1f}%)  "
                f"E={r.e_extent_m:5.2f}m ({r.e_extent_err_pct:+.1f}%)  "
                f"alt={r.peak_alt_m:5.2f}m ({r.alt_err_pct:+.1f}%)  "
                f"items={len(r.items_reached)}  "
                f"{r.duration_s:.0f}s  {ok}"
            )
        log("")
        log(f"  Output JSON             :  {json_path}")
        if png_path is not None:
            log(f"  Output PNG              :  {png_path}")
        log("")
        log(f"  Pass rate               :  {summary.pass_rate:.0f}%")
        if not summary.all_pass:
            log(f"FAIL: {sum(1 for r in runs if not r.pass_extents or not r.completed_mission)} of {args.runs} runs did not meet acceptance.")
            return 1
        log(f"PASS: all {args.runs} runs flew the mission within ±{args.tolerance_pct:.0f}% on every axis.")
        return 0

    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130
    finally:
        log("Stopping FSM + disconnecting bridge ...")
        fsm.stop()
        v.disconnect()


if __name__ == "__main__":
    sys.exit(run())
