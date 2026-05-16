"""
scripts/autonomy/test_wind_sweep.py

Wind / disturbance rejection sweep against ArduPilot SITL.

Re-runs the T4 hover and an abbreviated T6 mission at multiple SIM_WIND_SPD
levels (fixed direction, no turbulence) and reports the wind-vs-stability
relationship. Closes the engineering-backlog §5 wind / disturbance-rejection
item and provides a dissertation §6.2 subsection upgrade.

The script connects to one running SITL instance and uses
`Vehicle.set_param()` to ramp SIM_WIND_SPD between conditions. At each wind
level the bird:
    1. Lifts off in GUIDED to 5 m, holds for HOVER_DURATION_S seconds while
       sampling lat / lon / alt at 10 Hz (T4-style drift logging).
    2. Lands, disarms.
    3. Re-arms, takes off to 10 m, switches to AUTO to fly the box_50m
       mission, captures max NE extents and peak altitude (T6-style).
    4. Repeats step 3 once more (2 missions per wind level) for σ.
    5. Lands, disarms, advances to the next wind level.

After the final level the script resets SIM_WIND_SPD back to 0.

Outputs:
    scripts/sitl/logs/wind_sweep_<timestamp>.json
    scripts/sitl/logs/wind_sweep_<timestamp>.png

Pre-flight:
    1. SITL up:    scripts/sitl/run_sitl.sh
    2. Autonomy venv active

Run:
    source ~/Documents/AutonoBird/scripts/autonomy/venv/bin/activate
    python ~/Documents/AutonoBird/scripts/autonomy/test_wind_sweep.py
"""

from __future__ import annotations

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

from state_machine import FlightState, StateMachine  # noqa: E402
from test_mission_replicate import (  # noqa: E402
    parse_qgc_wpl,
    mission_target_extent_m,
)


# ---------------------------------------------------------------------- #
# Sweep parameters                                                       #
# ---------------------------------------------------------------------- #

WIND_LEVELS_MS = [0.0, 5.0, 10.0, 15.0]
WIND_DIR_DEG = 270.0    # wind from west (blowing east). Crosswind for the
                        # N–S legs of the box mission, head/tailwind on E/W.
HOVER_ALT_M = 5.0
HOVER_DURATION_S = 30.0
HOVER_SETTLE_S = 3.0
MISSION_TAKEOFF_ALT_M = 10.0
MISSIONS_PER_LEVEL = 2
SAMPLE_RATE_HZ = 10.0
EXTENT_TOLERANCE_PCT = 5.0

# T4 dissertation bound.
HORIZONTAL_DRIFT_BOUND_M = 0.5
VERTICAL_DRIFT_BOUND_M = 0.5

PER_MISSION_TIMEOUT_S = 240.0
ARM_RETRIES = 4
ARM_RETRY_DELAY_S = 5.0

MISSION_PATH = (_AUTONOMY_DIR / ".." / "sitl" / "missions" / "box_50m.txt").resolve()


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
            log(f"    arm attempt {i + 1}/{ARM_RETRIES} failed: {e}")
            time.sleep(ARM_RETRY_DELAY_S)
    raise BridgeError(f"arm() failed after {ARM_RETRIES} attempts: {last_err}")


def wait_for(fsm: StateMachine, target: FlightState, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fsm.state == target:
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------------- #
# Result containers                                                      #
# ---------------------------------------------------------------------- #


@dataclass
class HoverResult:
    wind_speed_ms: float
    n_samples: int
    duration_s: float
    max_horizontal_drift_m: float
    rms_horizontal_drift_m: float
    max_vertical_drift_m: float
    rms_vertical_drift_m: float
    horizontal_pass: bool
    vertical_pass: bool


@dataclass
class MissionResult:
    wind_speed_ms: float
    run_index: int
    duration_s: float
    n_extent_m: float
    e_extent_m: float
    peak_alt_m: float
    n_extent_err_pct: float
    e_extent_err_pct: float
    alt_err_pct: float
    completed_mission: bool
    pass_extents: bool


@dataclass
class WindLevelResult:
    wind_speed_ms: float
    hover: HoverResult | None = None
    missions: list[MissionResult] = field(default_factory=list)


@dataclass
class SweepSummary:
    wind_dir_deg: float
    wind_levels_ms: list[float]
    target_n_m: float
    target_e_m: float
    target_alt_m: float
    horizontal_bound_m: float
    vertical_bound_m: float
    extent_tolerance_pct: float
    levels: list[WindLevelResult] = field(default_factory=list)


# ---------------------------------------------------------------------- #
# Hover (T4-style)                                                       #
# ---------------------------------------------------------------------- #


def run_hover(v: Vehicle, fsm: StateMachine, wind_speed: float) -> HoverResult:
    log(f"  Hover @ {wind_speed:.0f} m/s ...")
    if not wait_for(fsm, FlightState.PREARMED, timeout_s=30.0):
        raise BridgeError(f"Hover: never reached PREARMED (state={fsm.state})")

    v.set_mode("GUIDED")
    arm_with_retry(v)
    v.takeoff(HOVER_ALT_M)
    time.sleep(HOVER_SETTLE_S)

    samples: list[tuple[float, float, float, float]] = []
    sample_period = 1.0 / SAMPLE_RATE_HZ
    t0 = time.time()
    next_sample = t0
    while time.time() < t0 + HOVER_DURATION_S:
        now = time.time()
        if now >= next_sample:
            st = v.state
            if st.lat is not None and st.lon is not None and st.alt_rel is not None:
                samples.append((now - t0, st.lat, st.lon, st.alt_rel))
            next_sample = now + sample_period
        time.sleep(0.005)

    v.land()
    deadline = time.time() + 60
    while time.time() < deadline and v.state.armed:
        time.sleep(0.5)

    if not samples:
        return HoverResult(
            wind_speed_ms=wind_speed,
            n_samples=0,
            duration_s=0.0,
            max_horizontal_drift_m=0.0,
            rms_horizontal_drift_m=0.0,
            max_vertical_drift_m=0.0,
            rms_vertical_drift_m=0.0,
            horizontal_pass=False,
            vertical_pass=False,
        )

    _, ref_lat, ref_lon, ref_alt = samples[0]
    h_sq = 0.0
    v_sq = 0.0
    h_max = 0.0
    v_max = 0.0
    for _t, lat, lon, alt in samples:
        n, e = latlon_to_local_ne(lat, lon, ref_lat, ref_lon)
        h = math.hypot(n, e)
        dv = abs(alt - ref_alt)
        h_sq += h * h
        v_sq += dv * dv
        h_max = max(h_max, h)
        v_max = max(v_max, dv)
    rms_h = math.sqrt(h_sq / len(samples))
    rms_v = math.sqrt(v_sq / len(samples))

    result = HoverResult(
        wind_speed_ms=wind_speed,
        n_samples=len(samples),
        duration_s=samples[-1][0] - samples[0][0],
        max_horizontal_drift_m=h_max,
        rms_horizontal_drift_m=rms_h,
        max_vertical_drift_m=v_max,
        rms_vertical_drift_m=rms_v,
        horizontal_pass=h_max < HORIZONTAL_DRIFT_BOUND_M,
        vertical_pass=v_max < VERTICAL_DRIFT_BOUND_M,
    )
    log(
        f"    hover @ {wind_speed:.0f} m/s: max_h={h_max:.3f} m, "
        f"max_v={v_max:.3f} m, n={len(samples)}"
    )
    return result


# ---------------------------------------------------------------------- #
# Mission (T6-style, abbreviated)                                        #
# ---------------------------------------------------------------------- #


def run_one_mission(
    v: Vehicle,
    fsm: StateMachine,
    items: list[dict],
    target: tuple[float, float, float],
    wind_speed: float,
    run_index: int,
) -> MissionResult:
    log(f"  Mission run {run_index} @ {wind_speed:.0f} m/s ...")
    n_target, e_target, alt_target = target

    if not wait_for(fsm, FlightState.PREARMED, timeout_s=30.0):
        raise BridgeError(f"Mission: never reached PREARMED (state={fsm.state})")

    v.upload_mission(items)
    v.set_mode("GUIDED")
    arm_with_retry(v)
    v.takeoff(MISSION_TAKEOFF_ALT_M)

    ref_state = v.state
    ref_lat = ref_state.lat or 0.0
    ref_lon = ref_state.lon or 0.0

    v.set_mode("AUTO")
    v.drain_events()

    sample_period = 1.0 / SAMPLE_RATE_HZ
    t0 = time.time()
    next_sample = t0
    n_min = n_max = 0.0
    e_min = e_max = 0.0
    peak_alt = 0.0
    completed = False
    items_reached: set[int] = set()
    while time.time() < t0 + PER_MISSION_TIMEOUT_S:
        for ev in v.drain_events():
            if ev.kind == "item_reached":
                items_reached.add(int(ev.payload))
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
        if not v.state.armed and (time.time() - t0) > 5.0:
            completed = True
            break
        time.sleep(0.05)

    duration = time.time() - t0
    n_ext = n_max - n_min
    e_ext = e_max - e_min
    n_err = 100.0 * (n_ext - n_target) / n_target if n_target else 0.0
    e_err = 100.0 * (e_ext - e_target) / e_target if e_target else 0.0
    a_err = 100.0 * (peak_alt - alt_target) / alt_target if alt_target else 0.0
    pass_extents = (
        abs(n_err) <= EXTENT_TOLERANCE_PCT
        and abs(e_err) <= EXTENT_TOLERANCE_PCT
        and abs(a_err) <= EXTENT_TOLERANCE_PCT
    )
    result = MissionResult(
        wind_speed_ms=wind_speed,
        run_index=run_index,
        duration_s=duration,
        n_extent_m=n_ext,
        e_extent_m=e_ext,
        peak_alt_m=peak_alt,
        n_extent_err_pct=n_err,
        e_extent_err_pct=e_err,
        alt_err_pct=a_err,
        completed_mission=completed,
        pass_extents=pass_extents,
    )
    log(
        f"    mission run {run_index} @ {wind_speed:.0f} m/s: "
        f"N={n_ext:.2f}m ({n_err:+.1f}%), E={e_ext:.2f}m ({e_err:+.1f}%), "
        f"alt={peak_alt:.2f}m, items={len(items_reached)}/6, {duration:.0f}s, "
        f"{'PASS' if pass_extents and completed else 'FAIL'}"
    )
    return result


# ---------------------------------------------------------------------- #
# Outputs                                                                #
# ---------------------------------------------------------------------- #


def write_outputs(
    summary: SweepSummary, out_dir: Path, stamp: str
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"wind_sweep_{stamp}.json"
    payload = {
        "test": "wind_disturbance_sweep",
        "timestamp": stamp,
        "summary": {
            **{k: v for k, v in asdict(summary).items() if k != "levels"},
        },
        "levels": [asdict(lvl) for lvl in summary.levels],
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

    if not summary.levels:
        return json_path, None

    winds = [lvl.wind_speed_ms for lvl in summary.levels]
    hover_h = [lvl.hover.max_horizontal_drift_m if lvl.hover else 0.0 for lvl in summary.levels]
    hover_v = [lvl.hover.max_vertical_drift_m if lvl.hover else 0.0 for lvl in summary.levels]

    miss_n: list[float] = []
    miss_e: list[float] = []
    miss_pass: list[float] = []
    for lvl in summary.levels:
        if lvl.missions:
            miss_n.append(
                sum(m.n_extent_m for m in lvl.missions) / len(lvl.missions)
            )
            miss_e.append(
                sum(m.e_extent_m for m in lvl.missions) / len(lvl.missions)
            )
            miss_pass.append(
                100.0
                * sum(1 for m in lvl.missions if m.pass_extents and m.completed_mission)
                / len(lvl.missions)
            )
        else:
            miss_n.append(0.0)
            miss_e.append(0.0)
            miss_pass.append(0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(winds, hover_h, "o-", color="tab:blue", label="max horizontal drift")
    ax.plot(winds, hover_v, "s-", color="tab:orange", label="max vertical drift")
    ax.axhline(
        summary.horizontal_bound_m,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"{summary.horizontal_bound_m} m T4 bound",
    )
    ax.set_xlabel("wind speed (m/s)")
    ax.set_ylabel("drift (m)")
    ax.set_title("T4 hover drift vs wind")
    ax.set_xticks(winds)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(winds, miss_n, "o-", color="tab:blue", label="N extent (mean)")
    ax.plot(winds, miss_e, "s-", color="tab:orange", label="E extent (mean)")
    ax.axhline(
        summary.target_n_m,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"target {summary.target_n_m:.0f} m",
    )
    tol_lo = summary.target_n_m * (1 - summary.extent_tolerance_pct / 100)
    tol_hi = summary.target_n_m * (1 + summary.extent_tolerance_pct / 100)
    ax.axhspan(tol_lo, tol_hi, color="r", alpha=0.08, label=f"±{summary.extent_tolerance_pct:.0f}%")
    ax.set_xlabel("wind speed (m/s)")
    ax.set_ylabel("extent (m)")
    ax.set_title("T6 mission extents vs wind")
    ax.set_xticks(winds)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(winds, miss_pass, width=1.5, color="tab:green", alpha=0.7)
    ax.set_xlabel("wind speed (m/s)")
    ax.set_ylabel("pass rate (%)")
    ax.set_title(f"T6 pass rate at ±{summary.extent_tolerance_pct:.0f}% extent")
    ax.set_xticks(winds)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Wind disturbance sweep — direction {summary.wind_dir_deg:.0f}° (from west)  "
        f"({stamp})"
    )
    fig.tight_layout()
    png_path = out_dir / f"wind_sweep_{stamp}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return json_path, png_path


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def run() -> int:
    log(f"Mission file: {MISSION_PATH}")
    if not MISSION_PATH.is_file():
        log(f"FAIL: mission file not found: {MISSION_PATH}")
        return 1
    items = parse_qgc_wpl(MISSION_PATH)
    n_target, e_target, alt_target = mission_target_extent_m(items)
    log(f"Mission target: N={n_target:.2f} m, E={e_target:.2f} m, alt={alt_target:.2f} m")
    log(f"Wind sweep:  {WIND_LEVELS_MS} m/s from {WIND_DIR_DEG:.0f}°")
    log(f"Per level:   {HOVER_DURATION_S:.0f} s hover + {MISSIONS_PER_LEVEL} mission runs")

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

    # Fix the wind direction and turbulence once; only SIM_WIND_SPD changes.
    try:
        v.set_param("SIM_WIND_DIR", WIND_DIR_DEG)
        v.set_param("SIM_WIND_TURB", 0.0)
    except BridgeError as e:
        log(f"FAIL: could not set wind direction / turbulence: {e}")
        fsm.stop()
        v.disconnect()
        return 1

    summary = SweepSummary(
        wind_dir_deg=WIND_DIR_DEG,
        wind_levels_ms=list(WIND_LEVELS_MS),
        target_n_m=n_target,
        target_e_m=e_target,
        target_alt_m=alt_target,
        horizontal_bound_m=HORIZONTAL_DRIFT_BOUND_M,
        vertical_bound_m=VERTICAL_DRIFT_BOUND_M,
        extent_tolerance_pct=EXTENT_TOLERANCE_PCT,
    )

    try:
        for wind_speed in WIND_LEVELS_MS:
            log(f"=== Wind level: {wind_speed:.0f} m/s ===")
            try:
                v.set_param("SIM_WIND_SPD", wind_speed)
            except BridgeError as e:
                log(f"  PARAM_SET SIM_WIND_SPD={wind_speed} failed: {e}")
                continue
            time.sleep(2.0)  # let the wind model settle before takeoff

            level = WindLevelResult(wind_speed_ms=wind_speed)

            try:
                level.hover = run_hover(v, fsm, wind_speed)
            except BridgeError as e:
                log(f"  hover failed: {e}")

            for run_index in range(1, MISSIONS_PER_LEVEL + 1):
                try:
                    m = run_one_mission(
                        v=v,
                        fsm=fsm,
                        items=items,
                        target=(n_target, e_target, alt_target),
                        wind_speed=wind_speed,
                        run_index=run_index,
                    )
                except BridgeError as e:
                    log(f"  mission {run_index} failed: {e}")
                    m = MissionResult(
                        wind_speed_ms=wind_speed,
                        run_index=run_index,
                        duration_s=0.0,
                        n_extent_m=0.0,
                        e_extent_m=0.0,
                        peak_alt_m=0.0,
                        n_extent_err_pct=-100.0,
                        e_extent_err_pct=-100.0,
                        alt_err_pct=-100.0,
                        completed_mission=False,
                        pass_extents=False,
                    )
                level.missions.append(m)

            summary.levels.append(level)
    finally:
        # Always restore zero wind so subsequent SITL work isn't affected.
        try:
            v.set_param("SIM_WIND_SPD", 0.0)
            log("Wind reset to 0 m/s.")
        except BridgeError as e:
            log(f"WARN: could not reset wind: {e}")

    out_dir = (_AUTONOMY_DIR / ".." / "sitl" / "logs").resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path, png_path = write_outputs(summary, out_dir, stamp)

    print()
    log("=" * 72)
    log("Wind disturbance sweep — results")
    log("=" * 72)
    log(f"  Wind direction          :  {WIND_DIR_DEG:.0f}° (from west)")
    log(f"  Wind levels             :  {WIND_LEVELS_MS} m/s")
    log(f"  Per level               :  {HOVER_DURATION_S:.0f} s hover + "
        f"{MISSIONS_PER_LEVEL} mission run(s)")
    log("")
    log(
        f"  {'Wind':>6}  {'Hover h-drift':>14}  {'Hover v-drift':>14}  "
        f"{'Mission N̄':>10}  {'Mission Ē':>10}  {'Pass':>6}"
    )
    for lvl in summary.levels:
        h = lvl.hover
        n_avg = (
            sum(m.n_extent_m for m in lvl.missions) / len(lvl.missions)
            if lvl.missions
            else 0.0
        )
        e_avg = (
            sum(m.e_extent_m for m in lvl.missions) / len(lvl.missions)
            if lvl.missions
            else 0.0
        )
        pass_n = (
            sum(
                1 for m in lvl.missions if m.pass_extents and m.completed_mission
            )
            if lvl.missions
            else 0
        )
        total = len(lvl.missions)
        log(
            f"  {lvl.wind_speed_ms:>4.0f} m/s  "
            f"{(h.max_horizontal_drift_m if h else 0.0):>12.3f} m  "
            f"{(h.max_vertical_drift_m if h else 0.0):>12.3f} m  "
            f"{n_avg:>8.2f} m  {e_avg:>8.2f} m  "
            f"{pass_n}/{total}"
        )
    log("")
    log(f"  Output JSON             :  {json_path}")
    if png_path is not None:
        log(f"  Output PNG              :  {png_path}")

    log("Stopping FSM + disconnecting bridge ...")
    fsm.stop()
    v.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(run())
