"""
scripts/autonomy/test_hover_stability.py

T4 hover-stability test against ArduPilot SITL.

What this measures:
    - Takeoff to a fixed altitude in GUIDED.
    - Stay in GUIDED — after NAV_TAKEOFF completes, ArduCopter holds
      the takeoff position as an active target until commanded otherwise.
      This is the autonomy stack's actual hover mode; LOITER is the
      pilot-stick equivalent and in headless SITL with no RC input
      defaults to min-throttle and the bird descends into the ground.
    - Sample lat/lon/alt at 10 Hz for HOVER_DURATION_S seconds.
    - Convert lat/lon deltas to local NE metres relative to the first
      LOITER sample (haversine, accurate to <1 cm at this scale).
    - Compute: max horizontal drift radius, RMS horizontal drift, max
      vertical drift, RMS vertical drift.

Pass criterion: max horizontal drift < HORIZONTAL_DRIFT_BOUND_M
(the dissertation T4 specification — 50 cm at typical SITL home).

Pre-flight:
    1. SITL up:    scripts/sitl/run_sitl.sh
    2. Autonomy venv active

Run:
    source ~/Documents/AutonoBird/scripts/autonomy/venv/bin/activate
    python ~/Documents/AutonoBird/scripts/autonomy/test_hover_stability.py

Outputs JSON at scripts/sitl/logs/t4_hover_<timestamp>.json
and an optional matplotlib PNG (if matplotlib is importable).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Cross-subsystem bridge import.
_AUTONOMY_DIR = Path(__file__).parent
_BRIDGE_DIR = (_AUTONOMY_DIR / ".." / "flight-controller").resolve()
if str(_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_DIR))

from bridge import BridgeError, Vehicle, load_config as load_bridge_config  # noqa: E402

from state_machine import FlightState, StateMachine  # noqa: E402


# ---------------------------------------------------------------------- #
# Test parameters                                                        #
# ---------------------------------------------------------------------- #

HOVER_ALT_M = 5.0
HOVER_DURATION_S = 60.0
SAMPLE_RATE_HZ = 10.0

# Pass criterion. The dissertation Ch 5 T4 spec is "hover within 50 cm of
# the commanded position over a sustained interval". 50 cm is a SITL-noise
# headroom number; real-bird is GPS-limited (~1-2 m without RTK).
HORIZONTAL_DRIFT_BOUND_M = 0.5
VERTICAL_DRIFT_BOUND_M = 0.5

# Settle time after takeoff before switching to LOITER. Lets the GUIDED
# climb finish + integrator wind-down.
SETTLE_S = 3.0

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
    """Convert (lat, lon) to local (north_m, east_m) relative to a reference.

    Uses two haversine evaluations (one with longitude fixed, one with
    latitude fixed) so the components are independent. Sign-aware.
    """
    n = haversine_m(ref_lat, ref_lon, lat, ref_lon)
    if lat < ref_lat:
        n = -n
    e = haversine_m(ref_lat, ref_lon, ref_lat, lon)
    if lon < ref_lon:
        e = -e
    return n, e


@dataclass
class HoverSample:
    t_offset: float
    lat: float
    lon: float
    alt_rel: float
    mode: str


@dataclass
class HoverStats:
    n_samples: int = 0
    duration_s: float = 0.0
    ref_lat: float = 0.0
    ref_lon: float = 0.0
    ref_alt_m: float = 0.0
    max_horizontal_drift_m: float = 0.0
    rms_horizontal_drift_m: float = 0.0
    max_vertical_drift_m: float = 0.0
    rms_vertical_drift_m: float = 0.0
    horizontal_bound_m: float = HORIZONTAL_DRIFT_BOUND_M
    vertical_bound_m: float = VERTICAL_DRIFT_BOUND_M
    horizontal_pass: bool = False
    vertical_pass: bool = False


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


def compute_stats(samples: list[HoverSample]) -> HoverStats:
    """Compute drift stats relative to the first valid sample."""
    stats = HoverStats()
    if not samples:
        return stats
    ref = samples[0]
    stats.ref_lat = ref.lat
    stats.ref_lon = ref.lon
    stats.ref_alt_m = ref.alt_rel
    stats.n_samples = len(samples)
    stats.duration_s = samples[-1].t_offset - samples[0].t_offset

    h_sq_sum = 0.0
    v_sq_sum = 0.0
    h_max = 0.0
    v_max = 0.0
    for s in samples:
        n, e = latlon_to_local_ne(s.lat, s.lon, ref.lat, ref.lon)
        h = math.hypot(n, e)
        v = abs(s.alt_rel - ref.alt_rel)
        h_sq_sum += h * h
        v_sq_sum += v * v
        h_max = max(h_max, h)
        v_max = max(v_max, v)
    stats.max_horizontal_drift_m = h_max
    stats.max_vertical_drift_m = v_max
    stats.rms_horizontal_drift_m = math.sqrt(h_sq_sum / len(samples))
    stats.rms_vertical_drift_m = math.sqrt(v_sq_sum / len(samples))
    stats.horizontal_pass = h_max < HORIZONTAL_DRIFT_BOUND_M
    stats.vertical_pass = v_max < VERTICAL_DRIFT_BOUND_M
    return stats


def write_outputs(
    samples: list[HoverSample],
    stats: HoverStats,
    out_dir: Path,
    stamp: str,
) -> tuple[Path, Path | None]:
    """Write JSON + optional matplotlib PNG. Returns (json_path, png_path|None)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"t4_hover_{stamp}.json"
    payload = {
        "test": "T4_hover_stability",
        "timestamp": stamp,
        "stats": asdict(stats),
        "samples": [asdict(s) for s in samples],
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

    if not samples:
        return json_path, None

    ref = samples[0]
    times = [s.t_offset for s in samples]
    ne = [latlon_to_local_ne(s.lat, s.lon, ref.lat, ref.lon) for s in samples]
    norths = [p[0] for p in ne]
    easts = [p[1] for p in ne]
    horiz = [math.hypot(n, e) for n, e in ne]
    valts = [s.alt_rel - ref.alt_rel for s in samples]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(easts, norths, "-", linewidth=0.7, alpha=0.7)
    ax.plot(easts, norths, ".", markersize=2)
    ax.plot([0], [0], "r+", markersize=12, label="hover ref")
    bound = HORIZONTAL_DRIFT_BOUND_M
    theta = [i * 2 * math.pi / 100 for i in range(101)]
    ax.plot(
        [bound * math.cos(t) for t in theta],
        [bound * math.sin(t) for t in theta],
        "r--",
        linewidth=1,
        label=f"±{bound} m bound",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("LOITER trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(times, horiz, "-")
    ax.axhline(HORIZONTAL_DRIFT_BOUND_M, color="r", linestyle="--", label=f"{HORIZONTAL_DRIFT_BOUND_M} m bound")
    ax.set_xlabel("t since LOITER (s)")
    ax.set_ylabel("Horizontal drift (m)")
    ax.set_title(
        f"Horizontal drift  (max={stats.max_horizontal_drift_m:.2f} m, "
        f"RMS={stats.rms_horizontal_drift_m:.2f} m)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(times, valts, "-")
    ax.axhline(VERTICAL_DRIFT_BOUND_M, color="r", linestyle="--", alpha=0.6)
    ax.axhline(-VERTICAL_DRIFT_BOUND_M, color="r", linestyle="--", alpha=0.6, label=f"±{VERTICAL_DRIFT_BOUND_M} m bound")
    ax.set_xlabel("t since LOITER (s)")
    ax.set_ylabel("Δalt (m)")
    ax.set_title(
        f"Vertical drift  (max={stats.max_vertical_drift_m:.2f} m, "
        f"RMS={stats.rms_vertical_drift_m:.2f} m)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"T4 hover stability — {stamp}")
    fig.tight_layout()
    png_path = out_dir / f"t4_hover_{stamp}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return json_path, png_path


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


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

    fsm = StateMachine(v, poll_hz=10.0)

    def on_fsm(old: FlightState, new: FlightState, reason) -> None:
        log(f"FSM: {old} -> {new}" + (f"  ({reason})" if reason else ""))

    fsm.subscribe(on_fsm)
    fsm.start()

    samples: list[HoverSample] = []

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

        log(f"Taking off to {HOVER_ALT_M} m ...")
        v.takeoff(HOVER_ALT_M)
        log(f"  reached alt={v.state.alt_rel:.2f} m")

        log(f"Settling for {SETTLE_S} s ...")
        time.sleep(SETTLE_S)

        # Stay in GUIDED. After NAV_TAKEOFF, ArduCopter holds the takeoff
        # lat/lon/alt as an active position target. Do NOT switch to LOITER
        # in headless SITL — see the docstring.
        log(f"Holding hover in GUIDED mode (mode={v.state.mode}) ...")

        # ------------------------------------------------------------ #
        # Sample loop                                                  #
        # ------------------------------------------------------------ #

        sample_period = 1.0 / SAMPLE_RATE_HZ
        t0 = time.time()
        next_sample = t0
        log(
            f"Sampling at {SAMPLE_RATE_HZ:.0f} Hz for {HOVER_DURATION_S} s "
            f"(expect ~{int(SAMPLE_RATE_HZ * HOVER_DURATION_S)} samples) ..."
        )
        while time.time() < t0 + HOVER_DURATION_S:
            now = time.time()
            if now >= next_sample:
                st = v.state
                if st.lat is not None and st.lon is not None and st.alt_rel is not None:
                    samples.append(
                        HoverSample(
                            t_offset=now - t0,
                            lat=st.lat,
                            lon=st.lon,
                            alt_rel=st.alt_rel,
                            mode=st.mode or "?",
                        )
                    )
                next_sample = now + sample_period
            time.sleep(0.005)

        log(f"Captured {len(samples)} samples over {time.time() - t0:.1f} s")

        # ------------------------------------------------------------ #
        # Land + disarm                                                #
        # ------------------------------------------------------------ #

        log("Switching to LAND ...")
        v.land()
        log("Waiting for disarm (up to 60 s) ...")
        deadline = time.time() + 60
        while time.time() < deadline and v.state.armed:
            time.sleep(0.5)
        if v.state.armed:
            log("WARN: did not disarm within 60 s (test result still valid)")
        else:
            log(f"  disarmed. final state: mode={v.state.mode}")

        # ------------------------------------------------------------ #
        # Analysis + outputs                                           #
        # ------------------------------------------------------------ #

        stats = compute_stats(samples)
        out_dir = (_AUTONOMY_DIR / ".." / "sitl" / "logs").resolve()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        json_path, png_path = write_outputs(samples, stats, out_dir, stamp)

        print()
        log("=" * 64)
        log("T4 hover stability — results")
        log("=" * 64)
        log(f"  Samples              :  {stats.n_samples}  over {stats.duration_s:.1f} s")
        log(f"  Max horizontal drift :  {stats.max_horizontal_drift_m:.3f} m   "
            f"(bound: {HORIZONTAL_DRIFT_BOUND_M} m)")
        log(f"  RMS horizontal drift :  {stats.rms_horizontal_drift_m:.3f} m")
        log(f"  Max vertical drift   :  {stats.max_vertical_drift_m:.3f} m   "
            f"(bound: {VERTICAL_DRIFT_BOUND_M} m)")
        log(f"  RMS vertical drift   :  {stats.rms_vertical_drift_m:.3f} m")
        log(f"  Output JSON          :  {json_path}")
        if png_path is not None:
            log(f"  Output PNG           :  {png_path}")
        else:
            log("  (matplotlib not available — install it to also write a PNG)")

        print()
        log(f"  Horizontal drift < {HORIZONTAL_DRIFT_BOUND_M} m :  "
            f"{'PASS' if stats.horizontal_pass else 'FAIL'}")
        log(f"  Vertical drift   < {VERTICAL_DRIFT_BOUND_M} m :  "
            f"{'PASS' if stats.vertical_pass else 'FAIL'}")
        if not (stats.horizontal_pass and stats.vertical_pass):
            log("FAIL: T4 hover stability did not meet acceptance criteria.")
            return 1
        log("PASS: T4 hover stability demonstrated in SITL.")
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
