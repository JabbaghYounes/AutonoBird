#!/usr/bin/env python3
"""
test_sik_link.py - Comprehensive SiK 433 MHz telemetry link test.

Runs a sequence of bench validations and an interactive walk-out range
test against the AutonoBird flight controller, via the SiK ground unit
on the dev workstation's USB.

Modes:
  bench    Quick bench validation (~30 s): enumeration, heartbeat,
           RADIO_STATUS baseline, parameter round-trip latency.
  walkout  Interactive walk-out range test only: continuous
           RADIO_STATUS CSV logging with operator-entered distance
           markers.
  full     bench + walkout (default).

Usage:
  cd ~/Documents/AutonoBird/scripts/flight-controller
  source venv/bin/activate

  # Full test (default mode):
  python3 test_sik_link.py

  # Bench only (quick health check, ~30 s):
  python3 test_sik_link.py --mode bench

  # Walkout only (skip bench, go straight to range testing):
  python3 test_sik_link.py --mode walkout

  # Override port / baud:
  python3 test_sik_link.py --port /dev/ttyUSB1 --baud 57600

During walkout:
  - The script writes every RADIO_STATUS sample to
    scripts/sitl/logs/sik_los_<timestamp>.csv (line-buffered).
  - Type "m <distance_m>" + Enter (or just "<distance_m>") to tag the
    next sample as a marker waypoint. The script prints the live RSSI
    when the marker lands.
  - Type "q" + Enter (or Ctrl+C) to end the walk; a post-walk summary
    is printed and the CSV path is reported.

Prerequisites:
  - SiK ground unit on /dev/ttyUSB0 (or override with --port)
  - Air unit powered (FC connected to LiPo or USB-C power)
  - pymavlink + pyserial in the flight-controller venv (setup.sh does
    pymavlink; this script adds pyserial if missing, see imports).
"""

from __future__ import annotations

import argparse
import csv
import queue
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn, cast

try:
    from pymavlink import mavutil
except ImportError:
    print(
        "ERROR: pymavlink not installed. Activate the flight-controller "
        "venv first:\n  cd ~/Documents/AutonoBird/scripts/flight-controller "
        "&& source venv/bin/activate"
    )
    sys.exit(1)


# ---------------------------------------------------------------------- #
# Config                                                                 #
# ---------------------------------------------------------------------- #

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 57600
DEFAULT_BENCH_SAMPLE_S = 15
DEFAULT_PARAM_TRIALS = 5
LOG_DIR = Path.home() / "Documents" / "AutonoBird" / "scripts" / "sitl" / "logs"


# ---------------------------------------------------------------------- #
# Output helpers                                                         #
# ---------------------------------------------------------------------- #


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def fail(msg: str) -> NoReturn:
    print(f"[FAIL] {msg}", flush=True)
    sys.exit(1)


def section(title: str) -> None:
    print()
    print("-" * 68)
    print(f"  {title}")
    print("-" * 68)


# ---------------------------------------------------------------------- #
# Connection                                                             #
# ---------------------------------------------------------------------- #


def connect(port: str, baud: int) -> "mavutil.mavfile":
    if not Path(port).exists():
        fail(
            f"{port} not found. Plug in the SiK ground unit and check "
            f"'lsusb' + 'ls /dev/ttyUSB* /dev/ttyACM*'."
        )
    info(f"connecting to {port} @ {baud} baud ...")
    # mavlink_connection's return is a union of several reader types
    # for log replay vs live links; for a real serial device it's mavserial
    # which is a subclass of mavfile and exposes wait_heartbeat().
    master = cast(
        "mavutil.mavfile",
        mavutil.mavlink_connection(port, baud=baud, source_system=255),
    )
    info("waiting for heartbeat (timeout 15 s) ...")
    hb = master.wait_heartbeat(timeout=15)
    if hb is None:
        fail(
            "no heartbeat received. Possible causes:\n"
            "  - Air unit not powered (FC needs LiPo or USB-C power)\n"
            "  - Air unit not on TELEM1 (check the cable)\n"
            "  - Air + ground radios not paired (check NetID / air rate)\n"
            "  - Distance too great / antenna orientation bad"
        )
    ok(
        f"heartbeat from autopilot sysid={hb.get_srcSystem()}  "
        f"compid={hb.get_srcComponent()}"
    )
    return master


# ---------------------------------------------------------------------- #
# Bench tests                                                            #
# ---------------------------------------------------------------------- #


def bench_radio_baseline(master, duration_s: int) -> list[dict]:
    info(f"sampling RADIO_STATUS for {duration_s} s ...")
    samples: list[dict] = []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        msg = master.recv_match(type="RADIO_STATUS", blocking=True, timeout=5)
        if msg is None:
            warn(
                "no RADIO_STATUS in 5 s — link may be one-way or down; "
                "still listening ..."
            )
            continue
        samples.append(
            {
                "rssi": msg.rssi,
                "remrssi": msg.remrssi,
                "noise": msg.noise,
                "remnoise": msg.remnoise,
                "rxerrors": msg.rxerrors,
                "fixed": msg.fixed,
                "txbuf": msg.txbuf,
            }
        )

    if not samples:
        fail(
            "zero RADIO_STATUS samples collected — heartbeat is flowing but "
            "the radio is not framing RADIO_STATUS. Check SiK firmware on "
            "both ends."
        )

    n = len(samples)
    ok(f"collected {n} RADIO_STATUS samples")
    print()
    print(f"  {'Field':12s}  {'min':>6s}  {'median':>6s}  {'max':>6s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}")
    for field in ("rssi", "remrssi", "noise", "remnoise", "txbuf"):
        vals = [s[field] for s in samples]
        print(
            f"  {field:12s}  {min(vals):>6d}  "
            f"{int(statistics.median(vals)):>6d}  {max(vals):>6d}"
        )

    err_total = samples[-1]["rxerrors"] - samples[0]["rxerrors"]
    fix_total = samples[-1]["fixed"] - samples[0]["fixed"]
    print()
    print(f"  cumulative rxerrors in window: {err_total}")
    print(f"  cumulative fixed errors in window: {fix_total}")

    median_rssi = int(statistics.median([s["rssi"] for s in samples]))
    median_remrssi = int(statistics.median([s["remrssi"] for s in samples]))

    print()
    if median_rssi > 180 and median_remrssi > 180:
        ok("link strength: STRONG (bench-distance baseline)")
    elif median_rssi > 100 and median_remrssi > 100:
        warn("link strength: MARGINAL at bench distance — check antenna "
             "orientation / placement")
    else:
        warn("link strength: WEAK at bench distance — likely antenna or "
             "interference issue")
    return samples


def bench_param_latency(master, n_trials: int = DEFAULT_PARAM_TRIALS) -> None:
    info(
        f"measuring param round-trip latency "
        f"({n_trials} trials of STAT_RUNTIME) ..."
    )

    latencies_ms: list[float] = []
    for i in range(n_trials):
        # Drain any pending PARAM_VALUE so we time only our own request.
        while master.recv_match(type="PARAM_VALUE", blocking=False) is not None:
            pass

        t0 = time.time()
        master.mav.param_request_read_send(
            master.target_system,
            master.target_component,
            b"STAT_RUNTIME",
            -1,
        )
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=10)
        if msg is None:
            warn(f"trial {i+1}: no PARAM_VALUE response within 10 s")
            continue
        latencies_ms.append((time.time() - t0) * 1000.0)
        time.sleep(0.2)

    if not latencies_ms:
        fail(
            "no PARAM_VALUE responses — link is broken in one direction. "
            "Heartbeat is flowing autopilot -> GCS but GCS -> autopilot is "
            "not getting through."
        )

    ok(f"{len(latencies_ms)}/{n_trials} trials succeeded")
    print()
    print(f"  min:    {min(latencies_ms):>6.1f} ms")
    print(f"  median: {statistics.median(latencies_ms):>6.1f} ms")
    print(f"  max:    {max(latencies_ms):>6.1f} ms")


# ---------------------------------------------------------------------- #
# Walk-out (interactive)                                                 #
# ---------------------------------------------------------------------- #


def input_thread(q: "queue.Queue") -> None:
    """Read lines from stdin, push markers / quit signals onto the queue."""
    while True:
        try:
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            q.put("QUIT")
            return
        if not line:
            continue
        low = line.lower()
        if low in ("q", "quit", "exit"):
            q.put("QUIT")
            return
        parts = line.split()
        try:
            if parts[0].lower() == "m" and len(parts) >= 2:
                q.put(float(parts[1]))
            else:
                q.put(float(parts[0]))
        except (ValueError, IndexError):
            warn(
                f"unrecognised input {line!r}. Use 'm <distance_m>' "
                f"(e.g. 'm 50') or just '<distance_m>' or 'q' to quit."
            )


def walkout(master, csv_path: Path, marker_q: "queue.Queue") -> int:
    info(f"walk-out: streaming RADIO_STATUS to {csv_path}")
    info(
        "controls: type 'm <distance_m>' + Enter to mark a waypoint, "
        "'q' + Enter (or Ctrl+C) to end"
    )
    print()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        [
            "t_iso",
            "t_unix",
            "rssi",
            "remrssi",
            "noise",
            "remnoise",
            "rxerrors",
            "fixed",
            "txbuf",
            "marker_distance_m",
        ]
    )
    f.flush()

    samples_written = 0
    next_marker_distance: float | None = None

    try:
        while True:
            # Consume any pending command from the input thread.
            try:
                cmd = marker_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd == "QUIT":
                break
            if isinstance(cmd, float):
                next_marker_distance = cmd
                info(
                    f"  >> next sample will be tagged as marker at "
                    f"{next_marker_distance:.0f} m"
                )

            msg = master.recv_match(
                type="RADIO_STATUS", blocking=True, timeout=3
            )
            if msg is None:
                warn(
                    "  >> no RADIO_STATUS in 3 s — possible link drop, "
                    "still listening ..."
                )
                continue

            t = time.time()
            t_iso = datetime.fromtimestamp(t).isoformat(timespec="seconds")
            marker_field = (
                f"{next_marker_distance:.0f}"
                if next_marker_distance is not None
                else ""
            )
            writer.writerow(
                [
                    t_iso,
                    f"{t:.3f}",
                    msg.rssi,
                    msg.remrssi,
                    msg.noise,
                    msg.remnoise,
                    msg.rxerrors,
                    msg.fixed,
                    msg.txbuf,
                    marker_field,
                ]
            )
            f.flush()
            samples_written += 1

            if next_marker_distance is not None:
                ok(
                    f"  marker logged at {next_marker_distance:.0f} m  |  "
                    f"rssi={msg.rssi}  remrssi={msg.remrssi}  "
                    f"noise={msg.noise}  rxerrors={msg.rxerrors}"
                )
                next_marker_distance = None
            elif samples_written % 5 == 0:
                # Periodic feedback so the operator knows samples are
                # landing without flooding the terminal.
                print(
                    f"  [{samples_written:4d}] rssi={msg.rssi:3d}  "
                    f"remrssi={msg.remrssi:3d}  noise={msg.noise:3d}  "
                    f"rxerrors={msg.rxerrors}  fixed={msg.fixed}"
                )
    except KeyboardInterrupt:
        print()
        info("KeyboardInterrupt - stopping walkout")
    finally:
        f.close()

    ok(f"walkout ended: {samples_written} samples to {csv_path}")
    return samples_written


# ---------------------------------------------------------------------- #
# Post-walk analysis                                                     #
# ---------------------------------------------------------------------- #


def analyse_csv(csv_path: Path) -> None:
    info(f"analysing {csv_path} ...")
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        warn("CSV is empty — no samples to analyse")
        return

    markers = [r for r in rows if r.get("marker_distance_m")]

    print()
    print(f"  total samples: {len(rows)}")
    print(f"  markers logged: {len(markers)}")

    if markers:
        print()
        print(
            f"  {'distance':>9s}  {'rssi':>5s}  {'remrssi':>8s}  "
            f"{'noise':>6s}  {'remnoise':>8s}  {'rxerrors':>8s}  "
            f"{'fixed':>5s}"
        )
        print(
            f"  {'-'*9}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*8}  "
            f"{'-'*8}  {'-'*5}"
        )
        for m in markers:
            d = float(m["marker_distance_m"])
            print(
                f"  {d:>7.0f} m  "
                f"{int(m['rssi']):>5d}  "
                f"{int(m['remrssi']):>8d}  "
                f"{int(m['noise']):>6d}  "
                f"{int(m['remnoise']):>8d}  "
                f"{int(m['rxerrors']):>8d}  "
                f"{int(m['fixed']):>5d}"
            )
        last = markers[-1]
        print()
        ok(
            f"furthest marker: {float(last['marker_distance_m']):.0f} m  "
            f"at RSSI {int(last['rssi'])}  (remRSSI {int(last['remrssi'])})"
        )

    all_rssi = [int(r["rssi"]) for r in rows]
    all_remrssi = [int(r["remrssi"]) for r in rows]
    print()
    print(
        f"  RSSI over whole walk:    min={min(all_rssi):3d}  "
        f"median={int(statistics.median(all_rssi)):3d}  "
        f"max={max(all_rssi):3d}"
    )
    print(
        f"  remRSSI over whole walk: min={min(all_remrssi):3d}  "
        f"median={int(statistics.median(all_remrssi)):3d}  "
        f"max={max(all_remrssi):3d}"
    )

    first_err = int(rows[0]["rxerrors"])
    last_err = int(rows[-1]["rxerrors"])
    first_fix = int(rows[0]["fixed"])
    last_fix = int(rows[-1]["fixed"])
    print()
    print(f"  rxerrors accumulated:  {last_err - first_err}")
    print(f"  fixed errors:           {last_fix - first_fix}")


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Comprehensive SiK 433 MHz telemetry link test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        choices=["bench", "walkout", "full"],
        default="full",
        help="bench: connectivity + RADIO_STATUS + param latency (~30 s). "
             "walkout: interactive walk-out range test with CSV log. "
             "full: both (default).",
    )
    ap.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"serial device for the SiK ground unit (default {DEFAULT_PORT}).",
    )
    ap.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"baud rate (default {DEFAULT_BAUD}).",
    )
    ap.add_argument(
        "--bench-duration",
        type=int,
        default=DEFAULT_BENCH_SAMPLE_S,
        help=f"seconds to sample RADIO_STATUS during bench "
             f"(default {DEFAULT_BENCH_SAMPLE_S}).",
    )
    args = ap.parse_args()

    print("=" * 68)
    print(f"  AutonoBird - SiK telemetry link test  (mode={args.mode})")
    print("=" * 68)

    master = connect(args.port, args.baud)

    if args.mode in ("bench", "full"):
        section("BENCH: RADIO_STATUS BASELINE")
        bench_radio_baseline(master, args.bench_duration)

        section("BENCH: PARAMETER ROUND-TRIP LATENCY")
        bench_param_latency(master)

    if args.mode in ("walkout", "full"):
        section("WALKOUT (interactive)")
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        csv_path = LOG_DIR / f"sik_los_{ts}.csv"

        q: "queue.Queue" = queue.Queue()
        t = threading.Thread(target=input_thread, args=(q,), daemon=True)
        t.start()

        walkout(master, csv_path, q)

        section("WALKOUT: ANALYSIS")
        analyse_csv(csv_path)
        print()
        info(f"CSV saved at: {csv_path}")

    print()
    ok("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
