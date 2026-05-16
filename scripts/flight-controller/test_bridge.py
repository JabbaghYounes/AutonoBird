"""
scripts/flight-controller/test_bridge.py

Smoke test for the MAVLink bridge against a running ArduPilot SITL.

Pre-flight:
  1. SITL is up (e.g. `scripts/sitl/run_sitl.sh` from another terminal)
  2. The bridge venv is active or the system python has pymavlink

What it does:
  * connect to SITL on tcp:127.0.0.1:5760 (override in config.json)
  * wait for GPS fix
  * mode GUIDED -> arm -> takeoff 10m -> mode RTL -> wait for land/disarm
  * drains and prints events along the way
  * disconnect

Exits 0 on success, 1 on any failure with the exception message.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running this file directly as `python test_bridge.py` without
# requiring a package install.
sys.path.insert(0, str(Path(__file__).parent))

from bridge import BridgeError, Vehicle, load_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_gps(v: Vehicle, timeout: float = 30.0) -> None:
    log("Waiting for GPS fix...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = v.state
        if s.gps_fix is not None and s.gps_fix >= 3 and s.lat not in (None, 0):
            log(f"  GPS fix={s.gps_fix} sats={s.satellites} "
                f"lat={s.lat:.6f} lon={s.lon:.6f}")
            return
        time.sleep(0.5)
    raise BridgeError(f"GPS fix not acquired in {timeout}s")


def drain_and_print_events(v: Vehicle, header: str) -> None:
    evts = v.drain_events()
    if not evts:
        return
    log(f"  {header}: {len(evts)} event(s)")
    for e in evts[-8:]:  # last 8 to keep output manageable
        log(f"    -> {e.kind:13s} {e.payload}")


def run() -> int:
    cfg = load_config()
    log(f"Connecting to {cfg['connection_uri']} ...")
    v = Vehicle(
        connection_uri=cfg["connection_uri"],
        source_system=cfg.get("source_system", 255),
        heartbeat_timeout=cfg.get("heartbeat_timeout", 30.0),
    )
    takeoff_alt = float(cfg.get("default_takeoff_alt_m", 10.0))

    try:
        v.connect()
    except BridgeError as e:
        log(f"FAIL: {e}")
        log("Is SITL running? Try `scripts/sitl/run_sitl.sh` in another terminal.")
        return 1

    try:
        s = v.state
        log(f"Connected. mode={s.mode} armed={s.armed} system_status={s.system_status}")

        wait_for_gps(v)
        drain_and_print_events(v, "post-GPS")

        log("Switching to GUIDED ...")
        v.set_mode("GUIDED")
        log(f"  mode={v.state.mode}")

        log("Arming ...")
        v.arm()
        log(f"  armed={v.state.armed}")
        drain_and_print_events(v, "post-arm")

        log(f"Takeoff to {takeoff_alt} m ...")
        v.takeoff(takeoff_alt)
        s = v.state
        log(f"  reached alt={s.alt_rel:.2f} m (target {takeoff_alt} m)")

        # Brief hover so we can observe steady state.
        hover_secs = 3.0
        log(f"Hovering for {hover_secs}s ...")
        time.sleep(hover_secs)
        s = v.state
        log(f"  state: alt={s.alt_rel:.2f}m  hdg={s.heading_deg}  speed={s.ground_speed}m/s")

        log("Switching to RTL ...")
        v.rtl()
        log(f"  mode={v.state.mode}")
        drain_and_print_events(v, "post-RTL")

        log("Waiting for landing + disarm (up to 90 s) ...")
        deadline = time.time() + 90.0
        while time.time() < deadline:
            if not v.state.armed:
                break
            time.sleep(0.5)
        else:
            raise BridgeError("Did not disarm within 90s of RTL")

        log(f"  disarmed. final state: mode={v.state.mode} alt={v.state.alt_rel:.2f}m")
        drain_and_print_events(v, "post-land")

        log("PASS: smoke test completed end-to-end.")
        return 0

    except BridgeError as e:
        log(f"FAIL: {e}")
        return 1
    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130
    finally:
        log("Disconnecting ...")
        v.disconnect()


if __name__ == "__main__":
    sys.exit(run())
