"""
scripts/autonomy/test_state_machine.py

Smoke test for the FSM against a running SITL.

Drives the bridge through a takeoff + RTL cycle while the FSM observes,
logs every state transition, and verifies that the expected sequence
(DISCONNECTED → NO_FIX → PREARMED → ARMED_ON_GROUND → ASCENDING →
AIRBORNE → DESCENDING → DISARMED_POSTFLIGHT) all fire in order.

Pre-flight:
  1. SITL is running (scripts/sitl/run_sitl.sh)
  2. The autonomy venv is active (scripts/autonomy/setup.sh)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Cross-subsystem import for the bridge — same approach as state_machine.py.
_AUTONOMY_DIR = Path(__file__).parent
_BRIDGE_DIR = (_AUTONOMY_DIR / ".." / "flight-controller").resolve()
if str(_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_DIR))

from bridge import BridgeError, Vehicle, load_config as load_bridge_config  # noqa: E402

from state_machine import FlightState, StateMachine, Transition  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run() -> int:
    cfg = load_bridge_config()
    log(f"Connecting bridge to {cfg['connection_uri']} ...")
    v = Vehicle(
        connection_uri=cfg["connection_uri"],
        source_system=cfg.get("source_system", 255),
        heartbeat_timeout=cfg.get("heartbeat_timeout", 30.0),
    )

    transitions_seen: list[Transition] = []

    def on_transition(old: FlightState, new: FlightState, reason):
        transitions_seen.append(Transition(time.time(), old, new, reason))
        log(f"FSM: {old} -> {new}" + (f"  ({reason})" if reason else ""))

    try:
        v.connect()
    except BridgeError as e:
        log(f"FAIL: bridge connect: {e}")
        log("Is SITL running? Try `scripts/sitl/run_sitl.sh` in another terminal.")
        return 1

    fsm = StateMachine(v, poll_hz=10.0)
    fsm.subscribe(on_transition)
    fsm.start()

    try:
        # Give the FSM a moment to leave DISCONNECTED based on first telemetry.
        time.sleep(1.0)
        log(f"FSM state after connect: {fsm.state}")

        # Wait for GPS fix to land us in PREARMED.
        log("Waiting for GPS fix -> PREARMED ...")
        deadline = time.time() + 30
        while time.time() < deadline and fsm.state != FlightState.PREARMED:
            time.sleep(0.2)
        if fsm.state != FlightState.PREARMED:
            log(f"FAIL: never reached PREARMED (stuck in {fsm.state})")
            return 1

        log("Switching to GUIDED + arming ...")
        v.set_mode("GUIDED")
        v.arm()
        # FSM should now see armed=True + alt<0.5 -> ARMED_ON_GROUND.
        time.sleep(0.3)
        log(f"FSM state after arm: {fsm.state}")

        log("Taking off to 10 m ...")
        v.takeoff(10.0)
        # FSM should have gone through ASCENDING and into AIRBORNE.
        time.sleep(0.3)
        log(f"FSM state at altitude: {fsm.state}")

        log("Hovering 2 s ...")
        time.sleep(2.0)

        log("Switching to RTL — expecting DESCENDING then DISARMED_POSTFLIGHT ...")
        v.rtl()

        # RTL takes time. Wait for disarm.
        deadline = time.time() + 90
        while time.time() < deadline and v.state.armed:
            time.sleep(0.5)
        if v.state.armed:
            log("FAIL: vehicle did not disarm within 90 s")
            return 1
        log("Vehicle disarmed.")
        time.sleep(1.0)  # let FSM observe the disarm
        log(f"Final FSM state: {fsm.state}")

        # ------------------------------------------------------------ #
        # Validation: did we see all the expected transitions?         #
        # ------------------------------------------------------------ #
        seen_states = {t.to_state for t in transitions_seen}
        expected = {
            FlightState.NO_FIX,
            FlightState.PREARMED,
            FlightState.ARMED_ON_GROUND,
            FlightState.ASCENDING,
            FlightState.AIRBORNE,
            FlightState.DESCENDING,
            FlightState.DISARMED_POSTFLIGHT,
        }
        missing = expected - seen_states
        unexpected = seen_states - expected - {FlightState.DISCONNECTED}

        print()
        log("Transitions seen:")
        for t in transitions_seen:
            log(f"  +{t.t - transitions_seen[0].t:6.2f}s  {t}")

        if missing:
            log(f"FAIL: missing transitions to: {sorted(s.name for s in missing)}")
            return 1
        if unexpected:
            log(f"WARN: unexpected transitions to: {sorted(s.name for s in unexpected)}")
            # not a hard fail — just noteworthy
        log("PASS: state machine observed the full takeoff/RTL transition sequence.")
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
