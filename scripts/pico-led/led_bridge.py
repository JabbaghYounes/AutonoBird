"""
scripts/pico-led/led_bridge.py — Pi-side LED driver

Bridges autonomy `StateMachine` events to the Pico 2 W LED firmware running
`scripts/pico-led/main.py`. Protocol matches main.py: line-based ASCII over
USB-serial.

Three ways to run it:

    # 1. Standalone state cycle (no SITL / FSM needed) — verifies the Pico
    #    + cable + firmware are alive without touching the flight stack.
    python led_bridge.py --test

    # 2. Drive a single state once and exit.
    python led_bridge.py --state ASCENDING

    # 3. Live FSM mirror: connect to SITL or the real Pixhawk through
    #    the autonomy bridge, subscribe to StateMachine, push transitions
    #    to the Pico for as long as the link is up.
    python led_bridge.py --watch-fsm

The LedBridge class can also be imported directly by the orchestrator:

    from led_bridge import LedBridge
    led = LedBridge(port="/dev/ttyACM0")
    led.connect()
    fsm.subscribe(lambda old, new, reason: led.set_state(new.name))
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import serial  # pyserial
except ImportError as e:  # pragma: no cover
    print(
        "pyserial is required. Install via:  pip install pyserial",
        file=sys.stderr,
    )
    raise


# Cross-subsystem imports for --watch-fsm. Optional — only loaded when needed.
_THIS_DIR = Path(__file__).resolve().parent
_AUTONOMY_DIR = (_THIS_DIR / ".." / "autonomy").resolve()
_BRIDGE_DIR = (_THIS_DIR / ".." / "flight-controller").resolve()


DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
READY_TIMEOUT_S = 5.0       # how long to wait for the Pico's READY banner

# FSM states the Pico knows about — kept in sync with main.py and
# state_machine.FlightState. Used by --test to cycle through them.
KNOWN_STATES = [
    "DISCONNECTED",
    "NO_FIX",
    "PREARMED",
    "ARMED_ON_GROUND",
    "ASCENDING",
    "AIRBORNE",
    "DESCENDING",
    "DISARMED_POSTFLIGHT",
    "FAULT",
]


class LedBridgeError(RuntimeError):
    """Raised when the bridge can't reach the Pico."""


class LedBridge:
    """Pi → Pico serial driver. One instance per Pico."""

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        ready_timeout_s: float = READY_TIMEOUT_S,
        verbose: bool = False,
    ):
        self.port = port
        self.baud = baud
        self.ready_timeout_s = ready_timeout_s
        self.verbose = verbose
        self._ser: Optional["serial.Serial"] = None
        self._lock = threading.Lock()
        self._current_state: Optional[str] = None

    # ---- lifecycle ---- #

    def connect(self) -> None:
        """Open the serial port and wait for the Pico's READY banner.

        If the banner isn't seen within ready_timeout_s, we still consider
        the connection live — older firmware (or a Pico already past boot)
        may not emit it. We log a warning instead of raising.
        """
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except serial.SerialException as e:
            raise LedBridgeError(f"could not open {self.port}: {e}") from e

        # Some USB stacks deliver a flurry of garbage right after open.
        # Give it a beat, then drain anything pending.
        time.sleep(0.1)
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

        # Wait for READY (best-effort).
        deadline = time.time() + self.ready_timeout_s
        saw_ready = False
        while time.time() < deadline:
            line = self._read_line(timeout=0.2)
            if line is None:
                continue
            if self.verbose:
                print(f"[led-bridge] pico>>  {line!r}")
            if line.strip() == "READY":
                saw_ready = True
                break
        if not saw_ready and self.verbose:
            print(
                "[led-bridge] note: did not see READY banner — assuming "
                "firmware is alive anyway"
            )

    def close(self) -> None:
        if self._ser is not None:
            try:
                self.off()
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def __enter__(self) -> "LedBridge":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- commands ---- #

    def set_state(self, state_name: str) -> None:
        """Push a new FSM state to the Pico. Idempotent — repeats are no-ops."""
        if state_name == self._current_state:
            return
        self._write_line(f"STATE {state_name}")
        self._current_state = state_name

    def off(self) -> None:
        self._write_line("OFF")
        self._current_state = "DISCONNECTED"

    def set_brightness(self, value: int) -> None:
        if not (0 <= value <= 255):
            raise ValueError("brightness must be 0..255")
        self._write_line(f"BRIGHTNESS {value}")

    def ping(self, timeout_s: float = 1.0) -> bool:
        self._write_line("PING")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self._read_line(timeout=0.1)
            if line is not None and line.strip() == "PONG":
                return True
        return False

    # ---- internals ---- #

    def _write_line(self, line: str) -> None:
        if self._ser is None:
            raise LedBridgeError("not connected — call connect() first")
        payload = (line + "\n").encode("ascii", errors="replace")
        with self._lock:
            try:
                self._ser.write(payload)
                self._ser.flush()
            except serial.SerialException as e:
                raise LedBridgeError(f"write failed: {e}") from e
        if self.verbose:
            print(f"[led-bridge] pi>>    {line}")

    def _read_line(self, timeout: float = 0.2) -> Optional[str]:
        if self._ser is None:
            return None
        self._ser.timeout = timeout
        try:
            raw = self._ser.readline()
        except serial.SerialException:
            return None
        if not raw:
            return None
        try:
            return raw.decode("ascii", errors="replace").rstrip("\r\n")
        except Exception:
            return None


# --------------------------------------------------------------------- #
# Standalone CLI                                                        #
# --------------------------------------------------------------------- #


def load_config(path: Path) -> dict:
    """Read scripts/pico-led/config.json (or .example.json) if present."""
    if path.is_file():
        with path.open() as f:
            return json.load(f)
    example = path.with_suffix("").with_name(path.stem + ".example.json")
    if example.is_file():
        with example.open() as f:
            return json.load(f)
    return {}


def run_test_cycle(led: LedBridge, dwell_s: float) -> None:
    print(f"[led-bridge] cycling {len(KNOWN_STATES)} states, {dwell_s:.1f} s each")
    try:
        for s in KNOWN_STATES:
            print(f"[led-bridge]   STATE {s}")
            led.set_state(s)
            time.sleep(dwell_s)
    finally:
        led.off()


def run_watch_fsm(led: LedBridge) -> None:
    """Subscribe to the autonomy StateMachine and mirror transitions live."""
    for d in (str(_BRIDGE_DIR), str(_AUTONOMY_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)

    from bridge import Vehicle, load_config as load_bridge_config  # type: ignore
    from state_machine import StateMachine  # type: ignore

    cfg = load_bridge_config()
    print(f"[led-bridge] connecting to {cfg['connection_uri']} ...")
    v = Vehicle(
        connection_uri=cfg["connection_uri"],
        source_system=cfg.get("source_system", 255),
        heartbeat_timeout=cfg.get("heartbeat_timeout", 30.0),
    )
    v.connect()
    fsm = StateMachine(v, poll_hz=10.0)

    def on_state(old, new, reason) -> None:
        print(f"[led-bridge] FSM: {old.name} -> {new.name}"
              + (f"  ({reason})" if reason else ""))
        try:
            led.set_state(new.name)
        except LedBridgeError as e:
            print(f"[led-bridge] write failed: {e}")

    fsm.subscribe(on_state)
    fsm.start()

    # Push the current state once at startup so the LED matches reality
    # without waiting for the first transition.
    led.set_state(fsm.state.name)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[led-bridge] interrupt")
    finally:
        fsm.stop()
        v.disconnect()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--test", action="store_true",
                   help="cycle through all FSM states (no SITL/FSM needed)")
    g.add_argument("--state",
                   help="push one state name (e.g. ASCENDING) and exit")
    g.add_argument("--ping", action="store_true",
                   help="send PING and print PONG (health check)")
    g.add_argument("--watch-fsm", action="store_true",
                   help="connect to SITL/Pixhawk via bridge, subscribe to "
                        "the state machine, mirror live")
    g.add_argument("--off", action="store_true",
                   help="turn LEDs off and exit")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="seconds per state in --test mode")
    ap.add_argument("--brightness", type=int, default=None,
                    help="apply brightness (0-255) before doing anything else")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    cfg = load_config(_THIS_DIR / "config.json")
    port = args.port or cfg.get("port", DEFAULT_PORT)
    baud = args.baud or int(cfg.get("baud", DEFAULT_BAUD))

    try:
        with LedBridge(port=port, baud=baud, verbose=args.verbose) as led:
            if args.brightness is not None:
                led.set_brightness(args.brightness)
            if args.off:
                led.off()
            elif args.ping:
                ok = led.ping()
                print("PONG" if ok else "(no response)")
                return 0 if ok else 1
            elif args.state:
                led.set_state(args.state.upper())
                time.sleep(0.3)  # let the Pico's animation start before close()
            elif args.test:
                run_test_cycle(led, dwell_s=args.dwell)
            elif args.watch_fsm:
                run_watch_fsm(led)
            else:
                print("Pick one of --test / --state / --ping / --watch-fsm / --off")
                return 2
        return 0
    except LedBridgeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
