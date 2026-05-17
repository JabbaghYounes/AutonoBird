"""
scripts/autonomy/orchestrator.py — single-process autonomy stack bring-up

Brings up the full autonomy stack as one runnable: Vehicle bridge → State
machine → (optional) Planner with a chosen Perception source → (optional)
LED bridge to the Pico 2 W. Prints a periodic compact status line plus
event-driven transition logs so the operator has a glance-readable picture
of what the system is doing.

The orchestrator deliberately defaults to **monitor mode** — it only
observes the autopilot. Issuing commands (takeoff, planner velocity
setpoints, etc.) requires explicit opt-in via `--enable-planner` plus a
perception source. This matches the project's safety invariant: "Don't
design anything that depends on autonomous commands winning over RC
input." The orchestrator never auto-arms or auto-takeoffs in monitor
mode.

Usage:

    # Monitor only — connect, watch FSM transitions, print status.
    python orchestrator.py

    # Monitor + LED status mirror (if a Pico is on /dev/ttyACM1).
    python orchestrator.py --led-port /dev/ttyACM1

    # Run the reactive planner against a live JSONL (from depth_detect.py
    # or gazebo_perception_bridge.py writing to /tmp/perception.jsonl).
    python orchestrator.py --enable-planner --perception jsonl \\
        --jsonl-path /tmp/perception.jsonl --tail-from-end

    # Run the planner against a synthetic obstacle timeline.
    python orchestrator.py --enable-planner --perception synthetic

The orchestrator is the single integration point that later subsystems
(voice → MAVLink, gesture → MAVLink) will hook into. Each subscriber
just attaches to `Orchestrator.fsm` (state changes) or calls
`Orchestrator.vehicle` methods (commands).
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Cross-subsystem bridge import.
_AUTONOMY_DIR = Path(__file__).resolve().parent
_BRIDGE_DIR = (_AUTONOMY_DIR / ".." / "flight-controller").resolve()
_PICO_DIR = (_AUTONOMY_DIR / ".." / "pico-led").resolve()
for _d in (str(_BRIDGE_DIR), str(_AUTONOMY_DIR), str(_PICO_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from bridge import BridgeError, Vehicle, load_config as load_bridge_config  # noqa: E402

from perception_source import (  # noqa: E402
    DepthDetectSource,
    PerceptionSource,
    SyntheticEvent,
    SyntheticPerceptionSource,
    obstacle_ahead,
)
from planner import Planner, PlannerMode  # noqa: E402
from state_machine import FlightState, StateMachine  # noqa: E402

# Gesture pipeline is optional — orchestrator can run without it. We
# only probe importability at module load; concrete instances are
# constructed lazily inside attach_gestures().
try:
    import gesture_classifier  # noqa: F401
    import gesture_action_map  # noqa: F401
    _GESTURES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GESTURES_AVAILABLE = False


# ---------------------------------------------------------------------- #
# Defaults                                                               #
# ---------------------------------------------------------------------- #

DEFAULT_PLANNER_RATE_HZ = 10.0
DEFAULT_CRUISE_SPEED_MS = 1.5
DEFAULT_AVOIDANCE_SPEED_MS = 1.5
DEFAULT_OBSTACLE_THRESHOLD_M = 1.5
DEFAULT_CLEAR_THRESHOLD_M = 2.0
DEFAULT_STATUS_PERIOD_S = 1.0


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def synthetic_demo_events() -> list[SyntheticEvent]:
    """A short scripted timeline for monitor-with-planner demonstration.

    Mirrors what test_avoidance.py uses, but a bit gentler since the
    orchestrator's primary purpose is operator-facing display rather than
    a pass/fail test.
    """
    return [
        SyntheticEvent(0.0, []),
        SyntheticEvent(15.0, [obstacle_ahead(0.8)]),
        SyntheticEvent(25.0, []),
    ]


# ---------------------------------------------------------------------- #
# LED bridge wrapper                                                     #
# ---------------------------------------------------------------------- #


class _NullLed:
    """Stand-in used when no Pico is configured / available."""

    def set_state(self, _name: str) -> None: ...
    def close(self) -> None: ...


def _try_open_led(port: Optional[str]) -> object:
    """Open the Pico LED bridge if requested. Graceful degrade on failure.

    Importing the led_bridge module on a system without pyserial would
    raise — we wrap import + connect together so a missing Pi-side dep
    is just "LED disabled" rather than an orchestrator crash.
    """
    if not port:
        return _NullLed()
    try:
        from led_bridge import LedBridge  # type: ignore
    except ImportError as e:
        log(f"LED bridge unavailable (pyserial not installed?): {e}")
        return _NullLed()
    try:
        led = LedBridge(port=port)
        led.connect()
        log(f"LED bridge connected on {port}")
        return led
    except Exception as e:
        log(f"LED bridge open failed on {port}: {e} — continuing without LEDs")
        return _NullLed()


# ---------------------------------------------------------------------- #
# Orchestrator                                                           #
# ---------------------------------------------------------------------- #


class Orchestrator:
    """Single-process integration of the autonomy stack."""

    def __init__(
        self,
        connection_uri: str,
        led_port: Optional[str] = None,
        status_period_s: float = DEFAULT_STATUS_PERIOD_S,
    ):
        self.connection_uri = connection_uri
        self.led_port = led_port
        self.status_period_s = status_period_s

        self.vehicle: Optional[Vehicle] = None
        self.fsm: Optional[StateMachine] = None
        self.planner: Optional[Planner] = None
        self.perception: Optional[PerceptionSource] = None
        self.led: object = _NullLed()

        # Gesture subsystem (opt-in via attach_gestures()).
        self.gesture_classifier: object = None
        self.gesture_action_map: object = None
        self._gesture_thread: Optional[threading.Thread] = None

        self._stop = threading.Event()
        self._status_thread: Optional[threading.Thread] = None

    # ---- start / stop ---- #

    def start(self) -> None:
        """Connect to the autopilot, start FSM + LED, but do NOT command.

        Planner attachment is a separate step (`attach_planner`) so the
        caller can stay in monitor mode by default.
        """
        cfg = load_bridge_config()
        # Allow the caller to override only the URI; everything else
        # comes from the bridge's config.json.
        log(f"connecting bridge to {self.connection_uri} ...")
        self.vehicle = Vehicle(
            connection_uri=self.connection_uri,
            source_system=cfg.get("source_system", 255),
            heartbeat_timeout=cfg.get("heartbeat_timeout", 30.0),
        )
        self.vehicle.connect()

        self.led = _try_open_led(self.led_port)

        self.fsm = StateMachine(self.vehicle, poll_hz=10.0)
        self.fsm.subscribe(self._on_fsm)
        self.fsm.start()

        # Push the current FSM state to the LED once at startup so the
        # indicator matches reality without waiting for the first
        # transition.
        try:
            self.led.set_state(self.fsm.state.name)  # type: ignore[attr-defined]
        except Exception:
            pass

        self._status_thread = threading.Thread(
            target=self._status_loop, name="orchestrator-status", daemon=True
        )
        self._status_thread.start()

    def attach_planner(
        self,
        perception: PerceptionSource,
        cruise_speed_ms: float = DEFAULT_CRUISE_SPEED_MS,
        avoidance_speed_ms: float = DEFAULT_AVOIDANCE_SPEED_MS,
        obstacle_threshold_m: float = DEFAULT_OBSTACLE_THRESHOLD_M,
        clear_threshold_m: float = DEFAULT_CLEAR_THRESHOLD_M,
        rate_hz: float = DEFAULT_PLANNER_RATE_HZ,
    ) -> None:
        """Attach the reactive planner. **Vehicle must already be airborne
        and in GUIDED** — the orchestrator does not auto-takeoff. Pilot or
        a separate mission script is responsible for getting the vehicle
        in the right state before opting into autonomous commands.
        """
        if self.vehicle is None:
            raise RuntimeError("call start() before attach_planner()")
        self.perception = perception
        self.planner = Planner(
            vehicle=self.vehicle,
            perception=perception,
            cruise_speed_ms=cruise_speed_ms,
            avoidance_speed_ms=avoidance_speed_ms,
            obstacle_threshold_m=obstacle_threshold_m,
            clear_threshold_m=clear_threshold_m,
            rate_hz=rate_hz,
        )
        self.planner.subscribe(self._on_planner)
        log(
            f"attaching planner (cruise={cruise_speed_ms} m/s, "
            f"avoidance={avoidance_speed_ms} m/s, "
            f"obstacle<{obstacle_threshold_m}m -> AVOIDING, "
            f"obstacle>{clear_threshold_m}m -> CRUISING)"
        )

    def start_planner(self) -> None:
        """Actually engage the planner. Separate from attach so the caller
        can wire things up at startup but only kick the planner once the
        vehicle is positively in the right state."""
        if self.perception is None or self.planner is None:
            raise RuntimeError("call attach_planner() first")
        log("starting planner + perception source")
        self.perception.start()
        self.planner.start()

    # ---- intent commands (called by gesture / voice / external code) ---- #

    def command_hold(self) -> None:
        """Pause autonomy and let the autopilot hold position.

        The reactive planner pauses (no more velocity setpoints). In
        GUIDED mode the autopilot then holds the last commanded
        position. The vehicle stays armed and airborne; nothing else
        changes. Safe to call from any state — no-op if there's no
        planner attached.
        """
        log("INTENT: HOLD")
        if self.planner is not None:
            try:
                self.planner.pause()
            except Exception as e:
                log(f"  planner.pause() failed: {e}")

    def command_resume(self) -> None:
        """Resume autonomy after a hold.

        Restarts the planner from where it paused. No-op if no planner
        is attached or it was never paused.
        """
        log("INTENT: RESUME")
        if self.planner is not None:
            try:
                self.planner.resume()
            except Exception as e:
                log(f"  planner.resume() failed: {e}")

    def command_land(self) -> None:
        """Initiate an immediate descent and landing.

        Stops the reactive planner outright (not pause — the bird is
        landing, no point keeping the planner loop alive) and switches
        the autopilot to LAND mode at the current XY position.
        """
        log("INTENT: LAND")
        if self.planner is not None:
            try:
                self.planner.stop()
            except Exception as e:
                log(f"  planner.stop() failed: {e}")
        if self.vehicle is not None:
            try:
                self.vehicle.land()
            except Exception as e:
                log(f"  vehicle.land() failed: {e}")

    def command_rtl(self) -> None:
        """Return-to-launch — autopilot navigates back to the home position
        and lands. Stops the reactive planner before handing off."""
        log("INTENT: RTL")
        if self.planner is not None:
            try:
                self.planner.stop()
            except Exception as e:
                log(f"  planner.stop() failed: {e}")
        if self.vehicle is not None:
            try:
                self.vehicle.rtl()
            except Exception as e:
                log(f"  vehicle.rtl() failed: {e}")

    # ---- gesture subsystem ---- #

    def attach_gestures(
        self,
        perception: Optional[PerceptionSource] = None,
        rate_hz: float = 10.0,
    ) -> None:
        """Attach the body-pose gesture classifier + action map.

        Uses the orchestrator's existing perception source by default —
        the gesture loop and the planner share the same perception
        stream (both want the same pose / depth events). Pass `perception`
        explicitly to use a separate source for testing.
        """
        if not _GESTURES_AVAILABLE:
            log("WARN: gesture module not importable — gestures disabled")
            return
        src = perception if perception is not None else self.perception
        if src is None:
            raise RuntimeError(
                "no perception source available — attach_planner() first "
                "or pass perception= to attach_gestures()"
            )
        # Late import keeps tests on systems without numpy/torch lean.
        from gesture_classifier import GestureClassifier
        from gesture_action_map import GestureActionMap
        self.gesture_classifier = GestureClassifier()
        self.gesture_action_map = GestureActionMap(
            self,
            on_dispatch=lambda g: log(f"GESTURE → {g.name} dispatched"),
        )
        self._gesture_perception = src
        self._gesture_period = 1.0 / float(rate_hz)
        log(f"gesture pipeline attached (rate {rate_hz} Hz)")

    def start_gestures(self) -> None:
        """Launch the gesture-recognition loop in a background thread."""
        if self.gesture_classifier is None or self.gesture_action_map is None:
            raise RuntimeError("call attach_gestures() first")
        self._gesture_thread = threading.Thread(
            target=self._gesture_loop,
            name="orchestrator-gesture",
            daemon=True,
        )
        self._gesture_thread.start()
        log("gesture loop started")

    def _gesture_loop(self) -> None:
        """Per-frame: pull latest perception, classify, dispatch."""
        last_logged: Optional[str] = None
        while not self._stop.is_set():
            frame = self._gesture_perception.latest()
            if frame is not None:
                # classifier.update is non-mutating from the caller's POV;
                # it owns its own state.
                g = self.gesture_classifier.update(frame)  # type: ignore[union-attr]
                if g.name != last_logged:
                    # Log raw classifier output once per change for debug
                    # (debounced separately from action-map cooldown).
                    if g.name != "NONE":
                        log(f"GESTURE.raw={g.name}")
                    last_logged = g.name
                self.gesture_action_map.dispatch(g)  # type: ignore[union-attr]
            self._stop.wait(self._gesture_period)

    def stop(self) -> None:
        """Tear down in reverse order. Idempotent — safe to call from a
        signal handler."""
        self._stop.set()
        if self._gesture_thread is not None and self._gesture_thread.is_alive():
            try:
                self._gesture_thread.join(timeout=2.0)
            except Exception:
                pass
        if self.planner is not None:
            try:
                self.planner.stop()
            except Exception:
                pass
        if self.perception is not None:
            try:
                self.perception.stop()
            except Exception:
                pass
        if self.fsm is not None:
            try:
                self.fsm.stop()
            except Exception:
                pass
        try:
            self.led.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        if self.vehicle is not None:
            try:
                self.vehicle.disconnect()
            except Exception:
                pass
        log("orchestrator stopped")

    def run_forever(self) -> None:
        """Block until Ctrl-C / SIGTERM. The status thread keeps printing
        while we sleep here."""
        try:
            while not self._stop.is_set():
                time.sleep(0.25)
        except KeyboardInterrupt:
            log("interrupted by user")

    # ---- subscriber callbacks ---- #

    def _on_fsm(self, old: FlightState, new: FlightState, reason) -> None:
        suffix = f"  ({reason})" if reason else ""
        log(f"FSM: {old.name} -> {new.name}{suffix}")
        try:
            self.led.set_state(new.name)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _on_planner(self, old: PlannerMode, new: PlannerMode) -> None:
        log(f"Planner: {old.name} -> {new.name}")

    # ---- status display ---- #

    def _status_loop(self) -> None:
        while not self._stop.wait(self.status_period_s):
            self._print_status()

    def _print_status(self) -> None:
        if self.vehicle is None or self.fsm is None:
            return
        st = self.vehicle.state
        plan_mode = self.planner.mode.name if self.planner is not None else "-"
        # Compact one-liner. Lat/lon are omitted from the periodic STATUS
        # to keep it scannable; they appear in the FSM transition log when
        # relevant.
        alt = f"{st.alt_rel:.2f}m" if st.alt_rel is not None else "—"
        gs = f"{st.ground_speed:.1f}" if st.ground_speed is not None else "—"
        sats = st.satellites if st.satellites is not None else "—"
        batt = f"{st.battery_v:.1f}V" if st.battery_v is not None else "—"
        log(
            f"STATUS  fsm={self.fsm.state.name:<19s} "
            f"mode={(st.mode or '-'):<10s} "
            f"armed={'Y' if st.armed else 'N'}  "
            f"alt={alt:<8s} "
            f"gs={gs}m/s  "
            f"sats={sats}  batt={batt}  "
            f"planner={plan_mode}"
        )


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def build_perception_source(
    kind: str,
    jsonl_path: Optional[str],
    tail_from_end: bool,
    rate_hz: float,
) -> PerceptionSource:
    kind = kind.lower()
    if kind == "synthetic":
        return SyntheticPerceptionSource(synthetic_demo_events(), rate_hz=rate_hz)
    if kind == "jsonl":
        if not jsonl_path:
            raise ValueError("--jsonl-path is required with --perception jsonl")
        return DepthDetectSource(
            jsonl_path=str(jsonl_path),
            tail_from_end=tail_from_end,
        )
    raise ValueError(f"unknown perception source: {kind}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--connection-uri",
        default=None,
        help="override the bridge config's connection_uri (e.g. udpin:127.0.0.1:14550)",
    )
    ap.add_argument(
        "--led-port",
        default=None,
        help="USB-serial device for the Pico LED status indicator. If omitted, LEDs are skipped.",
    )
    ap.add_argument(
        "--status-period-s",
        type=float,
        default=DEFAULT_STATUS_PERIOD_S,
        help="how often to print the compact STATUS line",
    )

    # Planner (opt-in)
    ap.add_argument(
        "--enable-planner",
        action="store_true",
        help="run the reactive planner — only safe if the vehicle is in GUIDED and airborne",
    )
    # Gestures (opt-in)
    ap.add_argument(
        "--enable-gestures",
        action="store_true",
        help="run the body-pose gesture classifier; gestures dispatch to "
             "orchestrator.command_* methods (HOLD / LAND / RESUME / RECEDE)",
    )
    ap.add_argument(
        "--gesture-rate-hz",
        type=float,
        default=10.0,
        help="how often the gesture loop pulls a perception frame",
    )
    ap.add_argument(
        "--perception",
        choices=["synthetic", "jsonl"],
        default="synthetic",
        help="perception source for the planner (when --enable-planner is set)",
    )
    ap.add_argument(
        "--jsonl-path",
        default=None,
        help="JSONL file for --perception jsonl (depth_detect.py --jsonl output, or the Gazebo bridge)",
    )
    ap.add_argument(
        "--tail-from-end",
        action="store_true",
        help="seek to EOF on the JSONL — useful when a producer is writing concurrently",
    )
    ap.add_argument(
        "--cruise-speed-ms",
        type=float,
        default=DEFAULT_CRUISE_SPEED_MS,
    )
    ap.add_argument(
        "--avoidance-speed-ms",
        type=float,
        default=DEFAULT_AVOIDANCE_SPEED_MS,
    )
    ap.add_argument(
        "--obstacle-threshold-m",
        type=float,
        default=DEFAULT_OBSTACLE_THRESHOLD_M,
    )
    ap.add_argument(
        "--clear-threshold-m",
        type=float,
        default=DEFAULT_CLEAR_THRESHOLD_M,
    )
    ap.add_argument(
        "--planner-rate-hz",
        type=float,
        default=DEFAULT_PLANNER_RATE_HZ,
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve connection URI: CLI override > bridge config > error.
    if args.connection_uri:
        connection_uri = args.connection_uri
    else:
        try:
            cfg = load_bridge_config()
            connection_uri = cfg["connection_uri"]
        except Exception as e:
            log(f"FAIL: no --connection-uri and no bridge config: {e}")
            return 1

    orch = Orchestrator(
        connection_uri=connection_uri,
        led_port=args.led_port,
        status_period_s=args.status_period_s,
    )

    # SIGTERM/SIGINT clean shutdown — the daemon status thread won't
    # block exit, but planner / perception / vehicle resources will if
    # not stopped explicitly.
    def _signal_handler(signum, frame):
        log(f"signal {signum} — shutting down")
        orch.stop()
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        orch.start()
    except BridgeError as e:
        log(f"FAIL: bridge connect: {e}")
        return 1

    try:
        # Build a single perception source if any consumer (planner or
        # gestures) needs one. Both pipelines share it.
        perception = None
        if args.enable_planner or args.enable_gestures:
            perception = build_perception_source(
                kind=args.perception,
                jsonl_path=args.jsonl_path,
                tail_from_end=args.tail_from_end,
                rate_hz=args.planner_rate_hz,
            )

        if args.enable_planner:
            orch.attach_planner(
                perception=perception,
                cruise_speed_ms=args.cruise_speed_ms,
                avoidance_speed_ms=args.avoidance_speed_ms,
                obstacle_threshold_m=args.obstacle_threshold_m,
                clear_threshold_m=args.clear_threshold_m,
                rate_hz=args.planner_rate_hz,
            )
            orch.start_planner()
            log("PLANNER ENGAGED — autonomous velocity commands will be issued. "
                "Ensure the vehicle is in GUIDED and airborne, and pilot is "
                "ready to override via RC.")

        if args.enable_gestures:
            if not _GESTURES_AVAILABLE:
                log("FAIL: --enable-gestures requested but gesture_classifier "
                    "/ gesture_action_map could not be imported.")
                return 1
            if perception is None:
                log("FAIL: --enable-gestures requires a perception source. "
                    "Pass --perception jsonl --jsonl-path PATH (or --perception synthetic).")
                return 1
            if not args.enable_planner:
                # Gestures alone still need the source running so latest()
                # returns frames.
                perception.start()
            orch.attach_gestures(
                perception=perception,
                rate_hz=args.gesture_rate_hz,
            )
            orch.start_gestures()
            log("GESTURES ENGAGED — STOP / LAND / COME / RECEDE will dispatch "
                "to orchestrator.command_* methods.")

        if not (args.enable_planner or args.enable_gestures):
            log("monitor mode (no commands will be sent). "
                "Use --enable-planner / --enable-gestures to attach autonomy.")

        orch.run_forever()
        return 0
    finally:
        orch.stop()


if __name__ == "__main__":
    sys.exit(main())
