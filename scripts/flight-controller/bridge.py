"""
scripts/flight-controller/bridge.py

Pi-side MAVLink bridge for AutonoBird.

A Vehicle class wrapping pymavlink that abstracts the transport (SITL TCP /
real Pixhawk USB-serial) behind a single connect URI, runs a background
reader that keeps a thread-safe snapshot of vehicle state, and exposes a
small command API (mode set, arm, takeoff, RTL, velocity-target, mission
upload) for higher-level consumers (path planner, voice command mapper,
orchestrator).

The same code base targets SITL during development and the real Pixhawk
during flight; only the connection URI changes.

Design notes:
- A single background thread (`_reader`) consumes every incoming message
  and dispatches it: heartbeats and telemetry update an internal state
  dict; mode changes / arm changes / status text / mission events are
  published to an `events` queue for consumers to read; COMMAND_ACK
  messages go to a dedicated `_acks` queue that `_send_command` waits on.
  This avoids the classic pymavlink footgun where two pieces of code
  call `recv_match()` and race for the same messages.
- Source-system filtering matches the dissertation's §6.3 finding: the
  MAVProxy GCS reports sysid 255 with custom_mode 0 (STABILIZE sentinel),
  while the autopilot reports sysid 1 with the real flight mode. Any
  naive heartbeat consumer that doesn't filter sees mode oscillate
  between the GCS sentinel and the autopilot's real mode. We only honour
  messages from `_target_system` (the autopilot we first heard from).
- Commands are synchronous (block until ACK or timeout). Callers
  should not issue concurrent commands from multiple threads on the same
  Vehicle instance.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from pymavlink import mavutil


# ArduCopter custom_mode -> human name. Subset relevant to this project.
COPTER_MODES: dict[int, str] = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    20: "GUIDED_NOGPS",
    27: "AUTO_RTL",
}
COPTER_MODES_REVERSE: dict[str, int] = {v: k for k, v in COPTER_MODES.items()}


@dataclass
class VehicleState:
    """Snapshot of vehicle telemetry. All fields may be None pre-connect."""

    connected: bool = False
    armed: bool = False
    mode: Optional[str] = None
    mode_id: Optional[int] = None
    system_status: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_rel: Optional[float] = None
    heading_deg: Optional[float] = None
    ground_speed: Optional[float] = None
    gps_fix: Optional[int] = None
    satellites: Optional[int] = None
    battery_v: Optional[float] = None
    battery_pct: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class VehicleEvent:
    """One thing that happened on the link, surfaced to consumers."""

    kind: str  # e.g. "mode", "arm", "status", "item_reached"
    payload: Any
    t: float = field(default_factory=time.time)

    def __repr__(self) -> str:  # pragma: no cover
        return f"VehicleEvent({self.kind}, {self.payload!r})"


class BridgeError(RuntimeError):
    """Bridge-level error (timeout, rejected command, bad state)."""


class Vehicle:
    """Pi-side MAVLink bridge — one instance per connected autopilot."""

    def __init__(
        self,
        connection_uri: str,
        source_system: int = 255,
        heartbeat_timeout: float = 30.0,
    ):
        self._uri = connection_uri
        self._source_system = source_system
        self._heartbeat_timeout = heartbeat_timeout

        # Wire-level
        self._mav: Optional[mavutil.mavfile] = None
        self._target_system: Optional[int] = None
        self._target_component: Optional[int] = None

        # State + concurrency
        self._state_lock = threading.Lock()
        self._state = VehicleState()
        self._events: queue.Queue[VehicleEvent] = queue.Queue()
        self._acks: queue.Queue[Any] = queue.Queue()
        self._stop_flag = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._heartbeat: Optional[threading.Thread] = None
        # Default 1 Hz GCS heartbeat — matches MAVProxy / Mission Planner /
        # QGroundControl behaviour and is what most autopilots expect from a
        # connected ground station.
        self._heartbeat_period = 1.0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Open the link and wait for the first heartbeat."""
        self._mav = mavutil.mavlink_connection(
            self._uri,
            source_system=self._source_system,
            autoreconnect=False,
        )
        hb = self._mav.wait_heartbeat(timeout=self._heartbeat_timeout)
        if hb is None:
            raise BridgeError(
                f"No heartbeat from {self._uri} within {self._heartbeat_timeout}s"
            )
        self._target_system = hb.get_srcSystem()
        self._target_component = hb.get_srcComponent()

        # Boot the reader + heartbeat threads before marking connected
        # so neither the first post-hello messages nor the autopilot's
        # GCS-presence detection get dropped on startup.
        self._stop_flag.clear()
        self._reader = threading.Thread(
            target=self._read_loop, name="mav-reader", daemon=True
        )
        self._reader.start()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="mav-heartbeat", daemon=True
        )
        self._heartbeat.start()

        with self._state_lock:
            self._state.connected = True
            # Seed mode/armed from the first heartbeat.
            self._state.armed = bool(hb.base_mode & 0x80)
            self._state.mode_id = hb.custom_mode
            self._state.mode = COPTER_MODES.get(hb.custom_mode, str(hb.custom_mode))
            self._state.system_status = hb.system_status

    def disconnect(self) -> None:
        """Stop the reader + heartbeat threads and close the link. Idempotent."""
        self._stop_flag.set()
        if self._heartbeat is not None and self._heartbeat.is_alive():
            self._heartbeat.join(timeout=2.0)
        self._heartbeat = None
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=2.0)
        self._reader = None
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None
        with self._state_lock:
            self._state.connected = False

    def __enter__(self) -> "Vehicle":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------------ #
    # State + events                                                     #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> VehicleState:
        """Snapshot of current vehicle state. Thread-safe."""
        with self._state_lock:
            return VehicleState(**self._state.__dict__)

    def events(self, timeout: Optional[float] = None) -> Iterator[VehicleEvent]:
        """Iterator over queued events. Yields until timeout or disconnect."""
        while True:
            try:
                yield self._events.get(timeout=timeout)
            except queue.Empty:
                return

    def drain_events(self) -> list[VehicleEvent]:
        """Return all currently-queued events without blocking."""
        out: list[VehicleEvent] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------------ #
    # Commands                                                           #
    # ------------------------------------------------------------------ #

    def set_mode(self, mode: str, timeout: float = 5.0) -> None:
        """Set flight mode by name (case-insensitive). Blocks until confirmed."""
        mode_id = COPTER_MODES_REVERSE.get(mode.upper())
        if mode_id is None:
            raise ValueError(
                f"Unknown mode {mode!r}; known: {sorted(COPTER_MODES_REVERSE)}"
            )
        self._require_connected()
        # Use the dedicated set_mode helper — works across MAVLink versions
        self._mav.set_mode(mode_id)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.state.mode_id == mode_id:
                return
            time.sleep(0.05)
        raise BridgeError(
            f"Mode change to {mode} not confirmed in {timeout}s (current={self.state.mode})"
        )

    def arm(self, force: bool = False, timeout: float = 5.0) -> None:
        """Arm motors. Blocks until armed or timeout."""
        self._send_command(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
            param2=21196.0 if force else 0.0,
            timeout=timeout,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.state.armed:
                return
            time.sleep(0.05)
        raise BridgeError(f"Arm acknowledged but armed=True not observed in {timeout}s")

    def disarm(self, force: bool = False, timeout: float = 5.0) -> None:
        """Disarm motors. Blocks until disarmed or timeout."""
        self._send_command(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
            param2=21196.0 if force else 0.0,
            timeout=timeout,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.state.armed:
                return
            time.sleep(0.05)
        raise BridgeError(f"Disarm acknowledged but armed=False not observed in {timeout}s")

    def takeoff(self, alt_m: float, timeout: float = 30.0) -> None:
        """Issue NAV_TAKEOFF and wait until the vehicle reaches 95 % of alt_m.

        Pre-conditions: vehicle is armed and in GUIDED mode. Caller is
        responsible for sequencing (set_mode -> arm -> takeoff).
        """
        if not self.state.armed:
            raise BridgeError("Vehicle is not armed")
        if self.state.mode != "GUIDED":
            raise BridgeError(
                f"Mode must be GUIDED for takeoff (currently {self.state.mode})"
            )
        # ACK timeout is short; altitude-reach gets the full timeout budget below.
        self._send_command(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            param7=float(alt_m),
            timeout=5.0,
        )
        deadline = time.time() + timeout
        target = alt_m * 0.95
        while time.time() < deadline:
            alt = self.state.alt_rel
            if alt is not None and alt >= target:
                return
            time.sleep(0.2)
        raise BridgeError(
            f"Did not reach {alt_m}m in {timeout}s (last alt={self.state.alt_rel}m)"
        )

    def rtl(self, timeout: float = 5.0) -> None:
        """Switch to RTL mode. Does not block on landing — listen on events()."""
        self.set_mode("RTL", timeout=timeout)

    def land(self, timeout: float = 5.0) -> None:
        """Switch to LAND mode in place. Does not block on touchdown."""
        self.set_mode("LAND", timeout=timeout)

    def send_velocity_ned(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float = 0.0,
    ) -> None:
        """Send a velocity setpoint in local NED frame (vx north, vy east, vz down).

        Useful for offboard path-planner control. Vehicle must be in GUIDED.
        Does not block — repeat at ~10 Hz for continuous control.
        """
        self._require_connected()
        # SET_POSITION_TARGET_LOCAL_NED type_mask: ignore position bits (0..2),
        # ignore acceleration bits (6..8), use yaw_rate (bit 10), ignore yaw (bit 9).
        type_mask = 0b0000_1000_1100_0111
        self._mav.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self._target_system,
            self._target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0.0, 0.0, 0.0,  # position (ignored)
            float(vx), float(vy), float(vz),
            0.0, 0.0, 0.0,  # acceleration (ignored)
            0.0,            # yaw (ignored)
            float(yaw_rate),
        )

    def upload_mission(self, items: list[dict]) -> None:
        """Upload a waypoint mission via the classic mission-protocol handshake.

        Each item dict must contain: seq, frame, command, current,
        autocontinue, param1..param4, x (lat * 1e7 or local), y, z.

        For now this is a minimal implementation; more elaborate FTP-style
        uploads can be added if needed.
        """
        self._require_connected()
        # Clear existing mission
        self._mav.mav.mission_clear_all_send(
            self._target_system, self._target_component
        )
        # Send count
        self._mav.mav.mission_count_send(
            self._target_system, self._target_component, len(items)
        )

        # The autopilot will request each item by seq; we respond in turn.
        deadline = time.time() + max(10.0, 0.5 * len(items))
        sent = 0
        while sent < len(items) and time.time() < deadline:
            req = self._mav.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                blocking=True,
                timeout=2.0,
            )
            if req is None:
                continue
            i = items[req.seq]
            self._mav.mav.mission_item_int_send(
                self._target_system,
                self._target_component,
                i["seq"],
                i["frame"],
                i["command"],
                i["current"],
                i["autocontinue"],
                float(i.get("param1", 0)),
                float(i.get("param2", 0)),
                float(i.get("param3", 0)),
                float(i.get("param4", 0)),
                int(i["x"]),
                int(i["y"]),
                float(i["z"]),
                0,  # mission_type = MAV_MISSION_TYPE_MISSION
            )
            sent += 1

        # Wait for MISSION_ACK
        ack = self._mav.recv_match(type="MISSION_ACK", blocking=True, timeout=5.0)
        if ack is None:
            raise BridgeError("No MISSION_ACK after upload")
        if ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
            raise BridgeError(f"Mission upload rejected: type={ack.type}")

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _require_connected(self) -> None:
        if not self._state.connected or self._mav is None:
            raise BridgeError("Vehicle is not connected — call connect() first")

    def _heartbeat_loop(self) -> None:
        """Send a GCS heartbeat at `_heartbeat_period` until stopped.

        Autopilots use the presence of an incoming heartbeat from a GCS
        source-system to gate behaviour: TCP-direct SITL keeps its serial
        link alive, real Pixhawk firmware suppresses the "no GCS / radio
        link lost" failsafe, and `FS_GCS_ENABLE`-driven RTL is timed off
        the gap between received GCS heartbeats. 1 Hz is the convention.
        """
        while not self._stop_flag.is_set():
            try:
                self._mav.mav.heartbeat_send(  # type: ignore[union-attr]
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0,
                )
            except Exception:
                # Transport may be closed; reader thread will exit too.
                return
            self._stop_flag.wait(self._heartbeat_period)

    def _read_loop(self) -> None:
        """Background message consumer. Runs until disconnect()."""
        while not self._stop_flag.is_set():
            try:
                msg = self._mav.recv_match(blocking=True, timeout=1.0)
            except Exception:
                # transport closed mid-read
                break
            if msg is None:
                continue
            try:
                # Filter to the autopilot we first met. MAVProxy / other GCS
                # heartbeats are skipped here; see file-level design notes.
                if msg.get_srcSystem() != self._target_system and msg.get_type() != "COMMAND_ACK":
                    # We do still want COMMAND_ACKs from any source (the
                    # FC routes them through its own sysid, but be lenient).
                    continue
                self._dispatch(msg)
            except Exception:
                # Never let a per-message exception kill the reader.
                continue

    def _dispatch(self, msg) -> None:
        t = msg.get_type()
        if t == "HEARTBEAT":
            armed = bool(msg.base_mode & 0x80)
            mode_id = msg.custom_mode
            with self._state_lock:
                prev_armed = self._state.armed
                prev_mode_id = self._state.mode_id
                self._state.armed = armed
                self._state.mode_id = mode_id
                self._state.mode = COPTER_MODES.get(mode_id, str(mode_id))
                self._state.system_status = msg.system_status
            if armed != prev_armed:
                self._events.put(VehicleEvent("arm", armed))
            if mode_id != prev_mode_id:
                self._events.put(VehicleEvent("mode", self._state.mode))
        elif t == "GLOBAL_POSITION_INT":
            with self._state_lock:
                self._state.lat = msg.lat / 1e7
                self._state.lon = msg.lon / 1e7
                self._state.alt_rel = msg.relative_alt / 1000.0
                self._state.heading_deg = msg.hdg / 100.0 if msg.hdg != 65535 else None
        elif t == "VFR_HUD":
            with self._state_lock:
                self._state.ground_speed = msg.groundspeed
        elif t == "GPS_RAW_INT":
            with self._state_lock:
                self._state.gps_fix = msg.fix_type
                self._state.satellites = msg.satellites_visible
        elif t == "SYS_STATUS":
            with self._state_lock:
                self._state.battery_v = msg.voltage_battery / 1000.0
                self._state.battery_pct = msg.battery_remaining
        elif t == "STATUSTEXT":
            text = msg.text.rstrip("\x00")
            self._events.put(VehicleEvent("status", text))
        elif t == "MISSION_ITEM_REACHED":
            self._events.put(VehicleEvent("item_reached", msg.seq))
        elif t == "COMMAND_ACK":
            self._acks.put(msg)

    def _send_command(
        self,
        command_id: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
        timeout: float = 5.0,
    ) -> None:
        """Send COMMAND_LONG, wait for matching COMMAND_ACK with ACCEPTED."""
        self._require_connected()
        # Drain any stale ACKs from prior commands.
        while not self._acks.empty():
            try:
                self._acks.get_nowait()
            except queue.Empty:
                break

        self._mav.mav.command_long_send(
            self._target_system,
            self._target_component,
            command_id,
            0,  # confirmation
            param1, param2, param3, param4, param5, param6, param7,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ack = self._acks.get(timeout=0.1)
            except queue.Empty:
                continue
            if ack.command != command_id:
                # Not the ACK we're waiting for — put it back is risky in
                # a queue; for our single-threaded command model, drop it.
                continue
            if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return
            raise BridgeError(
                f"Command {command_id} rejected: result={ack.result}"
            )
        raise BridgeError(f"No COMMAND_ACK for {command_id} within {timeout}s")


def load_config(path: Optional[Path] = None) -> dict:
    """Load config.json (or config.example.json if no override exists)."""
    here = Path(__file__).parent
    candidates = [path] if path else [here / "config.json", here / "config.example.json"]
    for c in candidates:
        if c is None:
            continue
        if c.is_file():
            with c.open() as f:
                return json.load(f)
    raise FileNotFoundError(
        "No config.json or config.example.json found in scripts/flight-controller/"
    )
