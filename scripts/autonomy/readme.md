# Autonomy

Decision-making layer for AutonoBird. Sits between perception (`scripts/perception/`) and the flight controller bridge (`scripts/flight-controller/bridge.py`), turning vehicle state + obstacle observations into commands sent down to the autopilot.

This subsystem will grow to host the full autonomy stack:

| Module | Status | Purpose |
|---|---|---|
| `state_machine.py` | ✅ live | Passive flight-state observer; emits transition events |
| `planner.py` | ⏳ next | Sector-based obstacle avoidance; emits velocity setpoints |
| `perception_source.py` | ⏳ next | Abstract perception input — synthetic for tests, `depth_detect.py` event tap for real use |
| `orchestrator.py` | ⏳ | Wires state machine + planner + voice + gesture + LEDs together; the single program entry-point |
| `voice_action_map.py` | ⏳ | Jarvis intent → flight command mapping |
| `gesture_action_map.py` | ⏳ | YOLO-pose gesture → flight command mapping |

Architecture note: this subsystem **imports the bridge** from `../flight-controller/bridge.py` via `sys.path` injection. The bridge stays in flight-controller because it's the transport layer; autonomy is decision-making on top of it.

## One-time setup

`scripts/flight-controller/setup.sh` must have been run first (this subsystem depends on the bridge).

```bash
cd scripts/autonomy
bash setup.sh
```

Creates `venv/` with `pymavlink>=2.4.40` and `numpy`. Verifies the cross-subsystem bridge import.

## State machine

`state_machine.py` is a passive observer — it consumes the bridge's `Vehicle.state` snapshot and event stream, and infers one of nine high-level flight states:

```
DISCONNECTED → NO_FIX → PREARMED → ARMED_ON_GROUND →
ASCENDING → AIRBORNE → DESCENDING → DISARMED_POSTFLIGHT
                                                ↳ FAULT (terminal)
```

Subscribers (orchestrator, LED driver, voice/gesture mappers) register a callback to receive state-change events. The FSM does **not** issue MAVLink commands — that's the bridge's job. The FSM tracks state; callers drive the vehicle.

### Usage

```python
from bridge import Vehicle, load_config              # via sys.path
from state_machine import StateMachine, FlightState

cfg = load_config()
v = Vehicle(cfg["connection_uri"])
v.connect()

with StateMachine(v) as fsm:
    fsm.subscribe(lambda old, new, reason:
        print(f"{old.name} -> {new.name}  {reason or ''}"))

    # drive the vehicle normally
    v.set_mode("GUIDED")
    v.arm()
    v.takeoff(10)
    # ... FSM emits ARMED_ON_GROUND → ASCENDING → AIRBORNE etc.

    print(f"now in {fsm.state.name}")

v.disconnect()
```

### Smoke test

With SITL running (`scripts/sitl/run_sitl.sh`):

```bash
./venv/bin/python test_state_machine.py
```

Drives a takeoff/RTL cycle through the bridge and validates that the FSM observed the full transition sequence (`NO_FIX` → `PREARMED` → `ARMED_ON_GROUND` → `ASCENDING` → `AIRBORNE` → `DESCENDING` → `DISARMED_POSTFLIGHT`).

### Design notes

- **Passive observer, not a controller.** The FSM does not call into the bridge to change vehicle state. It only watches. Higher-level code (orchestrator) uses both the FSM (for "where are we?") and the bridge (for "go there").
- **Polling at 10 Hz** is the inference rate. Vehicle telemetry only updates at ~4–7 Hz so we're slightly over-sampling, which is fine — the FSM coalesces redundant `_transition()` calls.
- **Climb rate is computed from polled altitude samples**, not from a VFR_HUD field, because the bridge state snapshot doesn't expose climb rate directly. Threshold ±0.2 m/s separates ASCENDING / DESCENDING from steady AIRBORNE.
- **Faults are terminal** until `fsm.reset()` is called. Detected via STATUSTEXT keyword matching: battery failsafe, EKF failsafe, crash disarm, GPS glitch, radio failsafe.
- **Subscriber callbacks run on the FSM thread.** Keep them short or push work to your own queue — slow callbacks block state inference.
