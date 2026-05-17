# Autonomy

Decision-making layer for AutonoBird. Sits between perception (`scripts/perception/`) and the flight controller bridge (`scripts/flight-controller/bridge.py`), turning vehicle state + obstacle observations into commands sent down to the autopilot.

This subsystem hosts the full closed-loop autonomy stack:

| Module | Status | Purpose |
|---|---|---|
| `state_machine.py` | ✅ live | Passive flight-state observer; nine states; Schmitt-trigger hysteresis on VFR_HUD climb rate |
| `planner.py` | ✅ live | Reactive sector-based obstacle avoider; CRUISING/AVOIDING/IDLE/LANDING modes with entry/exit hysteresis; cached perception frame for rate-mismatch tolerance |
| `perception_source.py` | ✅ live | Abstract perception input — `SyntheticPerceptionSource` (in-process scripted timeline) and `DepthDetectSource` (tails JSONL from `depth_detect.py --jsonl` or the Gazebo bridge) |
| `gazebo_perception_bridge.py` | ✅ live | Subscribes to a Gazebo Harmonic LaserScan topic via `gz topic -e --json-output`, converts each scan into a `PerceptionFrame` record matching `depth_detect.py`'s schema. Used by the T5 Gazebo closed-loop test |
| `orchestrator.py` | ⏳ | Wires state machine + planner + voice + gesture + LEDs together; the single program entry-point |
| `voice_action_map.py` | ⏳ | Jarvis intent → flight command mapping |
| `gesture_action_map.py` | ⏳ | YOLO-pose gesture → flight command mapping |

| Test / harness | Purpose |
|---|---|
| `test_state_machine.py` | FSM smoke test (takeoff/RTL cycle, verify 9-state sequence) |
| `test_avoidance.py` | Closed-loop SITL avoidance with in-process `SyntheticPerceptionSource` |
| `test_avoidance_jsonl.py` | Closed-loop SITL avoidance with JSONL replay (`DepthDetectSource`) |
| `test_avoidance_gazebo.py` | **T5 dissertation evidence**: closed-loop avoidance against a physics-grounded cylinder in Gazebo Harmonic; spawns `gazebo_perception_bridge.py`; measures clearance |
| `test_hover_stability.py` | **T4 dissertation evidence**: 60 s GUIDED auto-hold drift measurement |
| `test_mission_replicate.py` | **T6 dissertation evidence**: 10-run box mission replication with per-run extent stats |
| `test_wind_sweep.py` | Wind / disturbance rejection sweep at `SIM_WIND_SPD ∈ {0, 5, 10, 15}` m/s — runs both T4 and T6 at each level |
| `make_synthetic_jsonl.py` | Generate a synthetic JSONL session for `test_avoidance_jsonl.py` |

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
- **Climb rate uses VFR_HUD's autopilot-filtered `climb` field** (m/s, +up) exposed via `Vehicle.state.climb_rate`, with Schmitt-trigger hysteresis (enter at ±0.5 m/s, exit at ±0.15 m/s) to separate ASCENDING / DESCENDING from steady AIRBORNE. Earlier polling-and-differentiating logic produced 140+ spurious transitions per flight; the filtered-source + hysteresis approach lands at ~9 clean transitions per takeoff/RTL cycle.
- **Faults are terminal** until `fsm.reset()` is called. Detected via STATUSTEXT keyword matching: battery failsafe, EKF failsafe, crash disarm, GPS glitch, radio failsafe.
- **Subscriber callbacks run on the FSM thread.** Keep them short or push work to your own queue — slow callbacks block state inference.

## Planner

`planner.py` is the reactive obstacle avoider. Background thread reads the perception source at 10 Hz and emits velocity setpoints via `Vehicle.send_velocity_ned`. Four modes: CRUISING (forward velocity at `cruise_speed_ms`), AVOIDING (lateral sidestep), IDLE, LANDING. Mode transitions use hysteresis on entry (`obstacle_threshold_m`) vs exit (`clear_threshold_m`) so the planner doesn't flap at the obstacle boundary, and the sidestep direction is latched on entry to AVOIDING to prevent left-right oscillation. The most-recent perception frame is cached for up to 1 s so a 10 Hz planner doesn't misclassify "no fresh frame" as "no obstacle" when perception is slower (e.g., 6.8 Hz from `depth_detect.py`).

## Perception sources

`perception_source.py` defines the `PerceptionFrame` / `Detection` data contract and two concrete sources:

- `SyntheticPerceptionSource` — in-process scripted obstacle timeline for unit tests
- `DepthDetectSource` — tails a JSONL file (either `depth_detect.py --jsonl PATH` or the Gazebo bridge); supports replay mode (read from start) with real-time pacing keyed off the `t` field, or live-tail mode (seek to EOF; only react to new records)

**JSONL data contract** (one record per line):

```json
{"t": 1716941234.567,
 "detections": [{"class_name": "person", "confidence": 0.87,
                 "bbox_xyxy": [120, 150, 280, 380],
                 "depth_m": 1.23,
                 "bbox_centroid_norm": [-0.05, 0.10]}],
 "camera_hfov_deg": 67.0, "camera_vfov_deg": 41.0}
```

`bbox_centroid_norm` is a **2-element list** `[cx, cy]` in `[-1, +1]` from frame centre (cx=0 means centred, cx=+1 means at the right edge). Emitting a dict `{"x": ..., "y": ...}` looks like it ought to work but breaks silently — `perception_source.py:_parse_line` iterates the field and treats dict-iteration's key strings as floats, triggering an exception that the bare `except` drops. The Gazebo bridge had this bug originally; fixed in `gazebo_perception_bridge.py`.

## Gazebo perception bridge

`gazebo_perception_bridge.py` is the Gazebo-side analogue of `depth_detect.py --jsonl`. It subscribes to a Gazebo Harmonic LaserScan topic via `gz topic -e --json-output`, finds the closest in-range return per scan, and emits one `PerceptionFrame` record per scan to the JSONL file the autonomy `DepthDetectSource` tails. No code change is needed on the autonomy side — the perception layer is genuinely transport-agnostic.

The custom Gazebo model and world used by the T5 test live under `../sitl/gazebo/`:

- `models/iris_with_lidar/` — wraps the standard ArduPilot iris with a forward-facing GPU lidar (61 samples, 67° HFOV, 5 Hz, 0.2–15 m range) fixed to the iris `base_link`
- `worlds/iris_obstacle.sdf` — runway world with a 1 m diameter × 12 m tall collision-bodied red cylinder placed 12 m N of home, in the iris's cruise path

The T5 closed-loop test (`test_avoidance_gazebo.py`) is the strongest SITL-track evidence the autonomy stack produces. Result: 1.92 m min clearance from the cylinder surface (6.4× over the 0.3 m dissertation T5 bound), full CRUISING → AVOIDING → CRUISING cycle, bird passes obstacle to 17 m N. See dissertation § 6.2 *T5 — Closed-loop avoidance against a Gazebo-physics-grounded obstacle*.
