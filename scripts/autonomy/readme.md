# Autonomy

Decision-making layer for AutonoBird. Sits between perception (`scripts/perception/`) and the flight controller bridge (`scripts/flight-controller/bridge.py`), turning vehicle state + obstacle observations into commands sent down to the autopilot.

This subsystem hosts the full closed-loop autonomy stack:

| Module | Status | Purpose |
|---|---|---|
| `state_machine.py` | live | Passive flight-state observer; nine states; Schmitt-trigger hysteresis on VFR_HUD climb rate |
| `planner.py` | live | Reactive sector-based obstacle avoider; CRUISING/AVOIDING/IDLE/LANDING modes with entry/exit hysteresis; cached perception frame for rate-mismatch tolerance |
| `perception_source.py` | live | Abstract perception input — `SyntheticPerceptionSource` (in-process scripted timeline) and `DepthDetectSource` (tails JSONL from `depth_detect.py --jsonl` or the Gazebo bridge). Detections carry an optional `keypoints` field (17 COCO body keypoints) used by the gesture classifier when a pose-capable model is feeding the JSONL. |
| `gazebo_perception_bridge.py` | live | Subscribes to a Gazebo Harmonic LaserScan topic via `gz topic -e --json-output`, converts each scan into a `PerceptionFrame` record matching `depth_detect.py`'s schema. Used by the T5 Gazebo closed-loop test |
| `orchestrator.py` | live | Single-process autonomy stack bring-up: Vehicle bridge → FSM → optional Planner → optional Gesture pipeline → optional LED bridge. Defaults to monitor mode (no commands). Exposes `command_hold/resume/land/rtl` as intent hooks for voice/gesture/external code |
| `gesture_classifier.py` | live | Body-pose gesture recognition (STOP/LAND/COME/RECEDE) on COCO-17 keypoints; per-keypoint confidence gating + 3-frame temporal smoothing |
| `gesture_action_map.py` | live | Dispatches recognised gestures to orchestrator intent methods with cooldown to prevent re-fires |
| `voice_action_map.py` | pending | Jarvis intent → orchestrator intent mapping. Deprioritised relative to gestures — see "Why gesture is the primary modality" below |

| Test / harness | Purpose |
|---|---|
| `test_state_machine.py` | FSM smoke test (takeoff/RTL cycle, verify 9-state sequence) |
| `test_avoidance.py` | Closed-loop SITL avoidance with in-process `SyntheticPerceptionSource` |
| `test_avoidance_jsonl.py` | Closed-loop SITL avoidance with JSONL replay (`DepthDetectSource`) |
| `test_avoidance_gazebo.py` | **T5 dissertation evidence**: closed-loop avoidance against a physics-grounded cylinder in Gazebo Harmonic; spawns `gazebo_perception_bridge.py`; measures clearance |
| `test_hover_stability.py` | **T4 dissertation evidence**: 60 s GUIDED auto-hold drift measurement |
| `test_mission_replicate.py` | **T6 dissertation evidence**: 10-run box mission replication with per-run extent stats |
| `test_wind_sweep.py` | Wind / disturbance rejection sweep at `SIM_WIND_SPD ∈ {0, 5, 10, 15}` m/s — runs both T4 and T6 at each level |
| `test_gesture_pipeline.py` | Unit + integration test for the gesture stack — synthetic keypoints → classifier → action map → mock orchestrator. No Hailo / camera needed. |
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

## Gesture pipeline

A body-pose classifier converts the operator's posture into discrete drone commands. Four gestures, all safety-critical and unambiguous from 17 COCO body keypoints:

| Gesture | Pose | Intent | Orchestrator method |
|---|---|---|---|
| STOP | T-pose (arms out horizontal) | hold position | `command_hold()` (pauses planner) |
| LAND | both arms straight down | land here | `command_land()` (planner stops, vehicle LAND) |
| COME | both arms straight up | resume / approach | `command_resume()` |
| RECEDE | arms crossed in front of chest (X-pose) | back off | `command_hold()` for v1; reverse-cruise is a future extension |

Pipeline:

```
YOLOv8n-pose on Hailo-8 (depth_detect.py --pose, future work)
        │ COCO-17 keypoints per detection
        ▼
DepthDetectSource (perception_source.py)
        │ PerceptionFrame with Detection.keypoints populated
        ▼
GestureClassifier.update(frame) → Gesture
        │ per-keypoint confidence gating + 3-frame temporal smoothing
        ▼
GestureActionMap.dispatch(gesture)
        │ cooldown to suppress re-fires
        ▼
Orchestrator.command_hold / command_land / command_resume / command_rtl
```

The classifier picks the **closest person** as the primary operator — bystanders standing further from the drone don't trigger commands.

Per-keypoint confidence below `min_keypoint_confidence` (default 0.5) is treated as missing; gestures that depend on a missing keypoint stay NONE rather than firing on noise. The action map's cooldown (default 2 s) prevents the same gesture from dispatching repeatedly within one continuous hold.

### Why gesture is the primary in-flight modality

The original spec had voice as the primary command channel. The acoustic environment of an aerial drone makes that unworkable in flight:

- **Motor noise dominates the on-board mic** at ~75-85 dB from 1 m, swamping operator voice at any non-trivial altitude
- **Operator voice attenuates ~6 dB per doubling of distance** — at 10 m AGL the operator's voice is 20 dB quieter relative to the rotor noise
- **The mic is on the drone**, not the operator — fixing this means downlinking audio from a body-worn mic, which adds a separate radio channel
- **Voice latency** (wake-word + ASR + LLM intent + MAVLink) stacks to 1.5–3 s; gestures are sub-second perception-to-action because they share the existing perception loop

Visual perception isn't degraded by motor noise. The AR0144 stereo + Hailo NPU already point at the operator during cruise and return paths. Gesture → action runs through the same 138 ms loop that does obstacle avoidance, so it inherits the loop's latency budget for free.

**Voice doesn't go away** — it remains useful for:
- Pre-flight commands while the drone is on the bench ("Jarvis, run preflight check")
- Post-flight summaries / mission review
- Configuration / development from the operator station

But voice is demoted from the in-flight command path. FR6 (voice recognition capability) stays a met requirement; FR9 (visual + voice descriptions) stays where it is. The primary in-flight command interface is gesture.

### Testing the gesture pipeline

Without the Pi rig (no Hailo, no AR0144), the dev workstation runs `test_gesture_pipeline.py` end-to-end against synthetic keypoints:

```bash
source venv/bin/activate
python test_gesture_pipeline.py
```

Five sub-tests cover: per-gesture classification, low-confidence rejection, alternation-noise rejection, action map dispatch correctness, cooldown behaviour, and full classifier+action-map composition. All five pass against the canonical-pose synthetic fixtures.

### Running gestures live (Pi rig)

Perception side is code-complete: `scripts/perception/depth_detect.py --pose` loads `v8_pose_n_hailo8.hef`, runs the defensive multi-format pose decoder, and emits a `keypoints` list per detection in the JSONL. **Untested against live Hailo as of the 2026-05-20 dissertation submission** — the first Pi run validates the assumed `yolov8_pose_postprocess` output layout; the decoder may need tensor-shape iteration if the actual HEF output diverges from the assumed layout. See `project_current_status.md` auto-memory for the post-submission live-validation plan.

To run end-to-end on the Pi rig once the pose HEF has been validated:

1. Confirm the pose HEF is at `scripts/perception/models/v8_pose_n_hailo8.hef` (one of `~/Documents/Benchy/resources/hefs/v8_pose_n_hailo8.hef` or `v11_pose_n_hailo8.hef`).
2. Run perception with pose decoding enabled, emitting JSONL:

```bash
cd scripts/perception
./venv/bin/python depth_detect.py --pose --jsonl /tmp/perception.jsonl --no-gui
```

3. Launch the orchestrator with gestures enabled, pointed at the same JSONL:

```bash
cd scripts/autonomy
./venv/bin/python orchestrator.py --enable-gestures \
    --perception jsonl --jsonl-path /tmp/perception.jsonl --tail-from-end
```

The orchestrator's gesture loop pulls the latest frame at `--gesture-rate-hz` (default 10 Hz), classifies, and dispatches.
