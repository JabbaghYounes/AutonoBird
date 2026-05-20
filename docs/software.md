# Software

AutonoBird's software is organised as independent subsystems under `scripts/<name>/`, each with its own setup script, virtualenv, config template, and local docs.

## Subsystems

### Flight controller — configuration + MAVLink bridge

Two things live here: the documentation for the Pixhawk 6C Mini hardware build (port assignments, ArduPilot parameter set, pre-flight checklist) and the Pi-side MAVLink bridge that connects perception / autonomy code to the autopilot.

The bridge (`bridge.py`) wraps pymavlink in a `Vehicle` class with a transport-agnostic API — the same code drives SITL via UDP from MAVProxy's forwarder (`udpin:127.0.0.1:14550`) and the real Pixhawk over USB-serial (`serial:///dev/ttyACM0`). Smoke-tested end-to-end against SITL: connect → GUIDED → arm → takeoff → RTL → auto-land → disarm.

- Path: `scripts/flight-controller/`
- Entry points: `bridge.py` (library), `test_bridge.py` (smoke test)
- Docs: [`readme.md`](../scripts/flight-controller/readme.md)

### AR0144 stereo calibration

USB stereo camera calibration. Two calibration scripts: `guided_calibration.py` (recommended, 40-pose face-scan style across 4 distance zones, with manifest-based smart resume) and `basic_calibration.py` (manual SPACE-to-capture). Uses OpenCV SGBM stereo matching. Also includes `visualize_poses.py` — generates a per-pose 3D-geometry reference PDF showing board-on-wall + camera-position for each of the 40 poses.

Current calibration: session `cal-4` produced 2026-05-15 with 28 stereo pairs, RMS reprojection error 1.10 px, baseline 51.92 mm (matches AR0144 52 mm spec). Calibration file at `stereo_calibration_data/stereo_calibration.npz` is consumed by `scripts/perception/depth_detect.py`.

- Path: `scripts/ar0144/`
- Entry points: `guided_calibration.py {preview|capture|calibrate|depth|all}`, `visualize_poses.py`
- Docs: [`steps.md`](../scripts/ar0144/steps.md), [`pose-list.md`](../scripts/ar0144/pose-list.md)

### Perception (YOLO + depth fusion)

Integrated handheld perception pipeline running on the Pi 5 + AI HAT+ (Hailo-8) rig:

- `yolo_detect.py` — YOLOv8n on Hailo-8 against the AR0144 left frame only. No calibration needed; validates the detection half of the pipeline in isolation.
- `depth_detect.py` — full pipeline: capture → split → rectify → SGBM half-res → YOLO detection → depth-fused bbox annotations. Requires `cal-4` calibration `.npz` to exist.

Current measured performance: 6.8 fps end-to-end / 138 ms loop, with 18.5 ms NPU + 76 ms SGBM. Under the dissertation's NFR1 target of <250 ms.

Dedicated venv at `scripts/perception/venv` with version-pinned dependencies (`hailort 4.23` + `opencv-python<4.11` + `numpy<2` — interlocked, see subsystem readme). `hailo_platform` is symlinked from Benchy's venv. `setup.sh` creates and verifies the venv.

- Path: `scripts/perception/`
- Entry points: `yolo_detect.py`, `depth_detect.py`
- Docs: [`readme.md`](../scripts/perception/readme.md)

### ArduPilot SITL (QUAV250 overlay) + Gazebo Harmonic integration

Software-in-the-loop simulation of the ArduCopter flight controller, running on the dev workstation (x86_64, not the Pi). Exposes MAVLink over TCP exactly like the real Pixhawk over USB-serial, so MAVProxy, QGroundControl, Mission Planner, or a Pi-side companion-computer bridge can connect identically. Authorised path for validating autonomy without a physical hover.

ArduPilot itself lives out-of-tree at `~/Documents/ardupilot` — `scripts/sitl/` contains the QUAV250 overlay (`quav250.parm`: frame, FLTMODE assignments, throttle failsafe, battery thrust scaling — hardware-specific params intentionally omitted), a launch wrapper (`run_sitl.sh`), the mission catalogue (`missions/`), and the Gazebo Harmonic integration files (`gazebo/worlds/iris_obstacle.sdf`, `gazebo/models/iris_with_lidar/`) used by the T5 closed-loop test.

First scripted mission flown 2026-05-15: 50 m × 50 m box at 10 m AGL, all six mission items (NAV_TAKEOFF, 4 NAV_WAYPOINT, NAV_RTL) reached. `.tlog` analysis: 51.1 m × 51.1 m extent, 10.03 m peak altitude. Multi-run replication (10 runs) and wind sweeps subsequently extended the same harness; physics-grounded T5 avoidance runs against Gazebo's `iris_obstacle.sdf` world via the `gazebo-iris` SITL frame (`sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON ...` — note the explicit `--add-param-file` is required because frame defaults aren't auto-applied for the JSON model; see `scripts/sitl/readme.md`).

- Path: `scripts/sitl/` (overlay + missions) + `scripts/sitl/gazebo/` (Gazebo Harmonic world + custom model)
- Entry point: `run_sitl.sh` (QUAV250 box-mission), `gz sim -v4 -r iris_obstacle.sdf` (Gazebo T5)
- Docs: [`readme.md`](../scripts/sitl/readme.md), [`missions/readme.md`](../scripts/sitl/missions/readme.md)

### Autonomy (state machine + planner + perception input)

Decision-making layer above the flight-controller bridge. Sits between perception output and the bridge's `Vehicle.send_velocity_ned` API, turning detected obstacles into avoidance manoeuvres.

Three modules:

- `state_machine.py` — passive observer of the bridge's `Vehicle`. Nine high-level flight states (`DISCONNECTED / NO_FIX / PREARMED / ARMED_ON_GROUND / ASCENDING / AIRBORNE / DESCENDING / DISARMED_POSTFLIGHT / FAULT`), Schmitt-trigger hysteresis on the autopilot-filtered climb rate, subscriber API for downstream consumers (LED driver, voice/gesture mappers, orchestrator).
- `planner.py` — reactive sector-based obstacle avoider. Modes: CRUISING (forward velocity) / AVOIDING (lateral sidestep) / IDLE / LANDING. Hysteresis on the obstacle distance threshold, cached perception frame to absorb perception-rate vs planner-rate mismatch, sidestep direction latched on entry.
- `perception_source.py` — `PerceptionFrame` / `Detection` data contract plus two concrete sources: `SyntheticPerceptionSource` (in-process scripted timeline for tests) and `DepthDetectSource` (tails the JSONL file emitted by `depth_detect.py --jsonl`, supporting replay-from-start, live-tail, and real-time pacing keyed off record timestamps).

Cross-subsystem imports `Vehicle` from `../flight-controller` via `sys.path`. Own venv (pymavlink + numpy).

Five closed-loop SITL tests demonstrate the full chain:

- `test_avoidance.py` — in-process synthetic perception, scripted obstacle timeline
- `test_avoidance_jsonl.py` — JSONL replay of pre-recorded `depth_detect.py` output
- `test_avoidance_gazebo.py` — physics-grounded obstacle in Gazebo Harmonic with simulated forward-facing lidar feeding the autonomy stack via `gazebo_perception_bridge.py` (T5 dissertation evidence; min 1.92 m clearance from a collision-bodied cylinder)
- `test_hover_stability.py` — T4 hover stability (38 mm max h-drift over 60 s @ 5 m)
- `test_mission_replicate.py` — T6 multi-run replication (10 sequential box missions, 100 % pass at ±5 % extent tolerance)
- `test_wind_sweep.py` — wind / disturbance rejection sweep at `SIM_WIND_SPD ∈ {0, 5, 10, 15}` m/s (8/8 missions + 4/4 hovers pass)

- Path: `scripts/autonomy/`
- Entry points: see test list above; data contract producers `make_synthetic_jsonl.py` (synthetic) and `gazebo_perception_bridge.py` (Gazebo lidar → JSONL)
- Docs: [`readme.md`](../scripts/autonomy/readme.md)

### Jarvis voice assistant

Local-first wake-word + ASR + LLM + TTS pipeline. Wake word via openWakeWord ("hey jarvis"), speech via PyAudio with amplitude-based VAD, transcription via faster-whisper (auto-sized by available RAM), response via Gemini API or local Ollama (factory-selected), synthesis via Piper TTS. All run on the Pi.

- Path: `scripts/jarvis/`
- Entry point: `jarvis.py`, installed as systemd service `jarvis`
- Docs: [`context.md`](../scripts/jarvis/context.md)

### Pico 2 W RGB status LEDs

Runs MicroPython on a Raspberry Pi Pico 2 W (not the Pi 5). WS2812 NeoPixels on GPIO6, controlled via the `neopixel` module in the Pico REPL. Used for visible flight-state feedback.

- Path: `scripts/pico-led/`
- Docs: [`setup-guide.md`](../scripts/pico-led/setup-guide.md)

### Boot-time IP notifier

Emails the drone's IP address on boot for headless SSH access over Wi-Fi. SMTP config in `config.json` (gitignored).

- Path: `scripts/email-ip-notifier/`
- Entry point: `send_ip_email.py`, installed as systemd service via `setup.sh`

## Deployment model

All subsystems target the Raspberry Pi 5 with USB microphone, USB speaker, USB cameras, and (where applicable) Pixhawk over serial/USB. Each subsystem is installed via its own `setup.sh` script, which creates the virtualenv, installs dependencies, and registers a systemd service with resource limits (CPUQuota, MemoryMax). Services run as a non-root user with the `audio` supplementary group where needed.

Subsystem startup order is not coordinated: each starts independently on boot. Inter-subsystem communication (e.g. perception → flight commands) will use MAVLink and local IPC when those components are implemented.

## Configuration

- Per-subsystem `config.json` files (gitignored), with `config.example.json` templates checked in
- Jarvis config selects the LLM backend (`"llm_backend": "ollama"` or `"gemini"`), Whisper model size, wake word, voice, and conversation parameters

## Status

| Capability | Status |
|---|---|
| Flight controller bring-up (ArduCopter 4.6.3, ELRS, failsafes, motor direction) | done — Stage 2 closed |
| NPU selection (Benchy-driven, Hailo-8 over Hailo-10H) | done |
| AR0144 stereo calibration | done — cal-4, RMS 1.10 px |
| YOLO detection on Hailo-8 | done — live |
| Stereo depth + YOLO fusion (handheld) | done — live, 138 ms loop |
| Voice assistant (Jarvis) | done — standalone subsystem |
| ArduPilot SITL bring-up (QUAV250 overlay) | done — first box mission flown |
| Pi-side MAVLink bridge (`scripts/flight-controller/bridge.py`) | done — live; smoke-tested against SITL and validated on the airframe |
| Flight-state machine (`scripts/autonomy/state_machine.py`) | done — live; 9 states, hysteresis on climb rate |
| Reactive obstacle-avoidance planner (`scripts/autonomy/planner.py`) | done — live; CRUISING/AVOIDING modes with hysteresis |
| Perception → bridge → SITL closed loop | done — demonstrated against synthetic + JSONL transports |
| `depth_detect.py --jsonl` detection-event pipe | done — live; autonomy `DepthDetectSource` consumes it |
| T4 hover stability (SITL) | done — 38 mm max h-drift over 60 s @ 5 m hover (bound 500 mm) |
| T6 multi-run mission replication (SITL) | done — 10/10 box_50m runs at +/-5 % extent tolerance |
| Wind / disturbance rejection sweep (SITL) | done — 0/5/10/15 m/s, 100 % pass on 8 missions + 4 hovers |
| T5 closed-loop avoidance (Gazebo Harmonic, physics-grounded obstacle) | done — 1.92 m clearance from cylinder surface (bound 0.3 m, 6.4x headroom) |
| Hardware-airframe stack validation walkaround | done — 2026-05-18, 3:34 min handheld, 138.1 ms loop reproduced on flight hardware |
| SiK 433 MHz telemetry link validation | done — 2026-05-18, 25 m indoor through-wall LOS at 0 packet loss |
| Orchestrator + gesture pipeline (code complete, perception-side untested against live Hailo) | pending live Hailo validation |
| Global path planner (A*/RRT*) on top of the reactive avoider | pending — Sec. 6.6 future work |
| First physical hover with imaging stack mounted | pending Stage 4 hardware flight |
| ROS 2 / Meshtastic / ATAK | pending — dissertation Sec. 6.6 future work |

See dissertation Ch 5 for the implementation roadmap and Ch 6 for the empirical results.

## Further reading

- [architecture.md](architecture.md) — system-level architecture and subsystem relationships
- [hardware.md](hardware.md) — hardware the software targets
- [`../CLAUDE.md`](../CLAUDE.md) — orientation for contributors using Claude Code
