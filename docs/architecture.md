# Architecture

AutonoBird is an AI-driven autonomous drone built on a HolyBro QUAV250 carbon fibre airframe. The system is organised as a set of loosely coupled subsystems, each responsible for one concern, communicating through well-defined interfaces (MAVLink, USB, serial).

## System overview

```
┌───────────────────────────────────────────────────────────────────┐
│                          AutonoBird                                │
│                                                                    │
│  ┌──────────────┐        ┌─────────────────┐    ┌──────────────┐  │
│  │  Perception  │───────▶│  Raspberry Pi 5  │───▶│  Pixhawk 6C  │  │
│  │  (stereo +   │  USB   │  + HAILO AI HAT+ │    │  Mini        │  │
│  │   NPU infer) │        │                  │    │  (ArduPilot) │  │
│  └──────────────┘        └─────────────────┘    └──────────────┘  │
│         │                         │                     │          │
│         │                         │                     │          │
│         │                 ┌───────▼────────┐            ▼          │
│         │                 │  Voice asst    │       ESCs / motors   │
│         │                 │  (Jarvis)      │                       │
│         │                 └────────────────┘                       │
│         │                                                          │
│  ┌──────▼──────┐          ┌─────────────────┐                     │
│  │ AR0144      │          │  Ground station │                     │
│  │ stereo USB  │          │  (SiK + laptop) │                     │
│  └─────────────┘          └─────────────────┘                     │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

## Subsystems

Each subsystem lives under `scripts/<name>/` with its own setup script, systemd service (where applicable), config template, and local docs. Subsystems are independent — failure of one (voice assistant, for example) does not affect the others.

| Subsystem | Path | Purpose | Status |
|---|---|---|---|
| Flight controller + bridge | `scripts/flight-controller/` | Pixhawk 6C Mini configuration + port assignments + Pi-side MAVLink bridge (`bridge.py`) | Stage 2 closed; bridge live (UDP→SITL + serial→Pixhawk) |
| Autonomy | `scripts/autonomy/` | Flight-state machine + reactive obstacle-avoidance planner + perception-input abstraction (synthetic + JSONL-tail + Gazebo lidar bridge) | Closed-loop SITL avoidance demonstrated against three perception transports (synthetic, JSONL replay, Gazebo physics-grounded); T4 hover + T6 multi-run + wind-sweep closed in SITL |
| Stereo calibration (AR0144) | `scripts/ar0144/` | 40-pose guided calibration + depth viewer + pose-reference PDF generator | cal-4 calibration complete (RMS 1.10 px) |
| Perception (YOLO + depth) | `scripts/perception/` | YOLOv8n on Hailo-8 + stereo depth fusion (`yolo_detect.py`, `depth_detect.py`); JSONL detection-event emitter for the autonomy stack | Running end-to-end at 6.8 fps; JSONL pipe wired |
| ArduPilot SITL | `scripts/sitl/` | QUAV250 parameter overlay + launch wrapper + scripted missions for the simulated flight stack (dev-workstation, not Pi); Gazebo Harmonic integration under `gazebo/` (custom `iris_with_lidar` model + `iris_obstacle.sdf` world for T5) | Box mission flown, multi-run replication closed, wind sweep closed, T5 closed-loop against physics-grounded Gazebo obstacle passed (1.92 m clearance) |
| Voice assistant | `scripts/jarvis/` | Local wake-word + ASR + LLM + TTS pipeline | Standalone subsystem |
| Pico LED indicators | `scripts/pico-led/` | Status LEDs on a separate Pico 2 W (MicroPython) | Standalone subsystem |
| Boot IP notifier | `scripts/email-ip-notifier/` | Emails the drone's IP on boot for headless SSH access | Standalone subsystem |

## Data flow (runtime perception loop — implemented)

The handheld perception loop runs end-to-end on the calibration / perception rig (Pi 5 + AI HAT+ + AR0144 + UPS HAT). Measured stage-by-stage:

1. **AR0144 stereo camera** captures side-by-side 2560×720 MJPEG over USB 2.0 (~13 fps cap from the camera's USB controller).
2. **Pi 5** splits the frame, rectifies each half using cv2.remap with calibration maps from `stereo_calibration.npz`. ~5 ms.
3. **SGBM disparity** runs on the rectified halves at half resolution (640×360) — Pi 5 CPU, ~76 ms. Half-res is a pre-emptive optimisation; full-res would saturate the loop.
4. **Disparity → depth** via Q-matrix focal length + 52 mm baseline. Median-of-5×5-patch lookup at each detection bbox centroid.
5. **HAILO AI HAT+ (Hailo-8)** runs YOLOv8n on the rectified left frame letterboxed to 640×640 — ~18.5 ms end-to-end (NPU + HailoRT overhead).
6. **Fusion**: each detection annotated with `<class> <score> @ <depth>m`.
7. **End-to-end loop**: 138 ms / 6.8 fps. Under the 250 ms NFR1 target.

The Pi-side MAVLink bridge (`scripts/flight-controller/bridge.py`) and reactive autonomy stack (`scripts/autonomy/`) are both live. The bridge connects to either SITL via UDP (development) or the real Pixhawk via USB-serial (deployment) — only the URI changes. The autonomy stack runs against three perception transports: in-process synthetic, JSONL replay of recorded `depth_detect.py` output, and a Gazebo Harmonic forward-facing lidar bridge for physics-grounded T5 evaluation. A global path planner (A* / RRT*) on top of the reactive avoider is recorded as future work (dissertation § 6.6).

Success criteria for this loop are defined in the dissertation (§ 6.1 / § 6.2) and the [hardware](hardware.md) doc.

## Control paths

Three independent control paths run concurrently:

- **RC link** (ExpressLRS via RadioMaster Pocket) — manual pilot override, always available when receiver is bound
- **Ground station** (SiK telemetry to Mission Planner) — waypoint upload, parameter changes, live telemetry monitoring
- **Autonomous** (Pi → Pixhawk over MAVLink) — perception-driven waypoint/offboard commands from the companion computer

The RC link is authoritative: a pilot stick input overrides any autonomous command. This is a safety-critical invariant.

## Power architecture

A single 4S LiPo powers the entire vehicle:

```
LiPo 4S ──▶ PDB ──┬──▶ ESCs ──▶ motors
                   │
                   ├──▶ Pixhawk + avionics (via PDB 5V rail)
                   │
                   └──▶ UBEC (step-down) ──▶ USB-C ──▶ Pi 5 (+ AI HAT)
```

The UBEC isolates the Pi's 5V rail from motor current spikes. See [hardware](hardware.md) for sizing, placement constraints, and weight breakdown.

## See also

- [hardware.md](hardware.md) — frame, components, weight, placement
- [software.md](software.md) — software stack, subsystems, deployment
- [../scripts/flight-controller/readme.md](../scripts/flight-controller/readme.md) — Pixhawk port map, parameters, pre-flight checklist
- [../CLAUDE.md](../CLAUDE.md) — orientation for contributors using Claude Code
