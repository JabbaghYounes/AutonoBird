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

| Subsystem | Path | Purpose |
|---|---|---|
| Flight controller | `scripts/flight-controller/` | Pixhawk 6C Mini configuration, port assignments, build notes |
| Stereo depth (AR0144) | `scripts/ar0144/` | USB stereo camera calibration and depth estimation |
| Stereo depth (Arducam) | `scripts/arducam/` | Quad-camera kit alternative (replaced by AR0144 — retained for reference) |
| Voice assistant | `scripts/jarvis/` | Local wake-word + ASR + LLM + TTS pipeline |
| Pico LED indicators | `scripts/pico-led/` | Status LEDs on a separate Pico 2 W (MicroPython) |
| Boot IP notifier | `scripts/email-ip-notifier/` | Emails the drone's IP on boot for headless SSH access |

## Data flow (runtime perception loop — target)

Once the software stack is complete, the runtime perception loop will flow as follows:

1. **AR0144 stereo camera** captures side-by-side 2560×720 global-shutter frames over USB to the Pi.
2. **Pi 5** splits the frame to left/right, rectifies using pre-computed calibration maps, computes disparity via OpenCV SGBM.
3. **Depth map** is segmented into threat zones.
4. **HAILO AI HAT+** runs YOLO object detection on the same frames.
5. Perception outputs feed a **path-planning** module (A* / RRT*) running on the Pi.
6. Commands are streamed to the **Pixhawk** over MAVLink (UART or Ethernet).
7. Pixhawk executes flight commands and returns telemetry to the Pi and to the ground station via the **SiK 433 MHz telemetry radio**.

Success criteria for this loop are defined in the dissertation and the [hardware](hardware.md) doc.

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
