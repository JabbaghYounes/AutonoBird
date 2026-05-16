# AutonoBird

An AI-driven autonomous drone platform built on a HolyBro QUAV250 airframe, designed to demonstrate onboard perception, voice control, and mission execution using only edge hardware — no cloud connectivity required.

AutonoBird is a university dissertation project in Applied Computer Science. The aim is to show that a sub-£500 embedded system (Raspberry Pi 5 + HAILO AI HAT+) can deliver real-time obstacle detection and autonomous navigation comparable to commercial platforms on a fraction of the budget.

## Status

- **Hardware build**: complete (Stage 2 closed April 2026; AUW 850 g without imaging stack)
- **Flight controller**: ArduCopter 4.6.3 on Pixhawk 6C Mini, parameters + bench validation done
- **NPU selection**: AI HAT+ (Hailo-8) selected over AI HAT+ 2 (Hailo-10H) per the Benchy benchmark suite — 2.6×–6.9× YOLO throughput advantage at 640×640
- **Stereo calibration**: AR0144 calibrated 2026-05-15 (cal-4 session, 28 pairs, RMS 1.10 px, baseline 51.92 mm validated, accurate band 0.3–1.5 m)
- **Perception loop live on airframe**: stereo depth + YOLO detection fused, running at 6.8 fps end-to-end with 18.5 ms NPU + 76 ms SGBM (under the 250 ms NFR1 target)
- **ArduPilot SITL bring-up**: dev-workstation SITL build with QUAV250 parameter overlay, first scripted mission flown autonomously (50 m × 50 m box at 10 m AGL, all 6 mission items reached, .tlog captured)
- **Pi-side MAVLink bridge**: `scripts/flight-controller/bridge.py` — transport-agnostic `Vehicle` class (UDP for SITL via MAVProxy, USB-serial for real Pixhawk), smoke-tested end-to-end against SITL
- **Autonomy stack**: `scripts/autonomy/` — flight-state machine + reactive obstacle-avoidance planner + perception input abstraction (synthetic + JSONL-tail sources). Closed-loop SITL avoidance demonstrated against two perception transports (synthetic in-process + JSONL replay of `depth_detect.py` output), drone sidesteps ~10 m east on simulated obstacle injection and resumes the cruise heading
- **Perception → autonomy JSONL pipe**: `depth_detect.py --jsonl PATH` emits a structured detection-event stream that the autonomy-side `DepthDetectSource` tails — same code drives synthetic tests, JSONL replay, or live perception
- **Remaining**: physical mount of imaging stack on the airframe + first hover (Stage 4 hardware flight)

## Documentation

Top-level docs live in `docs/`:

- **[Architecture](docs/architecture.md)** — system overview, subsystems, data flow, control paths
- **[Hardware](docs/hardware.md)** — BOM, weight budget, power architecture, airframe history
- **[Software](docs/software.md)** — subsystem summaries, deployment model, roadmap

Subsystem-specific documentation lives alongside the code under `scripts/<subsystem>/`:

- [`scripts/flight-controller/readme.md`](scripts/flight-controller/readme.md) — Pixhawk port map, ArduPilot parameters, build log, pre-flight checklist, **Pi-side MAVLink bridge**
- [`scripts/autonomy/readme.md`](scripts/autonomy/readme.md) — flight-state machine, reactive avoider, perception sources, closed-loop SITL tests
- [`scripts/ar0144/steps.md`](scripts/ar0144/steps.md) — stereo camera calibration (40-pose guided)
- [`scripts/perception/readme.md`](scripts/perception/readme.md) — YOLO detection + depth-fused perception on Hailo-8 (includes the `--jsonl` detection-event emitter that feeds the autonomy stack)
- [`scripts/sitl/readme.md`](scripts/sitl/readme.md) — ArduPilot SITL setup, QUAV250 parameter overlay, scripted missions
- [`scripts/jarvis/context.md`](scripts/jarvis/context.md) — local voice assistant
- [`scripts/pico-led/setup-guide.md`](scripts/pico-led/setup-guide.md) — status LEDs on the Pico 2 W
- [`docs/engineering-backlog.md`](docs/engineering-backlog.md) — post-dissertation engineering work tracker (~40 items across 10 categories)

## Licence

Project licence to be confirmed.
