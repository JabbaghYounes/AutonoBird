# AutonoBird

An AI-driven autonomous drone platform built on a HolyBro QUAV250 airframe, designed to demonstrate onboard perception, voice control, and mission execution using only edge hardware — no cloud connectivity required.

AutonoBird is a university dissertation project in Applied Computer Science. The aim is to show that a sub-£500 embedded system (Raspberry Pi 5 + HAILO AI HAT+) can deliver real-time obstacle detection and autonomous navigation comparable to commercial platforms on a fraction of the budget.

## Status (TL;DR)

- **Hardware**: built and mounted on the airframe. AUW 945 g, T/W ~2.5:1. ArduCopter 4.6.3 on Pixhawk 6C Mini with all calibrations, failsafes, and MAVLink stream rates committed to EEPROM. First motors-spinning hover deferred behind dissertation submission.
- **Perception**: stereo + YOLO + depth fusion at 138 ms / 6.8 fps on the Hailo-8 NPU — under the 250 ms NFR1 target. Reproduced on the airframe-mounted Pi during the 2026-05-18 walkaround.
- **Autonomy**: closed-loop SITL avoidance (synthetic + JSONL + Gazebo physics-grounded) all pass; T4 hover (38 mm h-drift), T6 multi-run (10/10), wind sweep, and Gazebo T5 (1.92 m clearance) closed in SITL. Orchestrator + gesture pipeline code-complete on both sides.
- **Telemetry**: SiK 433 MHz validated to 25 m indoor through-wall LOS at 0 packet loss.
- **Dissertation**: V14 submitted 2026-05-20 — verdict **"supported in SITL plus hardware-validated software stack"**. Read `resources/Individual_Project_V12.md` for the unredacted ~33,500-word reference (V14 was compressed to fit the 15,000-word spec cap).
- **Remaining**: Stage 4 hardware flight (outdoor compass cal, CG re-check, thrust-margin test, first indoor netted hover); live Hailo pose-pipeline validation; Pico LED hardware swap.

Full milestone log: [`docs/status.md`](docs/status.md). Outstanding engineering work: [`docs/engineering-backlog.md`](docs/engineering-backlog.md).

## Documentation

Top-level docs live in `docs/`:

- [`architecture.md`](docs/architecture.md) — system overview, subsystems, data flow, control paths
- [`hardware.md`](docs/hardware.md) — BOM, weight budget, power architecture, airframe history
- [`software.md`](docs/software.md) — subsystem summaries, deployment model, capability table
- [`status.md`](docs/status.md) — detailed build milestone log
- [`engineering-backlog.md`](docs/engineering-backlog.md) — post-dissertation engineering work tracker

Subsystem-specific documentation lives alongside the code under `scripts/<subsystem>/`:

- [`scripts/flight-controller/readme.md`](scripts/flight-controller/readme.md) — Pixhawk port map, ArduPilot parameters, build log, pre-flight checklist, **Pi-side MAVLink bridge**
- [`scripts/autonomy/readme.md`](scripts/autonomy/readme.md) — flight-state machine, reactive avoider, perception sources, closed-loop SITL tests, **orchestrator**, **gesture pipeline + voice-vs-gesture rationale**
- [`scripts/perception/readme.md`](scripts/perception/readme.md) — YOLO detection + depth-fused perception on Hailo-8 (includes the `--jsonl` detection-event emitter that feeds the autonomy stack)
- [`scripts/sitl/readme.md`](scripts/sitl/readme.md) — ArduPilot SITL setup, QUAV250 parameter overlay, scripted missions, **Gazebo Harmonic obstacle world + iris_with_lidar model** under `scripts/sitl/gazebo/`
- [`scripts/ar0144/steps.md`](scripts/ar0144/steps.md) — stereo camera calibration (40-pose guided)
- [`scripts/jarvis/context.md`](scripts/jarvis/context.md) — local voice assistant
- [`scripts/pico-led/setup-guide.md`](scripts/pico-led/setup-guide.md) — status LEDs on the Pico 2 W (code complete, hardware blocked)
- [`scripts/demo/readme.md`](scripts/demo/readme.md) — one-command tmux launchers for the May 2026 presentation demos

## Licence

[MIT](LICENSE). © 2025 Younes Jabbagh.
