# AutonoBird

An AI-driven autonomous drone platform built on a HolyBro QUAV250 airframe, designed to demonstrate onboard perception, voice control, and mission execution using only edge hardware — no cloud connectivity required.

AutonoBird is a university dissertation project in Applied Computer Science. The aim is to show that a sub-£500 embedded system (Raspberry Pi 5 + HAILO AI HAT+) can deliver real-time obstacle detection and autonomous navigation comparable to commercial platforms on a fraction of the budget.

## Status

Hardware build complete as of April 2026. Software and firmware phase in progress.

## Documentation

Start with the documentation index:

- **[Architecture](docs/architecture.md)** — system overview, subsystems, data flow, control paths
- **[Hardware](docs/hardware.md)** — BOM, weight budget, power architecture, airframe history
- **[Software](docs/software.md)** — subsystem summaries, deployment model, roadmap

Subsystem-specific documentation lives alongside the code under `scripts/<subsystem>/`:

- [`scripts/flight-controller/readme.md`](scripts/flight-controller/readme.md) — Pixhawk port map, ArduPilot parameters, build log, pre-flight checklist
- [`scripts/ar0144/steps.md`](scripts/ar0144/steps.md) — stereo camera calibration
- [`scripts/jarvis/context.md`](scripts/jarvis/context.md) — local voice assistant
- [`scripts/pico-led/setup-guide.md`](scripts/pico-led/setup-guide.md) — status LEDs on the Pico 2 W

## Licence

Project licence to be confirmed.
