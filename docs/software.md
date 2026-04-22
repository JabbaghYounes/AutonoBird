# Software

AutonoBird's software is organised as independent subsystems under `scripts/<name>/`, each with its own setup script, virtualenv, config template, and local docs.

## Subsystems

### Flight controller configuration

Not code — documentation and ArduPilot parameters for the Pixhawk 6C Mini. Port assignments, CRSF setup, OSD MAVLink configuration, pre-flight checklist. Future home for MAVLink bridge and companion-computer → Pixhawk command scripts (ROS2, MAVSDK).

- Path: `scripts/flight-controller/`
- Docs: [`readme.md`](../scripts/flight-controller/readme.md)

### AR0144 stereo depth

USB stereo camera calibration and depth estimation. Two calibration scripts: `guided_calibration.py` (recommended, 40-pose face-scan style across 4 distance zones) and `basic_calibration.py` (manual SPACE-to-capture). Uses OpenCV SGBM stereo matching.

- Path: `scripts/ar0144/`
- Entry points: `guided_calibration.py {capture|calibrate|depth|all}`
- Docs: [`steps.md`](../scripts/ar0144/steps.md), [`pose-list.md`](../scripts/ar0144/pose-list.md)

### Arducam IMX519 stereo depth (legacy)

Quad-camera kit using Picamera2 with I2C channel switching. Replaced by AR0144 due to rolling-shutter artefacts and GPIO conflict with the HAILO AI HAT+, but retained for reference and possible future use on a larger airframe.

- Path: `scripts/arducam/`
- Entry point: `stereo_depth.py {calibrate|compute_cal|depth|capture}`
- Docs: [`readme.md`](../scripts/arducam/readme.md)

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
- Exception: Arducam stereo config is a Python `CONFIG` dict at the top of `stereo_depth.py`
- Jarvis config selects the LLM backend (`"llm_backend": "ollama"` or `"gemini"`), Whisper model size, wake word, voice, and conversation parameters

## Not yet implemented

Target architecture (dissertation Ch 4) includes several components not yet written:

- MAVLink bridge between Pi and Pixhawk (MAVSDK or pymavlink)
- Path planner (A* / RRT*) consuming depth maps and producing waypoints
- Object detector on the HAILO NPU (YOLO or equivalent)
- Perception → avoidance control loop
- ROS 2 integration (optional, planned)
- Meshtastic mesh-radio module (optional, planned)
- ATAK integration (optional, planned)

See dissertation Ch 5 for the implementation roadmap.

## Further reading

- [architecture.md](architecture.md) — system-level architecture and subsystem relationships
- [hardware.md](hardware.md) — hardware the software targets
- [`../CLAUDE.md`](../CLAUDE.md) — orientation for contributors using Claude Code
