# Software

AutonoBird's software is organised as independent subsystems under `scripts/<name>/`, each with its own setup script, virtualenv, config template, and local docs.

## Subsystems

### Flight controller configuration

Not code — documentation and ArduPilot parameters for the Pixhawk 6C Mini. Port assignments, CRSF setup, OSD MAVLink configuration, pre-flight checklist. Future home for MAVLink bridge and companion-computer → Pixhawk command scripts (ROS2, MAVSDK).

- Path: `scripts/flight-controller/`
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

## Status

| Capability | Status |
|---|---|
| Flight controller bring-up (ArduCopter 4.6.3, ELRS, failsafes, motor direction) | ✓ Stage 2 closed |
| NPU selection (Benchy-driven, Hailo-8 over Hailo-10H) | ✓ Resolved |
| AR0144 stereo calibration | ✓ cal-4, RMS 1.10 px |
| YOLO detection on Hailo-8 | ✓ Live |
| Stereo depth + YOLO fusion (handheld) | ✓ Live, 138 ms loop |
| Voice assistant (Jarvis) | ✓ Standalone subsystem |
| MAVLink bridge between Pi and Pixhawk | Pending (pymavlink or MAVSDK over USB-serial) |
| Path planner (A*/RRT*) consuming depth + detections | Pending |
| Perception → avoidance control loop | Pending — autonomy validation will go through ArduPilot SITL |
| ArduPilot SITL integration | Pending |
| First physical hover with imaging stack mounted | Deferred behind dissertation submission |
| ROS 2 / Meshtastic / ATAK | Future work (dissertation § 6.6) |

See dissertation Ch 5 for the implementation roadmap and Ch 6 for the empirical results.

## Further reading

- [architecture.md](architecture.md) — system-level architecture and subsystem relationships
- [hardware.md](hardware.md) — hardware the software targets
- [`../CLAUDE.md`](../CLAUDE.md) — orientation for contributors using Claude Code
