# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutonoBird is an AI-driven autonomous drone platform for Raspberry Pi (4/5). It combines a voice assistant (Jarvis), stereo vision depth mapping, and boot-time IP notification into independent subsystems under `scripts/`.

## Repository Structure

```
scripts/
├── jarvis/           # Voice assistant: wake word → STT → LLM → TTS pipeline
├── ar0144/           # AR0144 stereo camera guided calibration (40-pose, 4 distance zones)
├── arducam/          # Arducam IMX519 quad-camera stereo depth mapping
└── email-ip-notifier/# Boot-time IP email notification (stdlib only, no deps)
```

Each subsystem is self-contained with its own setup script, systemd service, and config files.

## Development Commands

### Jarvis Voice Assistant
```bash
cd scripts/jarvis
sudo bash setup.sh                  # Install system packages, venv, download models
cp config.example.json config.json  # Then edit with API key/settings
./venv/bin/python3 jarvis.py        # Run locally
sudo systemctl start jarvis         # Run as service
journalctl -u jarvis -f             # View logs
```

### AR0144 Stereo Calibration
```bash
cd scripts/ar0144
python guided_calibration.py capture    # Interactive 40-pose guided capture
python guided_calibration.py calibrate  # Compute calibration from captures
python guided_calibration.py depth      # View depth map
python guided_calibration.py all        # Full pipeline
```

### Arducam IMX519 Stereo Depth
```bash
cd scripts/arducam
python3 stereo_depth.py calibrate       # Capture checkerboard poses
python3 stereo_depth.py compute_cal     # Compute intrinsics/extrinsics
python3 stereo_depth.py depth           # Real-time depth viewer
```

### Email IP Notifier
```bash
cd scripts/email-ip-notifier
sudo bash setup.sh             # Install service
python3 send_ip_email.py       # Test manually
```

## Architecture Notes

**Jarvis** (`scripts/jarvis/jarvis.py`, ~700 lines) is class-based with these components:
- **WakeWordListener** — openWakeWord for local "hey jarvis" detection
- **SpeechRecorder** — PyAudio with VAD (silence detection)
- **Transcriber** — faster-whisper, auto-selects model size by available RAM (tiny/base/small)
- **Brain** — factory pattern: `GeminiBrain` (Gemini 2.0 Flash API) or `OllamaBrain` (local LLM via Ollama HTTP API with retry logic)
- **Speaker** — Piper TTS with WAV synthesis
- **AudioFeedback** — synthetic beeps for UX states (no external sound files)

Multi-turn conversation history is maintained with configurable `max_history_turns`.

**Stereo calibration** uses OpenCV with SGBM stereo matching and optional WLS filter (requires `opencv-contrib`). The AR0144 guided calibration uses 40 poses across 4 distance zones (NEAR 0.3–0.6m, MID-NEAR 0.6–1.2m, MID 1.2–2.5m, FAR 2.5–4.0m) with region-based positioning guidance.

## Configuration

All subsystems use JSON config files with `.example` templates:
- `config.json` files are **git-ignored** (contain API keys and SMTP credentials)
- Config is loaded from `config.json` with fallback to `config.example.json`

Jarvis config selects LLM backend (`"llm_backend": "ollama"` or `"gemini"`), whisper model, wake word, voice, and conversation settings.

## Key Dependencies

- **Python 3** with per-subsystem virtualenvs (`venv/` dirs are git-ignored)
- No centralized `requirements.txt` — dependencies installed via `setup.sh` scripts
- Camera scripts depend on `picamera2` (Raspberry Pi libcamera)
- WLS disparity filter gracefully falls back if `opencv-contrib` is unavailable

## Deployment

Target is Raspberry Pi with USB mic + speaker and cameras. Subsystems run as systemd services with resource limits (CPUQuota, MemoryMax). Services run as non-root user with supplementary `audio` group.
