# AutonoBird

An AI-driven autonomous drone platform built on a HolyBro QUAV250 airframe, designed to demonstrate onboard perception, voice control, and mission execution using only edge hardware — no cloud connectivity required.

AutonoBird is a university dissertation project in Applied Computer Science. The aim is to show that a sub-£500 embedded system (Raspberry Pi 5 + HAILO AI HAT+) can deliver real-time obstacle detection and autonomous navigation comparable to commercial platforms on a fraction of the budget.

## Status

- **Hardware build**: complete. Pi 5 + AI HAT+ + AR0144 stereo camera mounted on the QUAV250 top deck (2026-05-17) with rubber-grommet + plastic-screw vibration isolation. AUW with 4S 1500 mAh battery: **945 g** (within 0.5 % of design target; T/W ~2.5:1). Stage 2 avionics bring-up closed April 2026
- **Flight controller**: ArduCopter 4.6.3 on Pixhawk 6C Mini, parameters + bench validation done. Bench config session 2026-05-18 wrote `ARMING_CHECK=1`, geofence (`FENCE_ENABLE=1`, 20 m radius, RTL on breach), and `SR0_*` / `SR1_*` MAVLink stream rates to EEPROM
- **NPU selection**: AI HAT+ (Hailo-8) selected over AI HAT+ 2 (Hailo-10H) per the Benchy benchmark suite — 2.6×–6.9× YOLO throughput advantage at 640×640
- **Stereo calibration**: AR0144 calibrated 2026-05-15 (cal-4 session, 28 pairs, RMS 1.10 px, baseline 51.92 mm validated, accurate band 0.3–1.5 m)
- **Perception loop**: stereo depth + YOLO detection fused, running at 6.8 fps end-to-end with 18.5 ms NPU + 76 ms SGBM (under the 250 ms NFR1 target). Reproduced **on the airframe-mounted Pi** to within noise (138.1 ms loop / 6.76 Hz) during the 2026-05-18 walkaround
- **ArduPilot SITL bring-up**: dev-workstation SITL build with QUAV250 parameter overlay, first scripted mission flown autonomously (50 m × 50 m box at 10 m AGL, all 6 mission items reached, .tlog captured)
- **Pi-side MAVLink bridge**: `scripts/flight-controller/bridge.py` — transport-agnostic `Vehicle` class (UDP for SITL via MAVProxy, USB-serial for real Pixhawk), smoke-tested end-to-end against SITL
- **Autonomy stack**: `scripts/autonomy/` — flight-state machine + reactive obstacle-avoidance planner + perception input abstraction (synthetic + JSONL-tail sources). Closed-loop SITL avoidance demonstrated against two perception transports (synthetic in-process + JSONL replay of `depth_detect.py` output), drone sidesteps ~10 m east on simulated obstacle injection and resumes the cruise heading
- **Perception → autonomy JSONL pipe**: `depth_detect.py --jsonl PATH` emits a structured detection-event stream that the autonomy-side `DepthDetectSource` tails — same code drives synthetic tests, JSONL replay, or live perception
- **T4 hover stability (SITL)**: 60 s GUIDED auto-hold at 5 m, max horizontal drift 38 mm (two orders of magnitude under the 500 mm dissertation bound), max vertical drift 99 mm
- **T6 multi-run mission replication (SITL)**: 10 sequential box_50m runs, 100 % pass rate at ±5 % extent tolerance, mean extents 51.12 ± 0.07 m N / 51.17 ± 0.06 m E / 10.02 ± 0.00 m alt
- **Wind / disturbance rejection (SITL)**: `SIM_WIND_SPD ∈ {0, 5, 10, 15}` m/s sweep, 8/8 T6 missions + 4/4 T4 hovers pass — hover h-drift stays under 50 mm at 0–10 m/s, kicks to 118 mm at 15 m/s (still 4× under bound)
- **T5 closed-loop avoidance against Gazebo-physics-grounded obstacle**: full Gazebo Harmonic integration — custom `iris_with_lidar` model + `iris_obstacle.sdf` world with a 12 m tall cylinder, simulated forward-facing lidar feeds the autonomy stack via a `gz topic` → JSONL bridge. Min clearance 1.92 m from the cylinder surface (6.4× over the 0.3 m T5 bound), planner cycles CRUISING → AVOIDING → CRUISING, bird passes obstacle to 17 m N with 3.45 m east sidestep, RTL + disarm clean
- **Orchestrator**: `scripts/autonomy/orchestrator.py` — single-process bring-up of Vehicle bridge + FSM + optional Planner + optional Gesture pipeline + optional Pico LED bridge. Defaults to monitor mode (no commands sent — safe everywhere). Exposes `command_hold / command_resume / command_land / command_rtl` as the single intent surface that gesture (and future voice / gamepad) modalities bind into. Periodic 1 Hz STATUS line + FSM/planner transition logs. SITL smoke-tested
- **Gesture pipeline (replaces voice as primary in-flight modality)**: body-pose recognition of STOP / LAND / COME / RECEDE from COCO-17 keypoints. `scripts/autonomy/{gesture_classifier,gesture_action_map}.py` with confidence gating + temporal smoothing + cooldown. Perception side: `scripts/perception/depth_detect.py --pose` loads YOLOv8n-pose on the Hailo-8 NPU and emits keypoints in the JSONL. Code complete both sides; live Hailo validation pending (side-tabled until post-submission). Voice (FR6 Met via Jarvis) stays useful for bench / pre-flight commands but is demoted from the in-flight command path — motor noise + acoustic SNR make on-board voice recognition unworkable airborne
- **Hardware-airframe stack validation (2026-05-18)**: full software path exercised on the airframe-mounted Pi against the real Pixhawk for the first time. `scripts/autonomy/physical-drone-test.sh` (bundled bench + walkaround runner with tmux session management) ran a 3:34 min handheld walk with motors off. **1449 frames, 138.1 ms loop EMA reproduced on flight hardware, 161 planner CRUISING ↔ AVOIDING transitions on real stereo perception, zero `BridgeError` / timeout / `dmesg` under-voltage events.** Logs at `scripts/sitl/logs/walk_20260518-1755/`. This is the first end-to-end demonstration of the autonomy software stack on the actual airframe
- **SiK 433 MHz telemetry link validation (2026-05-18)**: `scripts/flight-controller/test_sik_link.py` — bench validation (RSSI 219 / 217 STRONG bidirectional, 240 ms median parameter round-trip latency, 1051 params synced via FTP) plus indoor through-wall walk-out at 25 m with **zero packet loss over 239 samples**. CSV + plot at `scripts/sitl/logs/sik_los_20260518-2345.{csv,_plot.png}`. RSSI plot generator: `scripts/flight-controller/plot_sik_los.py`
- **Dissertation**: read `resources/Individual_Project_V12.md` for the full design rationale + detailed measurement discussion (~33,500 chapter words). V14.docx (submitted 2026-05-20) is the trimmed academic snapshot — chapters at 14,765 words to fit the 15,000-word spec cap, achieved via 30 table-to-screenshot conversions and prose compression. Verdict: **"supported in SITL plus hardware-validated software stack"**. V12 is the working source for any post-defence revisions
- **Remaining**: Stage 4 hardware flight — outdoor compass cal on grass + CG re-check at 945 g + tethered low-throttle thrust-margin test + first indoor netted hover; live Hailo validation of the pose-pipeline gesture path

## Documentation

Top-level docs live in `docs/`:

- **[Architecture](docs/architecture.md)** — system overview, subsystems, data flow, control paths
- **[Hardware](docs/hardware.md)** — BOM, weight budget, power architecture, airframe history
- **[Software](docs/software.md)** — subsystem summaries, deployment model, roadmap

Subsystem-specific documentation lives alongside the code under `scripts/<subsystem>/`:

- [`scripts/flight-controller/readme.md`](scripts/flight-controller/readme.md) — Pixhawk port map, ArduPilot parameters, build log, pre-flight checklist, **Pi-side MAVLink bridge**
- [`scripts/autonomy/readme.md`](scripts/autonomy/readme.md) — flight-state machine, reactive avoider, perception sources, closed-loop SITL tests, **orchestrator**, **gesture pipeline + voice-vs-gesture rationale**
- [`scripts/pico-led/readme.md`](scripts/pico-led/readme.md) — Pico LED bridge (code complete, hardware blocked — Pico unit needs replacement)
- [`scripts/ar0144/steps.md`](scripts/ar0144/steps.md) — stereo camera calibration (40-pose guided)
- [`scripts/perception/readme.md`](scripts/perception/readme.md) — YOLO detection + depth-fused perception on Hailo-8 (includes the `--jsonl` detection-event emitter that feeds the autonomy stack)
- [`scripts/sitl/readme.md`](scripts/sitl/readme.md) — ArduPilot SITL setup, QUAV250 parameter overlay, scripted missions, **Gazebo Harmonic obstacle world + iris_with_lidar model** under `scripts/sitl/gazebo/`
- [`scripts/jarvis/context.md`](scripts/jarvis/context.md) — local voice assistant
- [`scripts/pico-led/setup-guide.md`](scripts/pico-led/setup-guide.md) — status LEDs on the Pico 2 W
- [`docs/engineering-backlog.md`](docs/engineering-backlog.md) — post-dissertation engineering work tracker (~40 items across 10 categories)

## Licence

[MIT](LICENSE). © 2025 Younes Jabbagh.
