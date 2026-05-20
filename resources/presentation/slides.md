# AutonoBird

AI-Driven Autonomous Drone Platform

BSc Applied Computer Science — Individual Project

University of Plymouth

21 May 2026

# The problem

- Consumer drones: cheap, closed, no onboard compute
- Defence-grade autonomous drones: capable but £10k+ and closed
- Gap: a £1000-class drone with NPU-accelerated perception and reactive autonomy

**Research question.** Can a Raspberry Pi 5 with a Hailo-8 NPU deliver vision-only autonomous obstacle avoidance on a 250 mm-class quadcopter under £1000?

# System architecture

![QUAV250 airframe with Pi 5 + AI HAT+ + AR0144 stereo on the top deck](../dissertation/figure-1-airframe-completed-topview.jpg){width=72%}

# The airframe

![Front and bottom views of the assembled QUAV250 at 945 g all-up weight](../images/Drone/front-view.jpg){width=46%} ![](../images/Drone/bottom-view.jpg){width=46%}

# Hardware summary

- QUAV250 carbon frame, ArduCopter 4.6.3
- **945 g all-up weight, 2.5:1 thrust-to-weight**
- **£761 total BOM** (under the £1000 budget)
- 4S 1500 mAh LiPo, UBEC USB-C feeding the Pi
- Pixhawk 6C Mini + Pi 5 + AI HAT+ + AR0144 stereo + SiK 433 MHz telemetry

# Perception pipeline

- Stereo capture: AR0144 USB, 13 fps @ 2560x720 MJPEG
- Rectify + SGBM half-resolution depth — 76 ms / frame
- YOLOv8n on Hailo-8 NPU — 18.5 ms inference
- Median-of-5x5-patch depth fusion at bounding-box centroids

**End-to-end: 138 ms / 6.8 fps on Pi 5 — under the 250 ms NFR1 target.**

# Live perception output

![YOLO detections with bounding boxes + depth fusion + colourised SGBM heatmap](../dissertation/figure-6-walkaround-perception.png){width=80%}

# NPU selection rationale

- Two candidate HATs: Hailo-8 (26 TOPS INT8) vs Hailo-10H (40 TOPS INT4)
- Selected empirically via the **Benchy** benchmark suite — identical YOLO models on both
- Hailo-8 wins YOLOv8n at 640x640 by **2.6x to 6.9x**
- Hailo-10H wins LLM workloads (10.35 TPS on llama3.2:1b) — recorded for future text-to-intent, not the vision-first autonomy path
- Decision: Hailo-8 on the airframe

# NPU throughput comparison

![Hailo-8 vs Hailo-10H YOLO throughput at 640x640, Benchy benchmark suite](../dissertation/figure-8-NPU-throughput-comparison.png){width=72%}

# Autonomy stack

- **Pi-side MAVLink bridge** — `Vehicle` class wrapping pymavlink, same code targets SITL + real Pixhawk
- **State machine** — 9 flight states with Schmitt-trigger hysteresis on autopilot-filtered climb rate
- **Reactive planner** — CRUISING / AVOIDING / IDLE / LANDING; sector-based sidestep at 10 Hz
- **Orchestrator** — single-process bring-up of bridge -> FSM -> planner -> optional gesture pipeline -> optional Pico LED status
- **Intent surface** — `command_hold / resume / land / rtl` — what voice, gestures, and future inputs bind into

# SITL hover stability (T4)

![60 s GUIDED hover at 5 m AGL — 582 samples, max horizontal drift 38 mm, RMS 19 mm](../../scripts/sitl/logs/t4_hover_20260516-185701.png){width=80%}

# SITL mission replicate (T6)

![10 sequential 50 m x 50 m box missions — 51.12 ± 0.07 m extents, 100% pass rate](../../scripts/sitl/logs/t6_replicate_20260516-203126.png){width=80%}

# SITL wind disturbance rejection

![SIM_WIND_SPD swept across 0/5/10/15 m/s — hover h-drift stays under 120 mm; T6 mission pass rate 100% at every wind level](../../scripts/sitl/logs/wind_sweep_20260516-214041.png){width=80%}

# Gazebo collision avoidance (T5)

![Custom iris + forward GPU lidar + 12 m red cylinder obstacle — min clearance 1.92 m, 6.4x the 0.3 m bound](../dissertation/figure-4-T5-gazebo-clearance.png){width=80%}

# Hardware-airframe stack validation

- 2026-05-18: airframe-mounted Pi + real Pixhawk, 3:34 min handheld walkaround, motors off
- **138.1 ms loop EMA** — reproduces bench baseline on the mounted platform
- 1449 frames in JSONL, **14 distinct YOLO classes**, 161 planner transitions on real stereo
- **Zero** bridge errors / timeouts / under-voltage events over the full session
- First hardware-validation of the perception + autonomy software path

# SiK 433 MHz telemetry link

![RSSI 219 baseline -> 145 at 25 m through multiple walls — 0 packet loss across 239 samples](../../scripts/sitl/logs/sik_los_20260518-2345_plot.png){width=80%}

# Requirements verdict — supported in SITL plus hardware-validated software stack

| | Met | Partial | Unmeasured |
|---|---:|---:|---:|
| Functional (FR1–FR8) | 6 | 1 | 1 |
| Non-functional (NFR1–NFR5) | 4 | 2 | 1 |
| **Total** | **10** | **3** | **2** |

# Limitations

- No first powered hover with the imaging stack mounted — deferred behind dissertation submission
- Outdoor compass calibration pending — gates the first hover
- Gesture pipeline is code-complete on both sides but untested against live Hailo pose decoder
- All telemetry distance tests so far are indoor through-wall, not outdoor LOS

# Future work

- Indoor netted hover -> outdoor open-field flight
- Live gesture validation (STOP / LAND / COME / RECEDE on COCO-17 keypoints)
- Person-recognition deployment via OSNet/CLIP + FAISS embeddings
- ROS2, Meshtastic mesh radio, ATAK tactical-overlay integration
- Vibration analysis post first powered flight

# Live demonstration

Three parts:

1. **Gazebo T5 outdoor flight** — recorded — 12 m obstacle, 1.92 m clearance
2. **Gazebo classroom replica** — recorded — same algorithms in a virtual copy of this room
3. **Live perception on the airframe** — handheld, motors off — YOLO detections + colourised depth heatmap + planner-would-command logs

# Thank you

Questions welcome.

**Supporting tools** (cited in the dissertation):

- **Benchy** — edge-AI NPU benchmark suite — github.com/JabbaghYounes/Benchy
- **UPSentinel** — Waveshare UPS HAT tray indicator — github.com/JabbaghYounes/UPSentinel

Younes Jabbagh — 10771837@cityplym.ac.uk
