# Engineering backlog

Post-dissertation engineering work to finish the AutonoBird system end-to-end. The dissertation (frozen as `resources/Individual_Project_V5.docx`) honestly reports what is and is not built; this backlog tracks the work that closes the autonomy / integration / hardware-flight gaps.

Items are grouped by category and tagged by priority:

- 🔴 **critical** — blocks the autonomy claim or the first hardware hover
- 🟡 **important** — needed for whole-system completeness or before first flight
- 🟢 **nice-to-have** — polish, observability, hand-off

Each row has a status: `[ ]` open · `[~]` in progress · `[x]` done. Update inline as work proceeds.

---

## 1. Software autonomy — perception → flight bridge

The largest gap. `depth_detect.py` outputs to a screen; `arducopter` flies blindly. These two systems do not yet communicate.

| | Item | Pri | Notes |
|---|---|---|---|
| `[x]` | **Pi-side MAVLink bridge** in `scripts/flight-controller/` | 🔴 | ✅ 2026-05-16. pymavlink-based `Vehicle` class in `bridge.py`. Connects via `udpin:127.0.0.1:14550` (MAVProxy forwarder) for SITL dev, `serial:///dev/ttyACM0` for real Pixhawk. Sends 1 Hz GCS heartbeats. API: `set_mode`, `arm`, `takeoff`, `rtl`, `land`, `send_velocity_ned`, `upload_mission`, `state` property, `events()` queue. Smoke-tested end-to-end against SITL (`test_bridge.py`): connect → GUIDED → arm → takeoff 10 m → RTL → auto-land → disarm. |
| `[x]` | **State machine** for vehicle state | 🔴 | ✅ 2026-05-16. `scripts/autonomy/state_machine.py`. Nine states (DISCONNECTED / NO_FIX / PREARMED / ARMED_ON_GROUND / ASCENDING / AIRBORNE / DESCENDING / DISARMED_POSTFLIGHT / FAULT). Passive observer: consumes `Vehicle.events()` + `Vehicle.state`, infers transitions, exposes subscriber API. Uses VFR_HUD's filtered `climb_rate` + Schmitt-trigger hysteresis (enter at ±0.5 m/s, exit at ±0.15 m/s) to suppress flap at the AIRBORNE boundary. Smoke-tested against SITL — 9 clean transitions across a full takeoff/RTL cycle. |
| `[x]` | **Path planner** — reactive obstacle-aware command generation | 🔴 | ✅ 2026-05-16. `scripts/autonomy/planner.py` — Planner class with CRUISING / AVOIDING / IDLE / LANDING modes. Consumes a `PerceptionSource`, emits `Vehicle.send_velocity_ned` setpoints at 10 Hz. Hysteresis on entry/exit (obstacle_threshold_m / clear_threshold_m). Sidestep direction chosen from `Detection.bbox_centroid_norm`. Tested standalone with `_NullVehicle` + synthetic perception timeline. |
| `[x]` | **Perception data contract** — `PerceptionFrame` + `Detection` | 🔴 | ✅ 2026-05-16. `scripts/autonomy/perception_source.py`. SyntheticPerceptionSource for tests (scripted obstacle timeline); DepthDetectSource stub awaits JSONL emit from `scripts/perception/depth_detect.py`. |
| `[x]` | **End-to-end closed loop**: bridge → planner → SITL → avoid → resume | 🔴 | ✅ 2026-05-16. `scripts/autonomy/test_avoidance.py` (synthetic perception in-process) AND `scripts/autonomy/test_avoidance_jsonl.py` (JSONL replay via DepthDetectSource). Both PASS — drone takes off, cruises N, sidesteps E on obstacle injection, resumes, RTL + land. The T5 collision-avoidance test in the SITL track of § 5.4 is now demonstrated via two independent perception transports. Real-world / hardware T5 still pending. |
| `[ ]` | **Unified orchestrator / management entry-point** | 🟡 | Single CLI / service that brings up perception, bridge, state machine, voice, LEDs together. Per-subsystem `setup.sh` stays, but one `run.py` ties them at runtime. |

## 2. Hardware integration — imaging stack on the airframe

Bench rig stays handheld today. Until the stack is mounted on the drone, nothing in this section is exercised in flight conditions.

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **Mount AI HAT+ + AR0144 stereo camera on the airframe** | 🔴 | Forward-facing, slight down-tilt for indoor obstacle clearance. Carbon top plate or new 3D-printed bracket. |
| `[ ]` | **Vibration-damped camera mount** | 🔴 | Silicone grommets or foam tape. SGBM is motion-sensitive; § 6.3 explicitly notes airframe vibration on disparity is unmeasured. |
| `[ ]` | **LiPo-powered bench test of full compute stack** | 🟡 | UBEC → Pi 5 → AI HAT+ → AR0144 stack drawing from the 4S pack. Run `depth_detect.py` for 5+ min, confirm no brownouts, monitor UBEC temp. |
| `[ ]` | **Center-of-gravity re-check** at new AUW (~950 g) | 🟡 | Stack mass distribution shifts CG; may need to slide battery fore/aft. |
| `[ ]` | **Thrust-margin re-test at imaging AUW** | 🟡 | First arm + low-throttle motor test at ~950 g before any hover. T/W approaches 2.5:1. |
| `[ ]` | **Active cooler installation** (already in BOM at £5) | 🟡 | Heat management once stack is enclosed. |
| `[ ]` | **Camera FOV / mounting orientation documented** | 🟢 | Add to `scripts/ar0144/` notes. |

## 3. Pre-flight blockers (per existing project memory)

These were flagged before the perception-pivot and are still open.

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **Outdoor compass re-calibration on grass** | 🔴 | 2026-04-24 indoor cal was yellow. Must redo away from rebar / parked cars. |
| `[ ]` | **Verify `ARMING_CHECK=1`** on the FC | 🔴 | Was toggled to 0 during Stage 2 motor diagnostics; memory says restored but unverified. |
| `[ ]` | **PM02 voltage-sense cable verification** | 🟡 | Multimeter continuity on all 6 JST-GH conductors; wiggle test with FC powered. |
| `[ ]` | **Pre-flight checklist walk-through** per `scripts/flight-controller/readme.md` | 🟡 | End-to-end run before first arm-and-hover. |

## 4. Ground station / telemetry

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **Mission Planner or QGroundControl installed on the laptop** | 🔴 | Telemetry radio is useless without a GCS to receive it. |
| `[ ]` | **Bench-test the 433 MHz SiK link** end-to-end | 🔴 | Air unit on TELEM1, ground unit on laptop USB. Verify MAVLink stream in GCS before any flight. |
| `[ ]` | **Distance / LOS test of the SiK link** | 🟡 | Walk-out test from the launch point. |
| `[ ]` | **Geofencing configured** (`FENCE_*` params) | 🟡 | At minimum a radius fence around launch. ArduPilot returns to launch on breach. |
| `[ ]` | **OSD overlay on FPV verified** | 🟢 | HolyBro Micro OSD V2 is wired; confirm battery / mode / GPS show. |

## 5. SITL extensions (cheap dissertation gap-closures)

These would lift the §6.4 "not measured" outcomes without leaving the simulator.

| | Item | Pri | Notes |
|---|---|---|---|
| `[x]` | **T4 hover-stability run** in SITL | 🟡 | ✅ 2026-05-16. `scripts/autonomy/test_hover_stability.py`. 5 m hover in GUIDED auto-hold (not LOITER — LOITER in headless SITL descends without RC throttle input), 60 s @ 10 Hz, 582 samples. Max horizontal drift 0.038 m, RMS 0.019 m. Max vertical drift 0.099 m, RMS 0.054 m. Both well under the 0.5 m bound. PASS. JSON + PNG in `scripts/sitl/logs/t4_hover_20260516-185701.{json,png}`. |
| `[x]` | **T5 collision avoidance** in SITL | 🟡 | ✅ 2026-05-17. Full Gazebo Harmonic integration: custom `iris_with_lidar` model (forward-facing GPU lidar, 61 samples × 67° HFOV, 5 Hz, 0.2–15 m range) + `iris_obstacle.sdf` world with a 12 m tall red cylinder 12 m N of home. New `scripts/autonomy/gazebo_perception_bridge.py` subscribes to `/iris/forward_lidar` via `gz topic -e --json-output`, converts each scan into a PerceptionFrame (closest-ray depth + horizontal centroid) the autonomy `DepthDetectSource` consumes unchanged. New `scripts/autonomy/test_avoidance_gazebo.py` runs the closed loop. **Result**: min clearance from cylinder surface **1.921 m** (bound 0.3 m, 6.4× headroom), bird flew past the obstacle to 17.04 m N with a 3.45 m east sidestep, clean CRUISING → AVOIDING → CRUISING transitions, RTL + disarm completed. JSON + PNG at `scripts/sitl/logs/t5_gazebo_20260517-030225.{json,png}`. Required a fix to gazebo_perception_bridge.py: `bbox_centroid_norm` must be emitted as a 2-element list `[cx, cy]` in `[-1, +1]` (not a dict in `[0, 1]`) to match the convention in depth_detect.py + perception_source.py — silent KeyError otherwise dropped all detections. |
| `[x]` | **Multi-run T6 mission replication** (10 missions) | 🟢 | ✅ 2026-05-16. `scripts/autonomy/test_mission_replicate.py`. 10 sequential box_50m runs against one SITL instance, mean ± σ extents over all 10: N=51.12±0.07 m (+2.2%), E=51.17±0.06 m (+2.2%), peak alt=10.02±0.00 m (+0.2%), duration 67.9 s/run uniform. 100% pass rate at ±5% tolerance. Required a bridge fix — `upload_mission` now routes `MISSION_REQUEST(_INT)` and `MISSION_ACK` through dedicated reader-fed queues (the original raw `recv_match` raced the reader thread and dropped requests). JSON + PNG at `scripts/sitl/logs/t6_replicate_20260516-203126.{json,png}`. Note: `MISSION_ITEM_REACHED` events arrive patchily under UDP loopback (some runs report 3-6 of 6 items) but flight execution is correct — extents + duration confirm the full box was flown each time. Closes FR5/T6 from "provisionally supported" to "supported (10/10)". |
| `[x]` | **Wind / disturbance rejection** runs | 🟢 | ✅ 2026-05-16. `scripts/autonomy/test_wind_sweep.py` (uses new `Vehicle.set_param` for runtime PARAM_SET — see bridge commit). Sweep `SIM_WIND_SPD ∈ {0, 5, 10, 15}` m/s from west (270°), no turbulence. Each level: 30 s hover + 2 box missions. **T4 hover h-drift**: 36 mm at 0 m/s → 38 mm @ 5 m/s → 43 mm @ 10 m/s → 118 mm @ 15 m/s — flat-then-knee at the strong-wind boundary, all 4× under the 500 mm bound. **T6 mission**: 100 % pass rate (8/8) at ±5 % extent tolerance at every wind level; extents stay 50.85–51.24 m N / 51.14–51.39 m E (target 50 m). JSON + PNG at `scripts/sitl/logs/wind_sweep_20260516-214041.{json,png}`. |

## 6. Hardware flight (Stage 4) — when ready

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **First indoor netted hover** with imaging stack mounted | 🔴 | Closes Stage 4. Pre-reqs: outdoor compass cal, arming check, telemetry link, vibration mount. |
| `[ ]` | **T1 quantitative detection accuracy** on a labelled AutonoBird test set | 🟡 | Currently §6.1 T1 is qualitative. Building a small labelled set + mAP measurement closes NFR2. |
| `[ ]` | **T7 endurance test** under full AI processing load | 🟡 | NFR3 / § 6.4 unmeasured. Needs imaging-stack-mounted airframe + flight time. |
| `[ ]` | **Vibration analysis** post-mount via FC DataFlash logs | 🟡 | Validates the vibration-damped mount; checks for clip events on IMU. |

## 7. Perception extensions

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **YOLO custom-class enrollment** | 🟡 | Two paths: (a) fine-tune YOLOv8n on a small custom dataset, recompile to HEF; (b) embedding-based open-set detection via OSNet/CLIP + FAISS. (b) is the §6.6 "person recognition deployment" item. |
| `[ ]` | **Vector database for face / object embeddings** | 🟡 | FAISS or Qdrant. Stores ArcFace + OSNet embeddings for retrieval. Pi 5 RAM-friendly. |
| `[x]` | **Detection event persistence** (replace stdout with structured logs) | 🟡 | ✅ 2026-05-16. `scripts/perception/depth_detect.py --jsonl PATH` (also `--no-gui` for headless runs) appends one JSON record per processed frame, line-buffered. Schema matches `scripts/autonomy/perception_source.PerceptionFrame`. The autonomy-side `DepthDetectSource` (now real, no longer a stub) tails this file — supports both replay (`tail_from_end=False`) and live tail. Round-trip tested. Real-perception → SITL closed-loop is now a config change in the planner. |
| `[ ]` | **Apply perception backlog optimisation #2 (ROI-only SGBM)** | 🟢 | Auto-memory `project_perception_optimizations_backlog.md` #2. Only triggers if current 138 ms loop becomes insufficient. |
| `[ ]` | **Apply optimisation #3 (YOLO-conditional depth)** | 🟢 | Same memory file. Couples perception stages instead of running both blindly. |

## 8. Interaction modalities

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **Wire Jarvis into a flight command map** | 🟡 | Voice intents → MAVLink commands. ("Jarvis, return home" → `NAV_RTL`.) |
| `[ ]` | **Gesture control** | 🟡 | YOLO-pose detection on the AR0144 left feed → gesture vocabulary → command map. Weight saving vs USB mic + speaker (~25 g). |
| `[ ]` | **Speaker / mic weight audit** | 🟢 | Measure actual weight of existing USB peripherals; decide if gesture replaces voice for the airborne use case. |

## 9. Status / observability

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **Pico LEDs driven by drone state machine** | 🟡 | Currently REPL-only. Need bridge: state machine event → serial command → LED colour. |
| `[ ]` | **Unified logging strategy** | 🟢 | FC DataFlash `.BIN`, MAVProxy `.tlog`, Jarvis logs, perception stdout — currently scattered. Conventional layout under `scripts/<subsystem>/logs/`. |
| `[ ]` | **Flight-data archival convention** | 🟢 | `.tlog` + `.BIN` retained per flight, named by date + mission. |

## 10. Documentation / hand-off

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **Top-level deployment runbook** — fresh Pi to first flight | 🟢 | Each subsystem has its own `setup.sh`; no master guide. |
| `[ ]` | **Operational guide** for someone other than the author | 🟢 | How to launch, arm, fly, RTL, recover. |
| `[ ]` | **Demo footage** of the perception loop and SITL mission | 🟢 | Already flagged as deferred. |

---

## Notes

- The dissertation in `resources/Individual_Project_V5.docx` reflects the project as-of 2026-05-16. Do not edit it. Any updates after the deadline go into a new `_V6.docx` if regenerated.
- Auto-memory `project_current_status.md` mirrors the cross-cutting state from this backlog and the dissertation. Update both when something material changes.
- Many items have inter-dependencies — the MAVLink bridge (§1) is the single largest unblocker. T5, path planning, the orchestrator, gesture control, and Jarvis-to-flight all depend on it.
