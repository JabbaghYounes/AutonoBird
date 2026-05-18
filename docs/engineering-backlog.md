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
| `[x]` | **Unified orchestrator / management entry-point** | 🟡 | ✅ 2026-05-17. `scripts/autonomy/orchestrator.py`. Single-process bring-up of Vehicle bridge → FSM → optional Planner with chosen perception source → optional LED bridge to the Pico. Defaults to **monitor mode** (no autonomous commands) so it's safe to run anywhere; opt into the reactive planner with `--enable-planner`. LED integration is best-effort — `--led-port /dev/ttyACM1` mirrors FSM state changes when a Pico is present, gracefully degrades to a `_NullLed` no-op when it isn't (currently the case — see §9 below). Periodic 1 Hz compact STATUS line plus event-driven FSM / planner transition logs. Clean SIGTERM / Ctrl-C shutdown. Imports the Pico bridge module via cross-subsystem sys.path injection, same pattern as the autonomy → flight-controller imports. Voice → MAVLink and gesture → MAVLink slot into this as additional event-driven subscribers (next item in §8). |
| `[ ]` | **Planner clear-detection debouncing** | 🟢 | Add N-consecutive-clear-frames vote in `planner.py:_tick()` before the AVOIDING → CRUISING transition. Airframe walkaround on 2026-05-18 logged **161 planner transitions across 3:34 min** (80 each direction). Depth-band analysis of the JSONL shows only 5.1 % of detection-present frames live in the 1.5–2.0 m hysteresis band; the flapping is driven by **75 detection-presence flips** (frame-to-frame None ↔ obstacle from confidence dips, occlusions, frame-edge dropouts), not depth oscillation. An N=3 vote on the None branch would cut transitions ~10× without affecting threat-entry latency. Source logs: `scripts/sitl/logs/walk_20260518-1755/`. |
| `[x]` | **Bridge: request extended MAVLink streams on connect** | 🟢 | ✅ 2026-05-18, resolved FC-side instead of bridge-side. Set `SR0_*` USB-MAVLink stream rates directly on the FC (durable in EEPROM, applies to every USB-MAVLink client): `SR0_EXT_STAT=2`, `SR0_EXTRA1/2=4`, `SR0_EXTRA3=2`, `SR0_POSITION=2`, `SR0_RC_CHAN=2`, `SR0_RAW_SENS=1`. Next bridge run will see `alt / gs / sats / batt` populated in the STATUS line — no bridge code patch needed. Verified via `param show SR0_*` over the SiK telemetry link. |

## 2. Hardware integration — imaging stack on the airframe

Bench rig stays handheld today. Until the stack is mounted on the drone, nothing in this section is exercised in flight conditions.

| | Item | Pri | Notes |
|---|---|---|---|
| `[x]` | **Mount AI HAT+ + AR0144 stereo camera on the airframe** | 🔴 | ✅ 2026-05-17. Pi 5 + AI HAT+ + AR0144 mounted on the QUAV250 top deck. UBEC USB-C reaches the Pi at the new mount point. AUW with battery: 945 g (matches the dissertation's ~950 g estimate). Forward-facing geometry confirmed by the 2026-05-18 walkaround perception data — detections fired across the room as expected. |
| `[x]` | **Vibration-damped camera mount** | 🔴 | ✅ 2026-05-17. Rubber grommets + plastic screws between the camera bracket and the airframe carbon. Plastic screws avoid bridging the damping layer with metal hardpoints. Vibration impact on SGBM will be measured properly once motors spin (still open as § 6 vibration analysis), but the walkaround confirmed clean depth values at rest (median 1.57 m person depth, σ = 0.88 m across handheld motion). |
| `[x]` | **LiPo-powered bench test of full compute stack** | 🟡 | ✅ 2026-05-18. UBEC → Pi 5 → AI HAT+ → AR0144 → FC all running off the 4S pack. First bench run (`depth_detect.py` for 5 min): `dmesg` clean of under-voltage / throttling, AR0144 enumerated stable at USB 480M, perception loop hit the dissertation baseline (138.1 ms / 18.7 ms NPU / 75.8 ms SGBM / 6.76 Hz). Second run during the airframe walkaround (3:34 min, 1449 frames): zero under-voltage events, FC and Pi share the 4S pack cleanly, MAVLink link stable for the full session. Logs at `scripts/sitl/logs/walk_20260518-1755/`. |
| `[ ]` | **Center-of-gravity re-check** at new AUW (~950 g) | 🟡 | Stack mass distribution shifts CG; may need to slide battery fore/aft. |
| `[ ]` | **Thrust-margin re-test at imaging AUW** | 🟡 | First arm + low-throttle motor test at ~950 g before any hover. T/W approaches 2.5:1. |
| `[ ]` | **Active cooler installation** (already in BOM at £5) | 🟡 | Heat management once stack is enclosed. |
| `[ ]` | **Camera FOV / mounting orientation documented** | 🟢 | Add to `scripts/ar0144/` notes. |

## 3. Pre-flight blockers (per existing project memory)

These were flagged before the perception-pivot and are still open.

| | Item | Pri | Notes |
|---|---|---|---|
| `[ ]` | **Outdoor compass re-calibration on grass** | 🔴 | 2026-04-24 indoor cal was yellow. Must redo away from rebar / parked cars. |
| `[x]` | **Verify `ARMING_CHECK=1`** on the FC | 🔴 | ✅ 2026-05-18, verified via SiK telemetry. Was set to bitmask `1047678` (all checks ENABLED except bits 7 Board Voltage and 8 Battery Level — leftover from Stage 2 motor diagnostics). Restored to `1` ("ALL"), re-engaging battery + voltage checks. Verified via `param show ARMING_CHECK`. Now correctly emits `Battery 1 unhealthy` PreArm under USB-only power. |
| `[~]` | **PM02 voltage-sense cable verification** | 🟡 | Partial 2026-05-18: SiK telemetry from FC shows `voltage_battery = 4 mV` and `BATTERY_STATUS voltages : [4, 65535, ...]` under USB-only power — confirms the PM02 sense path is **electrically alive** (correctly reports "no battery"). Full verification (continuity + wiggle test + LiPo-loaded voltage reading) still pending. |
| `[ ]` | **Pre-flight checklist walk-through** per `scripts/flight-controller/readme.md` | 🟡 | End-to-end run before first arm-and-hover. |

## 4. Ground station / telemetry

| | Item | Pri | Notes |
|---|---|---|---|
| `[x]` | **Mission Planner or QGroundControl installed on the laptop** | 🔴 | ✅ pre-existing. QGroundControl was already installed on the dev workstation; confirmed 2026-05-18. |
| `[x]` | **Bench-test the 433 MHz SiK link** end-to-end | 🔴 | ✅ 2026-05-18. Air unit on TELEM1 (SERIAL1 already MAVLink2 @ 57600, no param change needed); ground unit on workstation USB after a reboot cleared an xHCI host-controller hang (`/dev/ttyUSB0`, FTDI FT231X 0403:6015). MAVProxy connected: heartbeat received, full 1051 parameters synced via FTP, **RSSI 219 / remRSSI 217 bidirectional** (strong both directions). Full bench config session done over the SiK link: ARMING_CHECK + FENCE_* + SR0_* writes verified through the radio. |
| `[~]` | **Distance / LOS test of the SiK link** | 🟡 | Indoor through-wall walk 2026-05-18: 239 samples over ~4 min at distances 0/5/10/15/20/25 m, 0 rxerrors and 0 fixed-errors throughout, RSSI 219 baseline -> 145 at 25 m through multiple walls, well above the ~50 weak-link threshold. Bidirectional symmetry holds (rssi vs remrssi within 5-10 points). This is 2.5x the operator-to-drone distance the indoor netted hover requires. **Indoor / through-wall test PASS.** Tool: `scripts/flight-controller/test_sik_link.py`; CSV: `scripts/sitl/logs/sik_los_20260518-2345.csv`. **Outdoor LOS range characterisation still pending** for future outdoor-flight scenarios. |
| `[x]` | **Geofencing configured** (`FENCE_*` params) | 🟡 | ✅ 2026-05-18. Geometry was pre-set but `FENCE_ENABLE` was `0` (dormant). Enabled the fence and tightened the radius for indoor netted hover testing: `FENCE_ENABLE=1`, `FENCE_TYPE=7` (alt_max + circle + alt_min all active), `FENCE_RADIUS=20m` (was 300m), `FENCE_ALT_MAX=100m`, `FENCE_ALT_MIN=-10m`, `FENCE_MARGIN=2m`, `FENCE_ACTION=1` (RTL on breach), `FENCE_AUTOENABLE=0` (manual enable per flight). For outdoor open-field testing, bump `FENCE_RADIUS` back up via `param set` before arming. |
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
| `[ ]` | **Wire Jarvis into a flight command map** | 🟢 | Voice intents → MAVLink commands. **Deprioritised 2026-05-17**: acoustic SNR is hostile on an aerial drone (motor noise ~75-85 dB at the on-board mic, operator's voice attenuated 6 dB / 2× distance, mic is on wrong end of the link) — gestures are the right primary in-flight modality. Voice stays useful for pre/post-flight bench commands ("Jarvis, run preflight check"), so the wiring is still worth building, just not on the critical path. |
| `[~]` | **Gesture control** | 🟡 | **Side-tabled 2026-05-18, code complete on both sides; live integration deferred to post-submission.** Autonomy-side: `scripts/autonomy/{gesture_classifier,gesture_action_map}.py` recognise four safety-critical gestures (STOP/LAND/COME/RECEDE) from COCO-17 body keypoints with per-keypoint confidence gating + 3-frame temporal smoothing; dispatch to `Orchestrator.command_hold/land/resume/rtl` with cooldown. `Detection.keypoints` is a new optional field on the data contract (backward-compatible). Orchestrator gains `--enable-gestures`. `test_gesture_pipeline.py` validates all 5 paths end-to-end with synthetic keypoints (no Hailo / camera needed) — all PASS on both dev workstation + Pi (aarch64). Pi-side: `scripts/perception/depth_detect.py --pose` swaps to `models/v8_pose_n_hailo8.hef`, runs `decode_pose_detections()` (defensive multi-format decoder for Hailo Model Zoo's `yolov8_pose_postprocess` output layouts), unmaps keypoints to original-frame pixels via `unmap_keypoints()`, and emits a `keypoints` list in the JSONL (each entry `[x_norm, y_norm, conf]` in [-1, +1] frame-centred coords). Live cv2 overlay draws COCO skeleton + per-keypoint dots when confidence ≥ 0.3. **Untested against live Hailo** — first Pi run is the validation; the pose decoder may need tensor-shape iteration when the actual HEF output layout doesn't match the assumed layout. Picked up post-dissertation-submission. |
| `[ ]` | **Speaker / mic weight audit** | 🟢 | Now mostly informational since gesture is the primary in-flight modality. Speaker still useful for audible status alerts on-bench. |

## 9. Status / observability

| | Item | Pri | Notes |
|---|---|---|---|
| `[~]` | **Pico LEDs driven by drone state machine** | 🟡 | Code complete 2026-05-17; **hardware blocked**. Pi-side bridge (`scripts/pico-led/led_bridge.py`) + Pico 2 W MicroPython firmware (`scripts/pico-led/main.py`) + line-based ASCII protocol over USB-serial + nine FSM state → colour/animation mappings. Standalone CLI (`--test`, `--ping`, `--state NAME`, `--watch-fsm`) and `LedBridge` class imported by the orchestrator. Pi-side imports verified clean against `pyserial`. **Blocker**: existing Pico 2 W unit fails to enumerate on USB across 5 cables and BOOTSEL mode — board appears dead between sessions. When a replacement Pico is in hand, flash `main.py` via `mpremote connect /dev/ttyACM0 fs cp main.py :main.py` and the rest of the workflow just works. Until then, the orchestrator's terminal status output serves the same operator-feedback role. |
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
