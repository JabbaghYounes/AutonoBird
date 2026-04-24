# Flight Controller

Covers the Pixhawk 6C Mini installation on the QAV250 airframe: port assignments, wiring decisions, component placement, weight breakdown, ArduPilot parameters, and the pre-flight checklist.

There is no code in this subsystem yet. Future MAVLink bridge / companion-computer scripts (ROS2, MAVSDK, obstacle-avoidance → Pixhawk command loop) will live here.

## Airframe

- **Frame**: HolyBro QUAV250 carbon fibre kit
- **Flight controller**: Pixhawk 6C Mini (ships with the kit) running ArduPilot
- **Motors + ESCs + PDB**: stock HolyBro kit
- **Companion computer**: Raspberry Pi 5 + HAILO AI HAT+ (26 TOPS, Hailo-8) — HAT mounted separately during software development
- **Stereo camera**: AR0144 USB global-shutter (2560x720 side-by-side) — mounted separately during calibration
- **GPS**: HolyBro M10 (integrated compass)
- **RC link**: ExpressLRS — RadioMaster RP4TD receiver + RadioMaster Pocket transmitter
- **Video**: FPV camera → HolyBro Micro OSD V2 → VTX (1W)
- **Telemetry**: HolyBro SiK 433 MHz 100 mW radio pair
- **Power**: CNHL 1500 mAh 130C 4S LiPo → PDB → ZTW UBEC 8A G2 (5.0V) → USB-C → Pi
- **3D-printed additions**: landing legs, RP4TD antenna mounts, GPS mast, Pi mounting plate (elevated above carbon top plate on spacers)

## Port assignments

| Port   | Device                                  | ArduPilot serial | Notes |
|--------|-----------------------------------------|------------------|-------|
| TELEM1 | SiK 433 MHz 100 mW telemetry radio      | SERIAL1          | Default MAVLink2 @ 57600, no config change needed |
| TELEM2 | HolyBro Micro OSD V2 (serial side)      | SERIAL2          | `SERIAL2_PROTOCOL=2`, `SERIAL2_BAUD=57` for MAVLink OSD |
| GPS1   | HolyBro M10 GPS                         | SERIAL3          | GPS default, no config change |
| GPS2   | RadioMaster RP4TD ELRS receiver (CRSF)  | SERIAL4 / UART8  | `SERIAL4_PROTOCOL=23` (RCIN), `SERIAL4_BAUD=420000` (ELRS default; initially 460800, corrected after reading RP4TD UART config). `RSSI_TYPE=5` for CRSF RSSI |
| RC IN (PPM/SBUS) | **unused** | —              | The 6C Mini's RC IN port supports PPM/SBUS only, not CRSF — so an ELRS receiver cannot be connected here |

## Key wiring decisions

### ELRS receiver on GPS2 (not RC IN)

The Pixhawk 6C Mini's RC IN port only supports PPM and SBUS. The RP4TD ELRS receiver speaks CRSF, a bidirectional serial protocol requiring a full UART with TX and RX. GPS2 is a full UART that can be repurposed as RCIN in firmware, so the ELRS receiver lives there.

### RP4TD solder mapping

A 6-pin JST-GH GPS2 cable was sacrificed for the ELRS receiver. Only 4 of the 6 wires are used:

| GPS2 cable pin | Function          | RP4TD pad |
|----------------|-------------------|-----------|
| 1 (red)        | +5V               | **+**     |
| 2              | UART8_TX (Pixhawk out) → receiver RX | **RX** |
| 3              | UART8_RX (Pixhawk in) ← receiver TX  | **TX** |
| 4              | I2C2_SCL          | unused — cut and insulated |
| 5              | I2C2_SDA          | unused — cut and insulated |
| 6              | GND               | **−**     |

TX/RX crossover at the receiver matters: Pixhawk TX → receiver RX pad, Pixhawk RX → receiver TX pad. The `R` and `T` pads on the RP4TD are a secondary telemetry link and are intentionally left unconnected.

### UBEC wiring

ZTW UBEC 8A G2 with jumper set to 5.0V (factory default). Input red/black soldered directly to the PDB BAT+/BAT− pads (raw 4S voltage, not the regulated 5V rail). Output spliced to a USB-A → USB-C cable: USB-A end cut off, USB-C end retained for the Pi. Red → VBUS, black → GND, data wires (white/green) isolated and tucked.

### OSD on TELEM2 instead of TELEM1

TELEM1 is conventionally reserved for the telemetry radio. The OSD was moved to TELEM2 so TELEM1 is free for the SiK 433 MHz air unit, which defaults to MAVLink2 at 57600 with no configuration.

## Component placement

Back-to-front on the drone (nose forward):

| Location                | Device                                    |
|-------------------------|-------------------------------------------|
| Bottom plate, rear       | ELRS receiver (zip-tied), T-dipoles in printed vertical mounts clipped to rear pillars, V-shape, free air |
| Top carbon plate, rear   | SiK telemetry radio (taped), antenna up-left, clear of props |
| Top of Pixhawk           | OSD module (heat-shrunk, taped) |
| Top of bottom plate       | VTX (zip-tied), open airflow above and below |
| Top carbon plate, rear   | GPS M10 on printed mast, arrow forward |
| Pixhawk                 | Mounted forward on anti-vibration pad, arrow forward |
| Between top plates, front| UBEC (zip-tied), >5 cm from both radios |
| Top printed plate, front | Raspberry Pi 5 (ports facing **forward** — see below) |
| Top carbon plate, front  | FPV camera |
| Bottom plate, middle     | 4S LiPo (velcro strap), power leads routed around |

### Pi port orientation — forward

The Pi 5's ports face forward rather than rearward. The two middle pillar screws and their captive nuts protrude upward from the plate, physically blocking rear-port orientation. Flipping the screws doesn't help because the nuts also protrude, and lower-profile hardware wasn't available. The trade-off is accepted: peripheral cables (USB, HDMI, Ethernet if needed) exit forward alongside the FPV camera.

### Dual-plate top stack

The printed Pi mounting plate is elevated above the carbon top plate on spacers at all 6 pillars. Both plates are retained together. The elevation was originally needed because the FPV camera's mounting screws added height the printed plate wasn't designed for. Keeping both plates provides additional rigidity and extra mounting points.

### RP4TD antenna mounts

Printed vertical antenna mounts (RP4TD Folding TPU Antenna Holders, designer-recommended TPU, printed here in PLA due to material availability). Mount holes widened with a lighter and the click-in pegs heat-softened to snap over the QAV250 pillars. Each T-dipole is held clear of the carbon frame in free air, oriented in a V-shape.

## All-up weight

- **850 g** with the CNHL 1500 mAh 4S battery, **without** the AI HAT or stereo camera
- **~950 g** projected once the AI HAT (~35 g) and AR0144 stereo camera + mount + cable (~60–80 g) are added
- Thrust-to-weight ratio at current AUW: approximately 2.8:1 (estimated; confirm via hover throttle on first flight)
- Flight time estimate: 3–5 min per 1500 mAh pack at current AUW
- Prop guards (installed) contribute roughly 30–50 g and will be removed for outdoor flight to reclaim thrust margin

## ArduPilot parameters

The Pixhawk firmware is ArduCopter 4.6.3 (ChibiOS, pixhawk6C non-bdshot target). Parameters below are the full set configured during Stage 2 bench validation.

**Frame + orientation:**
```
FRAME_CLASS      = 1       # Quad
FRAME_TYPE       = 1       # X
AHRS_ORIENTATION = 0       # Forward (Pixhawk arrow forward)
```

**Serial ports** (mapping: SERIAL1=TELEM1, SERIAL2=TELEM2, SERIAL3=GPS1, SERIAL4=GPS2):
```
# SiK telemetry on TELEM1 — defaults correct (MAVLink2 @ 57600), no change

# OSD on TELEM2
SERIAL2_PROTOCOL = 2       # MAVLink2
SERIAL2_BAUD     = 57      # 57600 baud

# M10 GPS on GPS1 — defaults correct

# ExpressLRS RP4TD on GPS2 (repurposed as RCIN)
SERIAL4_PROTOCOL = 23      # RCIN
SERIAL4_BAUD     = 420     # 420000 baud — ELRS receiver default
RSSI_TYPE        = 5       # CRSF
```

**Battery monitor (HolyBro PM on POWER1):**
```
BATT_MONITOR     = 4       # Analog Voltage and Current (requires reboot for PIN params to appear)
BATT_VOLT_PIN    = 10
BATT_CURR_PIN    = 11
BATT_VOLT_MULT   = 18.182
BATT_AMP_PERVLT  = 36.364
BATT_CAPACITY    = 1500    # mAh (CNHL 4S 1500)
```

**Battery failsafe:**
```
BATT_LOW_VOLT    = 14.0    # 4 × 3.50 V
BATT_CRT_VOLT    = 13.2    # 4 × 3.30 V
BATT_LOW_TIMER   = 10      # s
BATT_FS_LOW_ACT  = 1       # Land
BATT_FS_CRT_ACT  = 1       # Land
```

**GCS failsafe** (disabled for bench work; re-enable before flight):
```
FS_GCS_ENABLE    = 0
```

**Compass (external IST8310 on M10 GPS, internal IST8310 disabled):**
```
COMPASS_USE       = 1         # Use external compass
COMPASS_USE2      = 0         # Disable internal (sits next to FC current traces, bad cal environment)
COMPASS_USE3      = 0         # Third slot empty
COMPASS_PRIO1_ID  = 658953    # DEV_ID of the external M10 IST8310
COMPASS_ORIENT    = 4         # YAW_270 (auto-detected during on-board cal)
COMPASS_OFS_X     = -2        # Offsets from 2026-04-24 indoor cal; magnitude ~40 mGauss (clean)
COMPASS_OFS_Y     = 20
COMPASS_OFS_Z     = -35
```

> **Outdoor re-cal planned before first flight.** The 2026-04-24 cal was done indoors and came back yellow on the quality indicator. The offset magnitudes themselves are clean; the yellow reflects sample distribution of a quick indoor spin. An outdoor re-cal on grass, away from rebar and cars, will likely land it green.

**ESC / motor output — PWM mode (see history below):**
```
MOT_PWM_TYPE     = 0       # Normal PWM 400 Hz (changed from 6 / DShot600 after BLHeli handshake lockout)
SERVO_BLH_AUTO   = 0       # Disabled — BLHeli passthrough no longer in use
SERVO_BLH_RVMASK = 0       # Disabled — motor direction corrected physically instead
```

> **Why PWM and not DShot600.** During Stage 2 bench bring-up on 2026-04-24, setting `SERVO_BLH_RVMASK=15` + `SERVO_BLH_AUTO=1` initiated a DShot direction-reverse handshake with each ESC. The handshake did not complete cleanly (suspected EMI from a momentarily flaky PM02 voltage-sense cable). After reverting `RVMASK=0` / `AUTO=0`, the four ESCs remained stuck waiting for handshake completion and refused all throttle commands in DShot600 mode — silent motors, BLHeli_S "no valid signal" beep pattern (one long + three short, repeating). Switching `MOT_PWM_TYPE` 6 → 0 after a full LiPo power-cycle let the ESCs re-autodetect the signal as PWM and resume normal throttle response. DShot restoration requires a classic BLHeliSuite (Wine) factory reset of all four ESCs; deferred until after first flight. PWM at 400 Hz is flight-adequate for the dissertation envelope — the only losses are DShot telemetry and a small latency margin, no flight-critical functionality.

> **Motor direction: physical phase swap, not RVMASK.** All four motors shipped factory-defaulted to the opposite rotation of ArduCopter's X-quad expectation. Direction was corrected by swapping any 2 of 3 phase wires at the motor bullet connectors on each motor, with the LiPo fully disconnected. After the swap, all four motors spin correctly: A (FR) CCW, B (BL) CCW, C (FL) CW, D (BR) CW. Motor mapping itself (A=FR, B=BL, C=FL, D=BR) is correct — only rotation was inverted.

**Safety switch:**
```
BRD_SAFETY_DEFLT = 0       # Boot with safety disabled — skip the press-and-hold on the M10 safety button for bench work
```

**Flight modes (SB 3-pos switch → CH6):**
```
FLTMODE_CH       = 6       # Mode channel (SB 3-pos mapped to CH6 on Pocket)
FLTMODE1         = 0       # Stabilize  (SB up,    PWM ≤ 1230)
FLTMODE4         = 2       # AltHold    (SB center, PWM 1491–1620 — centered 3-pos outputs ~1500 µs)
FLTMODE6         = 6       # RTL        (SB down,  PWM ≥ 1750)
# FLTMODE2/3/5 stay default — unreachable with this switch
```

> **FLTMODE3 vs FLTMODE4 gotcha.** ArduPilot maps flight-mode channel PWM to mode slots as: FLTMODE1 ≤1230, FLTMODE2 1231–1360, **FLTMODE3 1361–1490**, **FLTMODE4 1491–1620**, FLTMODE5 1621–1749, FLTMODE6 ≥1750. A centered 3-pos switch outputs ~1500 µs, which falls in FLTMODE4, not FLTMODE3. Always put the middle-position mode in `FLTMODE4` when using a single 3-pos switch.

**RC / throttle failsafe:**
```
FS_THR_ENABLE    = 1       # RTL on throttle failsafe (receiver enters failsafe when TX signal lost)
FS_THR_VALUE     = 975     # Trigger PWM threshold on CH3 — below this = failsafe
```

RP4TD failsafe position is not explicitly configured on the Pocket side; the RX defaults to "no pulses" when TX signal is lost, which causes ArduPilot to see CH3 drop below `FS_THR_VALUE` and trigger `FS_THR`. This is the standard ELRS + ArduPilot config — do not configure failsafe positions in the RX unless you have a specific reason.

## Stage 2 bench validation progress (closed 2026-04-24)

Hardware build complete 2026-04-22. Stage 2 firmware bring-up closed 2026-04-24:

- [x] ArduCopter 4.6.3 flashed (2026-04-23)
- [x] Parameter set above loaded (2026-04-23; audited and re-verified persistent on 2026-04-24 after several defaults had drifted back — likely from the 2026-04-23 session's edits not fully flushing to flash before power-down)
- [x] Accelerometer calibration (6-position, 2026-04-23)
- [x] Compass calibration (external M10 primary; internal disabled. Recalibrated 2026-04-24 after offsets were lost between sessions — see note in Compass params above)
- [x] Battery + GCS failsafes
- [x] Motor direction — corrected via physical phase-wire swap on all four motors 2026-04-24 (LiPo disconnected, any 2 of 3 phase bullets swapped per motor). Verified in PWM motor test: A (FR) CCW, B (BL) CCW, C (FL) CW, D (BR) CW.
- [x] End-to-end motor test validated at `BATT_MONITOR=4` with no bypass — all six core Stage 2 params persistent through reboot, motors spin cleanly in PWM mode.
- [x] ELRS bind — RP4TD + Pocket both on ExpressLRS 3.3.1 (version mismatch theory was wrong). Real blocker was that the Pocket's factory "pocket" model had Internal RF off. Fix: copied the factory pocket model (which ships with CH1–CH8 pre-wired correctly to sticks + SA/SB/SC/right-trigger), set Internal RF = CRSF 5.25M / 500 Hz / channels 1-16, 3x LiPo power-cycle on the drone to put RP4TD into bind mode, pressed Bind in the Pocket's ELRS Lua script. Link comes up solid blue on the RP4TD within 1 s. Binding phrases blank on both sides (factory).
- [x] Radio calibration in QGC (sticks + switches mapped, channel monitor all live)
- [x] RC / throttle failsafe (`FS_THR_ENABLE=1`, `FS_THR_VALUE=975`)
- [x] Flight mode assignments (CH6 / SB 3-pos → Stabilize / AltHold / RTL via `FLTMODE1` / `FLTMODE4` / `FLTMODE6`)
- [x] PM02 voltage-sense cable visual inspection — 6-pin JST-GH latched firmly at both ends, no bent pins or damaged insulation. Cleared for flight.
- [ ] Video feed (VTX + OSD end-to-end via Eachine goggles) — defer to Day 3 pre-flight
- [ ] Arm test — defer to Day 3, outdoor, after re-cal
- [ ] Outdoor compass re-cal on grass, away from rebar/cars — bench cal was indoor-yellow, outdoor expected green
- [ ] DShot600 restoration via classic BLHeliSuite (Wine) factory reset of all four ESCs — deferred, not blocking first flight

### Parameter backups saved during Stage 2 (in `resources/`, gitignored)

- `autonobird-stage2-<date>.params` — early-session audit snapshot (pre-BLHeli-revert)
- `autonobird-stage2-post-blheli-revert.params` — after `SERVO_BLH_RVMASK=0` / `SERVO_BLH_AUTO=0`
- `autonobird-stage2-motors-fixed.params` — PWM mode, motors verified correct direction
- `autonobird-stage2-elrs-ready.params` — **current golden baseline**, adds ELRS-bound RC config, CH6 flight-mode selector, and throttle failsafe on top of the motors-fixed snapshot

## ESC firmware note

The HolyBro QUAV250 kit ships with **BLHeli_S 20A 4-in-1** ESCs (SiLabs BB21, A_H_30 target, firmware 16.7), **not** BLHeli_32. This matters:

- `BLHeliSuite32` cannot talk to these ESCs — it will show `No ESC found` even when ArduPilot passthrough negotiates successfully.
- For direct ESC configuration, use classic `BLHeliSuite` (Wine on Linux) or the MAVProxy `blheli` module.
- ArduPilot's `SERVO_BLH_RVMASK` DShot direction command **failed on this build** (Stage 2 bench, 2026-04-24): the reverse handshake did not complete cleanly and left all four ESCs stuck waiting for handshake completion, refusing all throttle commands in DShot600 mode. Suspected EMI contamination during the handshake from a flaky PM02 voltage-sense cable, but not root-caused. Recovery required reverting the BLHeli params *and* switching `MOT_PWM_TYPE` to `0` (Normal PWM 400 Hz) to let the ESCs re-autodetect the signal protocol. Motor direction was then corrected by **physical phase-wire swap** at the motor bullet connectors instead.
- Recommendation for this build: prefer the physical phase-swap for direction reversal — it is deterministic and does not depend on a fragile DShot handshake. `SERVO_BLH_RVMASK` may still be safe on a future build with cleaner power-sense wiring, but validate on the bench before committing.

## ExpressLRS binding (as actually performed 2026-04-24)

Both ends shipped on ExpressLRS 3.3.1 with blank binding phrases. The bind procedure that worked:

1. **Confirm versions match.** On the Pocket: System → Tools → ExpressLRS → Version String (should read 3.3.1). RP4TD firmware 3.3.1 is the factory build. If versions differ, flash the older side via ExpressLRS Configurator.
2. **Confirm the Pocket's active model has Internal RF = CRSF.** The factory "pocket" model has all stick/switch mixes pre-wired on CH1–CH8 but ships with Internal RF OFF. Either modify the active model to set Internal RF = CRSF (5.25M / 500 Hz / channels 1-16), or copy the factory pocket model to a new slot (e.g. `qav250`) and enable CRSF there. The Pocket main screen will show an RF/antenna indicator when transmitting.
3. **Put the RP4TD into bind mode.** With the Pocket powered on, do a rapid 3x power-cycle on the drone: plug LiPo → unplug → plug → unplug → plug, each cycle under ~1 s, leaving it powered on the 3rd plug. The RP4TD LED should switch from slow yellow flash (searching) to fast double-blink (bind mode). If it stays slow-flashing, the power-cycle timing was not fast enough — try again.
4. **Trigger bind from the Pocket.** System → Tools → ExpressLRS → Bind. The RP4TD should go from fast double-blink to solid blue within ~1 s, and the Pocket's Lua screen starts showing RSSI/LQ telemetry.
5. **Verify in QGC.** Vehicle Setup → Radio → channel monitor should show CH1–CH4 responding to sticks and CH5–CH8 responding to switches. Red dot on the Radio tab goes green. Run Calibrate to set `RC*_MIN` / `RC*_MAX` endpoints.

### Pocket switch → channel map (factory model)

The factory `pocket` model wires its limited physical controls to CH5–CH8:

| Pocket control      | EdgeTX source | Output channel | Positions         |
|---------------------|---------------|----------------|-------------------|
| Left trigger        | SA            | CH5            | 2-pos (-100 / +100) |
| Top 3-pos switch    | SB            | **CH6**        | 3-pos (-100 / 0 / +100) — **flight-mode selector** |
| Other 3-pos switch  | SC            | CH7            | 3-pos             |
| Right trigger       | (2-pos)       | CH8            | 2-pos             |

The Pocket has no TX16S-style toggle switches; all mode selection has to come from the two 3-pos switches. Either is usable as the flight-mode channel — this build uses SB/CH6.

### Model Match gotcha

If `Model Match` is ON in the Pocket's ELRS Lua script, each EdgeTX model needs its own bind; switching models breaks the link until re-bound. For a single-vehicle build, leave Model Match OFF so the RX links to any active model with CRSF enabled.

## Pre-flight checklist

Run through this before any powered bench test or first flight.

**Before connecting battery:**
- [ ] Smoke stopper inline between battery and drone
- [ ] All avionics cables seated, no pinched wires between plates
- [ ] Propellers removed (for bench tests) or correctly installed with matched CW/CCW rotation (for flight)
- [ ] Battery strap seated, battery orientation correct (heavy end forward recommended for CG)
- [ ] UBEC jumper confirmed at 5.0V

**On power up:**
- [ ] Smoke stopper shows green LED (no short)
- [ ] Pixhawk startup tune plays
- [ ] Pixhawk IO + FMU power LEDs: solid green; ACT: blinking (blue indoors = no GPS fix, green = fix acquired)
- [ ] UBEC: solid red LED (output in normal range)
- [ ] Pi: orange power LED + blinking green activity LED
- [ ] GPS M10: red blinking (searching) — should go solid blue outdoors with clear sky
- [ ] ELRS receiver: solid after binding; blinking yellow means not yet bound
- [ ] SiK telemetry radio: starts alternating red/green, goes solid green when ground end is linked
- [ ] OSD: solid green + flickering yellow (processing video)
- [ ] VTX: solid green + blue (transmitting)

**In Mission Planner:**
- [ ] MAVLink link established via ground SiK radio
- [ ] All sensors calibrated: accelerometer (6-position), compass, radio, ESC
- [ ] Flight modes assigned to transmitter switches
- [ ] Failsafes configured (throttle failsafe, GCS failsafe, battery failsafe)
- [ ] Geofence set if flying outdoors
- [ ] ARM check passes with no errors

**On the field before arming:**
- [ ] 3D GPS fix (HDOP < 2.0)
- [ ] Compass consistent (no `Compass inconsistent` warning)
- [ ] Battery voltage ≥ 16.0V (fully charged 4S ≈ 16.8V)
- [ ] Vibration levels acceptable (check after a short motor spin)
- [ ] Props torqued to spec, rotation direction verified

## Photos

See `resources/drone-pics/Drone/` for current build photos (2026-04-22).

## See also

- `scripts/ar0144/steps.md` — stereo camera calibration
- `scripts/ar0144/pose-list.md` — 40-pose calibration plan
- `scripts/jarvis/context.md` — voice assistant integration
