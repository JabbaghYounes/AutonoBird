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
| GPS2   | RadioMaster RP4TD ELRS receiver (CRSF)  | SERIAL4 / UART8  | `SERIAL4_PROTOCOL=23` (RCIN), `SERIAL4_BAUD=460800`. `RSSI_TYPE=5` for CRSF RSSI |
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

The Pixhawk firmware is ArduPilot (not PX4). Parameters to set via Mission Planner (or QGroundControl) before first flight:

```
# ELRS receiver on GPS2 (SERIAL4 / UART8) — CRSF protocol
SERIAL4_PROTOCOL = 23
SERIAL4_BAUD     = 460800
RSSI_TYPE        = 5

# OSD on TELEM2 — MAVLink for telemetry overlay
SERIAL2_PROTOCOL = 2
SERIAL2_BAUD     = 57

# TELEM1 (SiK telemetry) defaults are correct — no change needed
```

Verify `AHRS_ORIENTATION = 0` (arrow forward) — Pixhawk was mounted arrow-forward, so default is correct.

## ExpressLRS binding

The RP4TD receiver must be bound to the RadioMaster Pocket transmitter before RC input works:

1. Flash both ends to a matching ExpressLRS firmware version using the ExpressLRS Configurator.
2. Set a matching binding phrase on both ends.
3. Power-cycle both; receiver LED should go from blinking yellow to solid.

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
