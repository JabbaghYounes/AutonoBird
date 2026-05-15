# Hardware

Summary of the AutonoBird hardware stack. For the full flight-controller build log (port map, wiring decisions, component placement, ArduPilot parameters, pre-flight checklist), see [`scripts/flight-controller/readme.md`](../scripts/flight-controller/readme.md).

## Bill of materials

| Category | Part | Notes |
|---|---|---|
| Airframe | HolyBro QUAV250 complete kit | 250 mm wheelbase carbon fibre |
| Flight controller | Pixhawk 6C Mini | Ships with the kit; runs ArduPilot |
| Motors / ESCs / PDB | HolyBro kit stock | — |
| Companion computer | Raspberry Pi 5 (8 GB) | Powered via USB-C from UBEC |
| AI accelerator | Raspberry Pi AI HAT+ (26 TOPS, Hailo-8) | **Selected** over the AI HAT+ 2 (40 TOPS, Hailo-10H) per the [Benchy](https://github.com/JabbaghYounes/Benchy) benchmark suite — Hailo-8 wins by 2.6–6.9× across YOLO models at 640×640. See dissertation § 6.1 NPU Selection. |
| Stereo camera | AR0144 USB global-shutter (2560×720 SBS) | Replaces original Arducam IMX519 quad-camera plan |
| GPS | HolyBro M10 (with compass) | On a printed mast |
| RC receiver | RadioMaster RP4TD (ExpressLRS) | CRSF protocol, connected to GPS2 port |
| RC transmitter | RadioMaster Pocket | ExpressLRS |
| Video | FPV camera → HolyBro Micro OSD V2 → 1 W VTX | OSD serial to TELEM2 |
| Telemetry | HolyBro SiK 433 MHz 100 mW radio pair | Air unit to TELEM1, ground unit to laptop USB |
| Power regulator | ZTW UBEC 8A G2 (15 A peak) | Input 7–34V (2–8S), output 5.0V (jumper-selectable) |
| Battery | CNHL 1500 mAh 130C 4S 14.8V | Bottom-mounted on velcro strap |
| Auxiliary MCU | Raspberry Pi Pico 2 W | WS2812 status LEDs on GPIO6 |
| 3D-printed parts | Pi mounting plate, landing legs, GPS mast, antenna mounts, prop guards | PLA (TPU where recommended but substituted due to material availability) |

## Airframe history

The QUAV250 was chosen over larger alternatives (e.g. HolyBro X500 V2) for accessibility and cost, despite being space- and payload-constrained for a full AI companion-computer stack. The tight payload budget drives several downstream decisions:

- **Battery**: 1500 mAh 4S (not 2200 mAh) to minimise weight
- **Companion computer**: Pi 5 + HAT+ rather than a Jetson Orin Nano
- **Stereo camera**: AR0144 USB global-shutter, replacing the original Arducam IMX519 quad-camera plan. The Arducam boards produced rolling-shutter artefacts during capture and required GPIO expansion that conflicted with the HAILO AI HAT+
- **Power source**: UBEC (not a Waveshare UPS E hat) — the UPS hat was dropped due to weight and vibration-induced pogo-pin failures

## Weight budget

All-up weight is the dominant constraint on this airframe.

| Configuration | Mass (g) |
|---|---|
| Base: frame + avionics + prop guards + 1500 mAh 4S | **850** (measured) |
| + HAILO AI HAT+ | +35 |
| + AR0144 stereo camera, mount, USB cable | +60 to +80 |
| Projected with imaging stack | **~950** |
| With prop guards removed | −30 to −50 |

At 850 g the current thrust-to-weight ratio is estimated at ~2.8:1 (to be confirmed by observed hover throttle on first flight). At ~950 g with the imaging stack installed, the ratio approaches 2.5:1 — the lower bound of what is generally considered safe for autonomous flight. Prop guard removal for outdoor flight reclaims approximately 30–50 g of margin.

Expected flight time: **3–5 minutes per 1500 mAh pack** at the working AUW.

## Power architecture

```
LiPo 4S (16.8V fully charged) ─▶ PDB
                                  ├─▶ 4 × ESCs ─▶ motors
                                  ├─▶ Pixhawk + avionics rail (5V regulated by PDB)
                                  └─▶ ZTW UBEC 8A G2 input
                                          │
                                          ▼  5.0V output
                                        USB-C ─▶ Pi 5 (+ AI HAT when installed)
```

The UBEC's input is wired directly to the PDB's BAT pads (raw 4S voltage, not the regulated 5V rail). Output feeds a spliced USB-A-to-USB-C cable (USB-A end cut off, USB-C end retained). This isolates the Pi's supply from motor-current switching noise.

The ZTW UBEC was chosen over the Waveshare UPS E hat after the UPS hat's pogo pins failed repeatedly under frame vibration. The UBEC's 8 A continuous / 15 A peak rating is well above the Pi 5 + HAT+ peak demand (~5 A).

## Component placement principles

Three placement constraints shaped the layout:

1. **GPS on a mast, away from power leads and VTX** — otherwise compass interference ruins yaw control
2. **ELRS antennas in free air, V-shaped, 90° apart** — carbon fibre attenuates RF severely
3. **UBEC > 5 cm from both radios** — per ZTW manual, to avoid switching noise desensing receivers

See [`scripts/flight-controller/readme.md`](../scripts/flight-controller/readme.md) for the full placement table and photos reference.

## Mechanical modifications

- **Dual-plate top stack**: the 3D-printed Pi mounting plate sits on spacers above the original carbon top plate; both plates are retained for rigidity and mounting points
- **Custom landing legs**: printed, bolted through the existing arm mounting holes; motor cables re-routed under the arms to clear the leg screw heads
- **GPS mast**: printed, screwed to the carbon top plate at the rear, M10 attached with double-sided tape
- **ELRS antenna mounts**: printed vertical clip-on mounts for each T-dipole (PLA substituted for designer-recommended TPU)

## Further reading

- [`scripts/flight-controller/readme.md`](../scripts/flight-controller/readme.md) — full build log, port map, ArduPilot params, pre-flight checklist
- [`scripts/ar0144/steps.md`](../scripts/ar0144/steps.md) — stereo camera calibration procedure
- [architecture.md](architecture.md) — system-level architecture and data flow
