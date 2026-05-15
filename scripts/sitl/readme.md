# ArduPilot SITL — QUAV250 overlay

ArduPilot SITL (Software In The Loop) runs the same ArduCopter firmware that flashes onto the Pixhawk, compiled as a desktop x86 binary instead. It exposes MAVLink over TCP exactly like the real Pixhawk exposes it over USB-serial, so any MAVLink client — MAVProxy, QGroundControl, Mission Planner, or a future Pi-side companion-computer bridge — can connect to it identically.

This subsystem is the dissertation's authorised path for validating autonomy without physical flight. Marking criteria do not require a hardware flight demo; an autonomous mission demonstrated in SITL satisfies the autonomy success criteria.

## What lives here

| File | Purpose |
|---|---|
| `quav250.parm` | ArduCopter parameter overlay so SITL behaves like the real airframe: X-frame, 4S battery, our FLTMODE assignments, our failsafe thresholds |
| `run_sitl.sh` | Wrapper that launches `sim_vehicle.py` with the overlay applied and `-f quad` (X-frame physics) |
| `missions/` | QGC waypoint files. See `missions/readme.md` for the format and the catalogue. |
| `logs/` | Captured `.tlog` / `.bin` flight logs from SITL runs (kept out of git by `.gitkeep` only) |
| `readme.md` | This file |

ArduPilot itself lives **out of tree** at `~/Documents/ardupilot` (override via `$ARDUPILOT_DIR`). It is not vendored into this repo — it is a tool, not source.

## One-time install

On the dev workstation (not the Pi — SITL runs on x86):

```bash
cd ~/Documents
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y   # apt deps + MAVProxy + pymavlink
./waf configure --board sitl
./waf copter
```

Add `sim_vehicle.py` to PATH. For zsh:

```zsh
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.zshrc
echo 'export PATH=$PATH:$HOME/Documents/ardupilot/Tools/autotest' >> ~/.zshrc
source ~/.zshrc
```

For bash, swap `~/.zshrc` for `~/.bashrc`.

Verify:

```bash
which sim_vehicle.py
which mavproxy.py
```

Both should print paths.

## Launching the QUAV250 in SITL

```bash
cd ~/Documents/AutonoBird
./scripts/sitl/run_sitl.sh
```

What opens:

- An **ArduCopter** terminal window — the simulated flight-controller binary
- A **Console** window — MAVProxy text status (mode, battery, GPS, EKF health)
- A **Map** window — satellite view centred on ArduPilot's default sim home near Canberra (Jerrabomberra Grasslands)
- The launch terminal becomes the `MAV>` / `STABILIZE>` prompt

Smoke test (in the MAV prompt):

```
mode GUIDED
arm throttle
takeoff 10
mode RTL
```

Exit with `Ctrl+C` in the launch terminal (the typed command `exit` is not a MAVProxy command).

## Running a scripted mission

Once the smoke test works, the next step is an autonomous waypoint mission. Mission files live in `missions/` — see `missions/readme.md` for the catalogue and waypoint-file format.

For the canonical 50 m square box at 10 m AGL (no perception, no autonomy bridge — just AUTO-mode waypoint navigation):

```
wp load /home/vt/Documents/AutonoBird/scripts/sitl/missions/box_50m.txt
mode GUIDED
arm throttle
takeoff 10
```

Wait for the Console to show `height 10` (about 10 s), then:

```
mode AUTO
```

The bird, already airborne, skips the mission's `NAV_TAKEOFF` item and picks up at waypoint 2. It flies the four corners clockwise, RTLs, lands at home, and disarms.

> **Why not just `arm throttle` → `mode AUTO`?** ArduCopter's `DISARM_DELAY` (default 10 s) auto-disarms if motors aren't spun up after arming. Typing two commands across that timeout leaves the bird disarmed when AUTO finally engages, so the mission's `NAV_TAKEOFF` silently fails. The GUIDED-takeoff-first dance avoids the race.

MAVProxy resolves `wp load` paths relative to wherever `sim_vehicle.py` was launched from (typically `~/Documents/ardupilot/`), not the shell's cwd. Use absolute paths for mission files to avoid the footgun.

The drone auto-takes-off, flies the 4 corners clockwise, RTLs, lands, and disarms. The entire run is captured in `mav.tlog` (written to wherever you launched `sim_vehicle.py` from — by default `~/Documents/ardupilot/`). Copy or symlink the `.tlog` into `scripts/sitl/logs/` to keep it associated with this subsystem.

Inspect the log:

```bash
mavlogdump.py --types ATTITUDE,GLOBAL_POSITION_INT ~/Documents/ardupilot/mav.tlog | head -40
```

Or graph it interactively:

```bash
MAVExplorer.py ~/Documents/ardupilot/mav.tlog
```

These logs are the primary data source for dissertation §6.2 (navigation accuracy: cross-track error, alt hold, mission completion time) and §6.3 (system integration: command → ack → telemetry latency, mode-switch behaviour).

## What `quav250.parm` overlays

Mirrors the FC state from `reference_ardupilot_params.md` and `scripts/flight-controller/readme.md`. Only behaviour-relevant params are included; hardware-specific ones are intentionally omitted. The overlay covers:

- **Frame**: `FRAME_CLASS=1`, `FRAME_TYPE=1` (Quad X). SITL default is `FRAME_TYPE=0` (+ frame), so this is the critical change.
- **Battery scaling**: `MOT_BAT_VOLT_MIN/MAX` set for 4S (14.0 V / 16.8 V). SITL default is 3S.
- **Battery capacity + failsafe**: `BATT_CAPACITY=1500`, `BATT_LOW_VOLT=14.0`, `BATT_CRT_VOLT=13.2`, `BATT_FS_LOW_ACT=BATT_FS_CRT_ACT=1` (Land).
- **GCS failsafe**: `FS_GCS_ENABLE=0` (matches current bench state — re-enable before link-loss autonomy tests).
- **Flight modes on CH6**: `FLTMODE_CH=6`, `FLTMODE1=Stabilize`, `FLTMODE4=AltHold`, `FLTMODE6=RTL`. Mirrors the SB switch assignment on the RadioMaster Pocket.
- **Throttle failsafe**: `FS_THR_ENABLE=1`, `FS_THR_VALUE=975`.

What is **not** overlaid and why:

| Class | Reason |
|---|---|
| `SERIAL*_*` | No real UARTs in SITL |
| `COMPASS_OFS/PRIO/ORIENT` | SITL compass is ideal; offsets harmful |
| `COMPASS_USE2/3` | SITL has one simulated mag — leave default |
| `MOT_PWM_TYPE`, `SERVO_BLH_*` | No real ESCs |
| `BATT_VOLT_PIN`, `BATT_CURR_PIN`, `BATT_VOLT_MULT`, `BATT_AMP_PERVLT` | No real ADC; SITL provides synthetic voltage/current |
| `BRD_SAFETY_DEFLT` | No safety switch in sim |
| `AHRS_ORIENTATION` | Pixhawk-mount-specific; default is correct for SITL |
| `RSSI_TYPE` | No CRSF link in sim |
| `RC*_MIN/MAX/TRIM` | SITL's RC simulator overrides these |

If a field campaign turns up new airframe params worth mirroring, add them to `quav250.parm` and note the rationale in the comment block.

## Gotchas

- **SITL runs on the desktop, not the Pi.** The Pi 5 is busy with the perception loop (138 ms loop budget). Putting SITL on the Pi would force the two to compete for CPU. The autonomy bridge that will later run on the Pi connects to SITL over the network (`tcp:127.0.0.1:5760` locally, or `tcp:<laptop-ip>:5760` from the Pi).
- **Frame mismatch shows up as bad attitude control.** If you launch without `-f quad`, the physics model is `+` (the SITL default) while `FRAME_TYPE=1` (X) is set in params. The motor mapping is then wrong and the bird flips on takeoff. `run_sitl.sh` always passes `-f quad`.
- **`MAV> exit` is invalid.** Use `Ctrl+C` in the MAVProxy terminal to close everything cleanly.
- **`arm throttle` auto-disarms after 10 s if motors don't spin** (ArduCopter `DISARM_DELAY` default). Always issue the takeoff (`takeoff <alt>`) immediately after `arm throttle`, then switch to AUTO once airborne. Don't try `arm throttle` → `mode AUTO` — the gap between commands typically exceeds the timeout and AUTO engages a disarmed vehicle, silently failing the mission's `NAV_TAKEOFF`.
- **The matplotlib "Unable to import Axes3D" warnings are cosmetic.** Caused by having both system-apt and pip matplotlib installed. The map still renders fine.
- **`eeprom.bin` persists across runs.** SITL stores params in `$ARDUPILOT_DIR/eeprom.bin`. Subsequent launches load from there, not from `quav250.parm`. To re-apply the overlay cleanly, delete `eeprom.bin` first or pass `--wipe-eeprom` to `sim_vehicle.py`.
- **Battery failsafe is disabled in SITL on purpose.** SITL's simulated battery sits at ~12.6 V regardless of `SIM_BATT_VOLTAGE` — the param exists and accepts a value, but the sim battery model in current ArduPilot master doesn't pick it up. A 4S-tuned `BATT_LOW_VOLT=14.0` therefore latches the low-voltage failsafe at boot (`PreArm: Battery 1 low voltage failsafe`) and the FC refuses to arm. The overlay sets `BATT_LOW_VOLT=0` and `BATT_CRT_VOLT=0` to disable the failsafe. The real airframe's actual failsafe thresholds are documented in `scripts/flight-controller/readme.md` and validated on hardware; SITL is the navigation / autonomy tool.

## Where this fits in the dissertation

- **§5 implementation**: SITL setup is part of the verification-environment narrative
- **§6.2 navigation**: SITL flight logs (`.tlog` / `.bin`) are the primary data source
- **§6.3 system integration**: SITL + Pi MAVLink bridge proves the perception → flight-command path
- **§7 conclusions**: SITL bridges the gap between "perception works" (§6.1) and "autonomous flight works", without depending on the deferred hardware hover

## Next steps in this subsystem

| Task | State |
|---|---|
| Phase 1 — install + smoke test SITL on the workstation | done |
| Phase 2 — apply QUAV250 param overlay, re-run smoke test | done |
| Phase 3 — scripted mission via MAVProxy (`wp load`, mode AUTO), capture `.tlog` | in progress |
| Phase 4 — Pi-side MAVLink bridge in `scripts/flight-controller/` connecting to SITL over TCP | pending |
| Phase 5 — perception → bridge → SITL closed loop | pending |

See `docs/architecture.md` and the dissertation for the broader picture.
