# SITL missions

QGC `.txt` waypoint files for the ArduPilot SITL. Loaded into MAVProxy with `wp load <file>`, executed by switching to `AUTO` mode after arming.

## Format

`QGC WPL 110` header, then one line per waypoint:

```
<seq> <current> <frame> <command> <p1> <p2> <p3> <p4> <x> <y> <z> <autocontinue>
```

- `frame=0` (absolute alt), `frame=3` (alt relative to home)
- Common commands: `16=NAV_WAYPOINT`, `20=NAV_RETURN_TO_LAUNCH`, `21=NAV_LAND`, `22=NAV_TAKEOFF`
- Seq 0 is the home location reference (current=1)

ArduPilot reference: <https://ardupilot.org/copter/docs/common-mavlink-mission-command-messages-mav_cmd.html>

## Missions

### `box_50m.txt`

50 m × 50 m square box around ArduPilot SITL's default home (Jerrabomberra Grasslands near Canberra, −35.363262 / 149.165237), flown clockwise at 10 m AGL, then RTL.

| Seq | Cmd | Coordinate | Alt | Notes |
|---|---|---|---|---|
| 0 | NAV_WAYPOINT | home | 584 m abs | Home reference |
| 1 | NAV_TAKEOFF | — | 10 m | Climb to 10 m |
| 2 | NAV_WAYPOINT | 50 m N of home | 10 m | First corner |
| 3 | NAV_WAYPOINT | 50 m N + 50 m E | 10 m | Second corner |
| 4 | NAV_WAYPOINT | 50 m E of home | 10 m | Third corner |
| 5 | NAV_WAYPOINT | back to home | 10 m | Closes the box |
| 6 | NAV_RTL | — | — | Land at home |

50 m offsets computed at home latitude (≈111 km/deg lat, ≈90.6 km/deg lon at −35.36° latitude). Adjust the lat/lon literals if `--home` is overridden on launch.

## How to run

In the MAVProxy prompt of a running SITL session:

```
wp load /home/vt/Documents/AutonoBird/scripts/sitl/missions/box_50m.txt
mode GUIDED
arm throttle
takeoff 10
```

Wait for `height 10` in the Console, then:

```
mode AUTO
```

The mission's `NAV_TAKEOFF` is auto-skipped (already airborne) and execution starts at waypoint 2. Do **not** try `arm throttle` → `mode AUTO` — ArduCopter's `DISARM_DELAY` (default 10 s) auto-disarms in the gap between commands, and AUTO then engages a disarmed vehicle.

MAVProxy resolves the mission-file path relative to wherever `sim_vehicle.py` was launched (`~/Documents/ardupilot/` by default), not the shell's cwd — use absolute paths.

The bird auto-takeoffs, flies the box, RTLs, lands, and disarms. The whole run is captured in the session's `mav.tlog` (written to whatever directory you launched `sim_vehicle.py` from — by default `~/Documents/ardupilot/`).

To analyse afterwards:

```bash
mavlogdump.py --types ATTITUDE,GLOBAL_POSITION_INT ~/Documents/ardupilot/mav.tlog
```

Or graph with MAVExplorer:

```bash
MAVExplorer.py ~/Documents/ardupilot/mav.tlog
```
