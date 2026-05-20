# Demo runner scripts

Three one-command launchers for the 21 May 2026 AutonoBird presentation, each set up for OBS screen recording.

```
scripts/demo/
├── gazebo_t5.sh           # Outdoor T5 collision avoidance demo
├── gazebo_classroom.sh    # Indoor classroom-replica demo
├── live_airframe.sh       # Live perception on the airframe-mounted Pi
└── readme.md
```

## Pre-flight (one-time)

The two Gazebo scripts run on the **dev workstation**. They expect:

- ArduPilot at `~/Documents/ardupilot` (override via `$ARDUPILOT_DIR`)
- ardupilot_gazebo at `~/Documents/ardupilot_gazebo` (override via `$ARDUPILOT_GAZEBO_DIR`)
- `gz`, `sim_vehicle.py`, `tmux` on PATH
- Python autonomy venv at `scripts/autonomy/venv`

The live script runs on the **airframe-mounted Pi** (not the workstation). It expects the standard `scripts/perception` + `scripts/autonomy` + `scripts/flight-controller` venvs to be in place.

## Usage

```bash
# Make scripts executable (one-time)
chmod +x scripts/demo/*.sh

# Demo 1 — outdoor T5
./scripts/demo/gazebo_t5.sh
# ... wait for SITL boot, arrange OBS, press Enter at the prompt
# ... ~3-4 min later, drone disarms
./scripts/demo/gazebo_t5.sh stop

# Demo 2 — indoor classroom
./scripts/demo/gazebo_classroom.sh
./scripts/demo/gazebo_classroom.sh stop

# Demo 3 — live on the airframe (run on the Pi, not workstation)
./scripts/demo/live_airframe.sh
./scripts/demo/live_airframe.sh stop
```

## What each script does

| Script | World | Test | Test runtime | Output |
|---|---|---|---|---|
| `gazebo_t5.sh` | `iris_obstacle.sdf` | `test_avoidance_gazebo.py` | ~60 s sim / ~4 min wall-clock | `scripts/sitl/logs/t5_gazebo_*.{json,png}` |
| `gazebo_classroom.sh` | `cs_classroom.sdf` | `test_avoidance_classroom.py` | ~50 s sim / ~3 min wall-clock | `scripts/sitl/logs/classroom_*.{json,png}` |
| `live_airframe.sh` | n/a (real airframe) | n/a (live walkaround) | as long as you walk | `scripts/sitl/logs/walk_<timestamp>/` |

## Recording workflow

For the two Gazebo demos:

1. Open OBS with a **Display Capture** source covering the screen.
2. Run the script. Gazebo + SITL boot (~35 s).
3. Script pauses with `Press Enter to start the closed-loop test ...`
4. Arrange Gazebo window + tmux terminal in your OBS scene.
5. Start OBS recording.
6. Press Enter in the launcher — the test starts.
7. Watch the drone take off, sidestep, RTL, disarm.
8. Stop OBS recording.
9. Detach from tmux (Ctrl-B then D), run the script with `stop`.

For the live airframe demo:

1. Power on the airframe (props removed).
2. SSH from laptop to Pi; open VNC viewer on laptop.
3. Start OBS recording (Display Capture of laptop screen showing VNC + SSH).
4. Run `./live_airframe.sh` on the Pi.
5. Walk in front of the airframe; show detections + depth + heatmap.
6. Stop recording.
7. Run `./live_airframe.sh stop` on the Pi.

## Skipping the OBS pause

Both Gazebo scripts pause before launching the test so you can start OBS recording at the right moment. To skip the pause (useful for non-recorded runs):

```bash
./scripts/demo/gazebo_t5.sh --no-pause
./scripts/demo/gazebo_classroom.sh --no-pause
```

## Troubleshooting

- **Gazebo never opens.** Check `/tmp/gazebo_*_demo.log`. Most common cause: `GZ_SIM_RESOURCE_PATH` doesn't include the world's folder. The scripts set this explicitly, but if you've moved `ardupilot_gazebo`, set `$ARDUPILOT_GAZEBO_DIR` first.
- **SITL exits with "Frame: UNSUPPORTED".** ArduCopter can't see the gazebo-iris param overlay. The script's `--add-param-file` flags should fix this; verify the files exist at `$ARDUPILOT_DIR/Tools/autotest/default_params/`.
- **Drone never takes off.** Usually the GPS hasn't locked yet. The script waits 20 s after SITL boot for GPS — if your machine is slow, bump `SITL_INIT_WAIT` at the top of the script.
- **Test fails with "FAIL: never reached PREARMED".** GPS lock didn't happen within 30 s of the test starting. Increase `SITL_INIT_WAIT` to give SITL more time.
- **Stale processes from a previous run.** Each script's first action is to clean up prior sessions (kill matching processes by name). If you're seeing weird crashes, run `./script.sh stop` then re-launch.
