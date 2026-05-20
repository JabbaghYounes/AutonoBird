#!/usr/bin/env bash
#
# scripts/demo/gazebo_t5.sh
#
# Launch the Gazebo T5 collision-avoidance demo end-to-end for screen
# recording. Spawns Gazebo (its own GUI window), then ArduPilot SITL +
# perception bridge + test_avoidance_gazebo.py inside a tmux session.
#
# Usage:
#   ./gazebo_t5.sh              # full launch, pause for OBS, then attach
#   ./gazebo_t5.sh stop         # tear everything down
#   ./gazebo_t5.sh --no-pause   # skip the "press Enter for OBS" prompt
#
# Recording workflow:
#   1. Start OBS with a Display Capture source covering the screen
#      (or two sources: Gazebo window + terminal).
#   2. Run this script. Gazebo + SITL boot.
#   3. When prompted "Press Enter to launch the test", arrange your
#      windows + start OBS recording, then press Enter.
#   4. Test runs ~3-4 min wall-clock (Gazebo at ~20% real-time factor).
#      Drone takes off, sees cylinder at 12 m N, sidesteps east,
#      passes obstacle, RTLs, lands, disarms.
#   5. Stop recording when the drone disarms.
#   6. Ctrl-B then D to detach, run "./gazebo_t5.sh stop" to clean up.
#
# Outputs from the test go to scripts/sitl/logs/t5_gazebo_<timestamp>.{json,png}

set -euo pipefail

PROJECT_DIR=~/Documents/AutonoBird
ARDUPILOT_DIR=${ARDUPILOT_DIR:-$HOME/Documents/ardupilot}
ARDUPILOT_GAZEBO_DIR=${ARDUPILOT_GAZEBO_DIR:-$HOME/Documents/ardupilot_gazebo}
GAZEBO_WORLD=iris_obstacle.sdf
TEST_SCRIPT=$PROJECT_DIR/scripts/autonomy/test_avoidance_gazebo.py
TMUX_SESSION=gz_t5_demo
GAZEBO_INIT_WAIT=15
SITL_INIT_WAIT=20

c_info() { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
c_ok()   { printf '\033[1;32m[OK]\033[0m   %s\n' "$*"; }
c_warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
c_fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; }
die() { c_fail "$*"; exit 1; }

cleanup() {
  c_info "Tearing down session ..."
  tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
  pkill -f "gz sim.*$GAZEBO_WORLD" 2>/dev/null || true
  pkill -f "sim_vehicle.py" 2>/dev/null || true
  pkill -f "ardupilot.*JSON" 2>/dev/null || true
  pkill -f "gazebo_perception_bridge.py" 2>/dev/null || true
  pkill -f "test_avoidance_gazebo.py" 2>/dev/null || true
  c_ok "Cleanup complete."
}

PAUSE_FOR_OBS=yes
case "${1:-start}" in
  stop)
    cleanup
    exit 0
    ;;
  --no-pause)
    PAUSE_FOR_OBS=no
    ;;
esac

[ -d "$ARDUPILOT_DIR" ] || die "ArduPilot not found at $ARDUPILOT_DIR"
[ -d "$ARDUPILOT_GAZEBO_DIR" ] || c_warn "ardupilot_gazebo dir not at $ARDUPILOT_GAZEBO_DIR (resource path may need manual override)"
command -v gz >/dev/null 2>&1 || die "gz command not on PATH (install gz-harmonic)"
command -v sim_vehicle.py >/dev/null 2>&1 || die "sim_vehicle.py not on PATH (source ~/.zshrc or add $ARDUPILOT_DIR/Tools/autotest to PATH)"
command -v tmux >/dev/null 2>&1 || die "tmux not installed"
[ -f "$TEST_SCRIPT" ] || die "Test script not found: $TEST_SCRIPT"

export GZ_SIM_RESOURCE_PATH="$ARDUPILOT_GAZEBO_DIR/models:$ARDUPILOT_GAZEBO_DIR/worlds:$PROJECT_DIR/scripts/sitl/gazebo/models:$PROJECT_DIR/scripts/sitl/gazebo/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

cleanup
sleep 1

c_info "[1/3] Starting Gazebo with $GAZEBO_WORLD ..."
gz sim -v4 -r $GAZEBO_WORLD >/tmp/gazebo_t5_demo.log 2>&1 &
GAZEBO_PID=$!
c_info "      Gazebo PID: $GAZEBO_PID — waiting $GAZEBO_INIT_WAIT s for init ..."
sleep $GAZEBO_INIT_WAIT

if ! kill -0 $GAZEBO_PID 2>/dev/null; then
  c_fail "Gazebo exited prematurely. Tail of log:"
  tail -20 /tmp/gazebo_t5_demo.log
  exit 1
fi
c_ok "      Gazebo running. Topics: $(gz topic --list 2>/dev/null | head -3 | tr '\n' ' ')"

c_info "[2/3] Starting ArduPilot SITL (gazebo-iris frame) ..."
tmux new-session -d -s $TMUX_SESSION -n demo -x 240 -y 80
tmux send-keys -t $TMUX_SESSION:demo "clear; cd $ARDUPILOT_DIR && rm -f eeprom.bin && sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --add-param-file=$ARDUPILOT_DIR/Tools/autotest/default_params/copter.parm --add-param-file=$ARDUPILOT_DIR/Tools/autotest/default_params/gazebo-iris.parm --console --map -w" C-m
c_info "      Waiting $SITL_INIT_WAIT s for SITL boot + GPS lock ..."
sleep $SITL_INIT_WAIT
c_ok "      SITL boot phase complete."

c_info "[3/3] Splitting tmux pane for test script ..."
tmux split-window -h -t $TMUX_SESSION:demo
tmux select-pane -t $TMUX_SESSION:demo.1

if [[ "$PAUSE_FOR_OBS" == "yes" ]]; then
  echo
  c_info "Ready to launch test_avoidance_gazebo.py."
  c_info "Arrange Gazebo + terminal in your OBS scene now."
  read -p "Press Enter to start the closed-loop test ... " </dev/tty
fi

tmux send-keys -t $TMUX_SESSION:demo.1 "cd $PROJECT_DIR && source scripts/autonomy/venv/bin/activate && python $TEST_SCRIPT" C-m

c_ok "Test launched. Attaching to tmux session $TMUX_SESSION."
c_info "Ctrl-B then D to detach. Test runs ~3-4 min wall-clock."
c_info "Run '$0 stop' afterwards to tear down."
sleep 1
tmux attach -t $TMUX_SESSION
