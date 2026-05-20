#!/usr/bin/env bash
#
# scripts/demo/gazebo_classroom.sh
#
# Launch the indoor classroom-replica collision-avoidance demo end-to-end
# for screen recording. Same architecture as gazebo_t5.sh but uses the
# cs_classroom world (a Gazebo replica of the AutonoBird demo room) and
# the indoor-envelope flight test (test_avoidance_classroom.py).
#
# Usage:
#   ./gazebo_classroom.sh              # full launch, pause for OBS, then attach
#   ./gazebo_classroom.sh stop         # tear everything down
#   ./gazebo_classroom.sh --no-pause   # skip the OBS pause
#
# Recording workflow:
#   1. Start OBS with a Display Capture source covering the screen.
#   2. Run this script. Gazebo opens the classroom world (room + tables +
#      chairs + servers + storage). SITL boots.
#   3. When prompted, arrange windows + start OBS recording, press Enter.
#   4. Test runs ~3-4 min wall-clock. Drone takes off to 1.5 m AGL in the
#      western corridor, cruises north, sidesteps the red cylinder, RTLs.
#   5. Stop recording when the drone disarms.
#   6. Ctrl-B then D to detach, run "./gazebo_classroom.sh stop" to clean up.
#
# Outputs from the test go to scripts/sitl/logs/classroom_<timestamp>.{json,png}

set -euo pipefail

PROJECT_DIR=~/Documents/AutonoBird
ARDUPILOT_DIR=${ARDUPILOT_DIR:-$HOME/Documents/ardupilot}
ARDUPILOT_GAZEBO_DIR=${ARDUPILOT_GAZEBO_DIR:-$HOME/Documents/ardupilot_gazebo}
GAZEBO_WORLD=cs_classroom.sdf
TEST_SCRIPT=$PROJECT_DIR/scripts/autonomy/test_avoidance_classroom.py
TMUX_SESSION=gz_classroom_demo
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
  pkill -f "test_avoidance_classroom.py" 2>/dev/null || true
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
command -v sim_vehicle.py >/dev/null 2>&1 || die "sim_vehicle.py not on PATH"
command -v tmux >/dev/null 2>&1 || die "tmux not installed"
[ -f "$TEST_SCRIPT" ] || die "Test script not found: $TEST_SCRIPT (run scripts/sitl/gazebo/build_room_world.py first if cs_classroom.sdf is missing too)"
[ -f "$PROJECT_DIR/scripts/sitl/gazebo/worlds/cs_classroom.sdf" ] || die "cs_classroom.sdf not generated yet. Run: python3 $PROJECT_DIR/scripts/sitl/gazebo/build_room_world.py"

export GZ_SIM_RESOURCE_PATH="$ARDUPILOT_GAZEBO_DIR/models:$ARDUPILOT_GAZEBO_DIR/worlds:$PROJECT_DIR/scripts/sitl/gazebo/models:$PROJECT_DIR/scripts/sitl/gazebo/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

cleanup
sleep 1

c_info "[1/3] Starting Gazebo with $GAZEBO_WORLD (classroom replica) ..."
gz sim -v4 -r $GAZEBO_WORLD >/tmp/gazebo_classroom_demo.log 2>&1 &
GAZEBO_PID=$!
c_info "      Gazebo PID: $GAZEBO_PID — waiting $GAZEBO_INIT_WAIT s for init ..."
sleep $GAZEBO_INIT_WAIT

if ! kill -0 $GAZEBO_PID 2>/dev/null; then
  c_fail "Gazebo exited prematurely. Tail of log:"
  tail -20 /tmp/gazebo_classroom_demo.log
  exit 1
fi
c_ok "      Gazebo running."

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
  c_info "Ready to launch test_avoidance_classroom.py."
  c_info "Arrange Gazebo + terminal in your OBS scene now."
  read -p "Press Enter to start the closed-loop test ... " </dev/tty
fi

tmux send-keys -t $TMUX_SESSION:demo.1 "cd $PROJECT_DIR && source scripts/autonomy/venv/bin/activate && python $TEST_SCRIPT" C-m

c_ok "Test launched. Attaching to tmux session $TMUX_SESSION."
c_info "Ctrl-B then D to detach. Test runs ~3-4 min wall-clock."
c_info "Run '$0 stop' afterwards to tear down."
sleep 1
tmux attach -t $TMUX_SESSION
