#!/usr/bin/env bash
# physical-drone-test.sh — bundled handheld walkaround test for the airframe
#
# Validates the full software stack (perception + autonomy / orchestrator)
# on the airframe-mounted Pi while the operator carries the powered-up
# drone around to exercise the perception loop and the reactive avoider.
# No motors spun, no props on, no flight.
#
# Runs the perception + orchestrator processes inside detached tmux
# sessions so a dropped SSH/VNC connection doesn't kill them. Logs go to
# a persistent timestamped directory under scripts/sitl/logs/.
#
# Usage:
#   ./physical-drone-test.sh                # standard test (perception + planner)
#   ./physical-drone-test.sh --pose         # also engage gesture pipeline
#                                             (first live-Hailo pose validation)
#   ./physical-drone-test.sh stop           # tear down both tmux sessions
#   ./physical-drone-test.sh status         # show running sessions + log tails
#   ./physical-drone-test.sh -h | --help

set -euo pipefail

# -------------------------------------------------------------------------- #
# Config                                                                     #
# -------------------------------------------------------------------------- #

REPO_DIR="$HOME/Documents/AutonoBird"
PERCEPTION_DIR="$REPO_DIR/scripts/perception"
AUTONOMY_DIR="$REPO_DIR/scripts/autonomy"
LOGS_PARENT="$REPO_DIR/scripts/sitl/logs"

FC_DEVICE="/dev/ttyACM0"
CONFIDENCE_THRESHOLD="0.35"
GUI_DISPLAY=":0"            # XWayland socket on wayvnc Bookworm

PERCEPTION_SESSION="walk-perception"
ORCH_SESSION="walk-orch"

# -------------------------------------------------------------------------- #
# Output helpers                                                             #
# -------------------------------------------------------------------------- #

c_info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
c_fail()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; }
die()     { c_fail "$*"; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|status] [--pose]

  start  (default)  Pre-flight checks, create a session log directory,
                    launch perception + orchestrator in detached tmux
                    sessions. Attach with:  tmux attach -t $ORCH_SESSION
  stop              Tear down both tmux sessions and the processes inside.
  status            Show whether sessions are running and tail recent logs.

  --pose            Engage the pose pipeline (depth_detect --pose +
                    orchestrator --enable-gestures). First live-Hailo
                    pose validation; retires the side-tabled backlog item.
  -h, --help        This message.

Logs land under: $LOGS_PARENT/walk_<timestamp>/
EOF
}

# -------------------------------------------------------------------------- #
# Pre-flight                                                                 #
# -------------------------------------------------------------------------- #

preflight() {
  c_info "pre-flight checks ..."

  # FC enumerated on USB
  if [[ ! -e "$FC_DEVICE" ]]; then
    die "$FC_DEVICE not present. Plug Pi USB into Pixhawk USB-C; check 'ls /dev/ttyACM*'."
  fi
  c_ok "FC enumerated at $FC_DEVICE"

  # AR0144: two UVC video interfaces on the same USB device
  local uvc_count
  uvc_count=$(lsusb -t 2>/dev/null | grep -c "Class=Video" || true)
  if (( uvc_count < 2 )); then
    die "AR0144 not enumerated (found $uvc_count UVC interfaces, expected ≥2). Check the camera USB cable."
  fi
  c_ok "AR0144 enumerated ($uvc_count UVC interfaces)"

  # Under-voltage / throttling history
  local uv_lines
  uv_lines=$(dmesg 2>/dev/null | grep -ciE "under.?volt|throttl" || true)
  if (( uv_lines > 0 )); then
    c_warn "dmesg has $uv_lines under-voltage/throttling entries — UBEC margin marginal?"
    dmesg | grep -iE "under.?volt|throttl" | tail -3
  else
    c_ok "no under-voltage / throttling in dmesg"
  fi

  command -v tmux >/dev/null 2>&1 || die "tmux not installed (apt install tmux)."
  c_ok "tmux available"

  [[ -d "$PERCEPTION_DIR/venv" ]] || die "Perception venv missing. Run 'bash setup.sh' in $PERCEPTION_DIR."
  [[ -d "$AUTONOMY_DIR/venv"   ]] || die "Autonomy venv missing. Run 'bash setup.sh' in $AUTONOMY_DIR."
  c_ok "venvs present"

  # pyserial is required for /dev/ttyACM0 even though autonomy's setup.sh
  # treats it as optional. Self-heal: install into the autonomy venv if absent.
  if ! "$AUTONOMY_DIR/venv/bin/python3" -c "import serial" >/dev/null 2>&1; then
    c_warn "pyserial missing in autonomy venv; installing ..."
    "$AUTONOMY_DIR/venv/bin/pip" install -q pyserial
  fi
  c_ok "pyserial available in autonomy venv"
}

# -------------------------------------------------------------------------- #
# Session management                                                         #
# -------------------------------------------------------------------------- #

is_running() { tmux has-session -t "$1" 2>/dev/null; }

make_walk_dir() {
  local ts dir
  ts=$(date +%Y%m%d-%H%M)
  dir="$LOGS_PARENT/walk_$ts"
  mkdir -p "$dir"
  echo "$dir"
}

start_test() {
  local pose_mode="${1:-no}"

  if is_running "$PERCEPTION_SESSION" || is_running "$ORCH_SESSION"; then
    die "A walk session is already running. Tear down with:  $(basename "$0") stop"
  fi

  preflight

  local WALK_DIR
  WALK_DIR=$(make_walk_dir)
  c_info "session log directory: $WALK_DIR"

  # Build the per-process commands. We tee stdout to logs and use --pose /
  # --enable-gestures conditionally.
  local pose_flag="" gestures_flag=""
  if [[ "$pose_mode" == "yes" ]]; then
    pose_flag="--pose"
    gestures_flag="--enable-gestures"
  fi

  local detect_cmd="python3 depth_detect.py $pose_flag \
    --jsonl $WALK_DIR/perception.jsonl \
    --threshold $CONFIDENCE_THRESHOLD \
    2>&1 | tee $WALK_DIR/perception.log"

  local orch_cmd="python3 orchestrator.py \
    --connection-uri $FC_DEVICE \
    --enable-planner $gestures_flag \
    --perception jsonl \
    --jsonl-path $WALK_DIR/perception.jsonl \
    --tail-from-end \
    2>&1 | tee $WALK_DIR/orch.log"

  local full_detect="cd $PERCEPTION_DIR && source venv/bin/activate && $detect_cmd"
  local full_orch="cd $AUTONOMY_DIR && source venv/bin/activate && $orch_cmd"

  c_info "launching $PERCEPTION_SESSION (perception + cv2 GUI on DISPLAY=$GUI_DISPLAY)"
  tmux new-session -d -s "$PERCEPTION_SESSION" \
    "export DISPLAY=$GUI_DISPLAY; $full_detect"

  # Let perception start writing the JSONL before the orchestrator tails it.
  sleep 2

  if [[ "$pose_mode" == "yes" ]]; then
    c_info "launching $ORCH_SESSION (orchestrator + planner + gestures)"
  else
    c_info "launching $ORCH_SESSION (orchestrator + planner)"
  fi
  tmux new-session -d -s "$ORCH_SESSION" "$full_orch"

  echo
  c_ok "test started. Walk pattern:"
  echo "  1. Static baseline (~30s) — confirm STATUS lines, no errors"
  echo "  2. Stationary detection (~60s) — vary distance ~3m down to ~0.5m"
  echo "  3. Carry-and-pan (~2-3min) — move while pointing at obstacles"
  echo "  4. Provoke recovery — wall <1.5m, then turn away"
  if [[ "$pose_mode" == "yes" ]]; then
    echo "  5. Gestures — STOP / LAND / COME / RECEDE at 1.5-2.5m, hold ~2s each"
  fi
  echo
  c_info "watch orchestrator: tmux attach -t $ORCH_SESSION"
  c_info "watch perception:   tmux attach -t $PERCEPTION_SESSION   (cv2 GUI on VNC)"
  c_info "detach inside tmux: Ctrl+B then D"
  c_info "stop test:          $(basename "$0") stop"
  c_info "logs in:            $WALK_DIR"
}

stop_test() {
  local stopped=0
  for s in "$PERCEPTION_SESSION" "$ORCH_SESSION"; do
    if is_running "$s"; then
      # SIGINT the foreground process first so its tee flushes, then kill the
      # session.
      tmux send-keys -t "$s" C-c 2>/dev/null || true
      sleep 0.5
      tmux kill-session -t "$s" 2>/dev/null || true
      c_ok "stopped tmux session $s"
      stopped=1
    fi
  done
  if (( stopped == 0 )); then
    c_info "no walk sessions running"
  fi
}

show_status() {
  local latest
  latest=$(ls -1d "$LOGS_PARENT"/walk_* 2>/dev/null | tail -1 || true)

  for s in "$PERCEPTION_SESSION" "$ORCH_SESSION"; do
    if is_running "$s"; then
      c_ok "tmux session '$s' running"
    else
      c_warn "tmux session '$s' NOT running"
    fi
  done
  echo

  if [[ -z "$latest" ]]; then
    c_info "no walk sessions on record yet"
    return 0
  fi
  c_info "latest log directory: $latest"

  if [[ -f "$latest/perception.log" ]]; then
    echo
    c_info "perception.log tail:"
    tail -5 "$latest/perception.log" | sed 's/^/  /'
  fi

  if [[ -f "$latest/orch.log" ]]; then
    echo
    c_info "orch.log — last 10 planner / FSM / STATUS lines:"
    grep -E "Planner: |FSM: |STATUS  fsm" "$latest/orch.log" | tail -10 | sed 's/^/  /'
    local err_count
    err_count=$(grep -ciE "error|fail|timeout" "$latest/orch.log" 2>/dev/null || true)
    echo
    if (( err_count > 0 )); then
      c_warn "$err_count error/fail/timeout lines in orch.log — investigate"
    else
      c_ok "orch.log clean (no error/fail/timeout entries)"
    fi
  fi
}

# -------------------------------------------------------------------------- #
# Arg dispatch                                                               #
# -------------------------------------------------------------------------- #

POSE_MODE="no"
COMMAND="start"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pose)             POSE_MODE="yes"; shift ;;
    -h|--help)          usage; exit 0 ;;
    start|stop|status)  COMMAND="$1"; shift ;;
    *)                  die "Unknown argument: $1 (see --help)" ;;
  esac
done

case "$COMMAND" in
  start)   start_test "$POSE_MODE" ;;
  stop)    stop_test ;;
  status)  show_status ;;
  *)       die "Unknown command: $COMMAND" ;;
esac
