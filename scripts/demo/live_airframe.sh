#!/usr/bin/env bash
#
# scripts/demo/live_airframe.sh
#
# Launch the live perception demo on the airframe-mounted Pi for screen
# recording. Thin wrapper around scripts/autonomy/physical-drone-test.sh
# that forces --show-depth so the colourised SGBM heatmap window is
# visible to the audience.
#
# This runs on the airframe Pi (NOT the dev workstation). SSH into the
# Pi from the laptop, then run this script. Two cv2 windows appear via
# VNC; the orchestrator log streams in tmux.
#
# Usage:
#   ./live_airframe.sh              # full launch
#   ./live_airframe.sh stop         # tear everything down
#   ./live_airframe.sh status       # show running tmux sessions + tail logs
#   ./live_airframe.sh --pose       # also engage gesture pipeline (untested live)
#
# Recording workflow:
#   1. Power on the airframe (props REMOVED, motors locked out).
#   2. SSH from laptop to the Pi; open a VNC viewer too.
#   3. Start OBS on the laptop with a Display Capture covering the VNC
#      viewer + the SSH terminal.
#   4. Begin recording.
#   5. Run this script. Two cv2 windows appear: YOLO+bbox+depth on the
#      left, colourised depth heatmap on the right. Orchestrator log
#      tmux window streams the FSM + planner state.
#   6. Walk in front of the airframe at various distances; show
#      detections + depth values + heatmap colour changes.
#   7. Stop recording when done.
#   8. Run "./live_airframe.sh stop" to tear down.
#
# Pre-flight checks the underlying script performs:
#   - Pi venv exists at scripts/perception/venv and scripts/autonomy/venv
#   - HEF model at scripts/perception/models/model.hef
#   - cs_classroom calibration .npz at scripts/ar0144/stereo_calibration_data/
#   - FC USB device exists (/dev/ttyACM0 or configured override)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSICAL_TEST="$SCRIPT_DIR/../autonomy/physical-drone-test.sh"

if [[ ! -f "$PHYSICAL_TEST" ]]; then
  printf '\033[1;31m[FAIL]\033[0m physical-drone-test.sh not found at %s\n' "$PHYSICAL_TEST"
  exit 1
fi

# Forward all args, but always force --show-depth on for the live demo
# (heatmap window is the headline visual).
FORWARDED_ARGS=()
case "${1:-start}" in
  stop|status)
    exec "$PHYSICAL_TEST" "$@"
    ;;
  start|"")
    FORWARDED_ARGS+=("start" "--show-depth")
    ;;
  --pose)
    FORWARDED_ARGS+=("start" "--show-depth" "--pose")
    ;;
  *)
    # Pass through any other flag combination.
    FORWARDED_ARGS=("$@" "--show-depth")
    ;;
esac

printf '\033[1;34m[INFO]\033[0m Forwarding to: %s %s\n' "$PHYSICAL_TEST" "${FORWARDED_ARGS[*]}"
exec "$PHYSICAL_TEST" "${FORWARDED_ARGS[@]}"
