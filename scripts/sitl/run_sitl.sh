#!/usr/bin/env bash
# Launch ArduCopter SITL with the QUAV250 parameter overlay.
#
# Requires ArduPilot cloned out-of-tree at $ARDUPILOT_DIR (default ~/Documents/ardupilot)
# and sim_vehicle.py / mavproxy.py on PATH -- see readme.md for one-time install.
#
# Any additional flags passed to this script are forwarded to sim_vehicle.py.
# Common extras:
#   -L <location>   one of Tools/autotest/locations.txt (default home is Canberra)
#   --speedup N     run the sim N times faster than wall clock
#   --out udp:HOST:PORT  forward MAVLink to a GCS (e.g. QGroundControl on a laptop)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/Documents/ardupilot}"
PARM_FILE="$SCRIPT_DIR/quav250.parm"

if [ ! -d "$ARDUPILOT_DIR" ]; then
    echo "error: ArduPilot not found at $ARDUPILOT_DIR" >&2
    echo "Set ARDUPILOT_DIR or clone: git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git ~/Documents/ardupilot" >&2
    exit 1
fi

if ! command -v sim_vehicle.py >/dev/null 2>&1; then
    echo "error: sim_vehicle.py not on PATH" >&2
    echo "Source your shell rc or add: export PATH=\$PATH:$ARDUPILOT_DIR/Tools/autotest" >&2
    exit 1
fi

cd "$ARDUPILOT_DIR"
exec sim_vehicle.py \
    -v ArduCopter \
    -f quad \
    --add-param-file="$PARM_FILE" \
    --console \
    --map \
    "$@"
