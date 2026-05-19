#!/usr/bin/env bash
# scripts/flight-controller/setup.sh
#
# Creates the Pi-side MAVLink bridge venv (pymavlink + pyserial) and
# verifies it can connect to a running SITL on tcp:127.0.0.1:5760 or to
# a real Pixhawk on /dev/ttyACM0 / SiK ground unit on /dev/ttyUSB0.
#
# Run from the project root or from this directory; the script normalises
# paths via $SCRIPT_DIR.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "==> flight-controller setup: building venv at $VENV_DIR"

# Use python3.11 if available (matches the perception subsystem), else system python3
PYTHON_BIN="$(command -v python3.11 || command -v python3)"
echo "==> Using $PYTHON_BIN"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate, upgrade pip, install pymavlink + pyserial.
# pyserial is required for any serial-transport MAVLink connection — pymavlink
# lazily imports `serial` when opening /dev/ttyACM* or /dev/ttyUSB*. Without
# it, test_bridge.py / test_sik_link.py / orchestrator.py all fail with
# `ModuleNotFoundError: No module named 'serial'` once a real Pixhawk or
# SiK radio is connected.
# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel >/dev/null
pip install "pymavlink>=2.4.40" "pyserial>=3.5"

echo
echo "==> Verifying pymavlink + pyserial imports"
python -c "import pymavlink, serial; print('pymavlink', pymavlink.__version__, '/ pyserial', serial.__version__)"

echo
echo "==> Copying config template if config.json doesn't exist"
if [ ! -f "$SCRIPT_DIR/config.json" ] && [ -f "$SCRIPT_DIR/config.example.json" ]; then
    cp "$SCRIPT_DIR/config.example.json" "$SCRIPT_DIR/config.json"
    echo "    Created config.json from template. Edit if your SITL / Pixhawk endpoint differs."
fi

echo
echo "==> Setup complete."
echo "    Activate the venv with:   source $VENV_DIR/bin/activate"
echo "    Smoke-test against SITL:  python $SCRIPT_DIR/test_bridge.py"
