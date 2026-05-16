#!/usr/bin/env bash
# scripts/flight-controller/setup.sh
#
# Creates the Pi-side MAVLink bridge venv (pymavlink only) and verifies it
# can connect to a running SITL on tcp:127.0.0.1:5760.
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

# Activate, upgrade pip, install pymavlink
# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel >/dev/null
pip install "pymavlink>=2.4.40"

echo
echo "==> Verifying pymavlink import"
python -c "import pymavlink; print('pymavlink', pymavlink.__version__)"

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
