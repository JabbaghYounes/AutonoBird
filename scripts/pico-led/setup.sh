#!/usr/bin/env bash
# scripts/pico-led/setup.sh
#
# Pi-side setup for the Pico LED bridge. Creates a small venv with
# pyserial (the only runtime dep beyond the autonomy bridge's deps,
# which are reached via sys.path injection when --watch-fsm is used).
#
# The Pico firmware itself (scripts/pico-led/main.py) is flashed onto
# the device separately via mpremote — see readme.md for that workflow.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "==> pico-led setup: building venv at $VENV_DIR"

PYTHON_BIN="$(command -v python3.11 || command -v python3)"
echo "==> Using $PYTHON_BIN"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel >/dev/null
pip install "pyserial>=3.5"

echo
echo "==> Verifying import"
python -c "import serial; print('pyserial', serial.__version__)"

echo
echo "==> Copying config template if config.json doesn't exist"
if [ ! -f "$SCRIPT_DIR/config.json" ] && [ -f "$SCRIPT_DIR/config.example.json" ]; then
    cp "$SCRIPT_DIR/config.example.json" "$SCRIPT_DIR/config.json"
    echo "    Created config.json from template."
fi

echo
echo "==> Setup complete."
echo "    Activate with:    source $VENV_DIR/bin/activate"
echo "    Standalone test:  python $SCRIPT_DIR/led_bridge.py --test"
echo "    Health check:     python $SCRIPT_DIR/led_bridge.py --ping"
echo "    Watch FSM:        python $SCRIPT_DIR/led_bridge.py --watch-fsm"
