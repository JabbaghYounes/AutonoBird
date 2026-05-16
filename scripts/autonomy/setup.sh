#!/usr/bin/env bash
# scripts/autonomy/setup.sh
#
# Creates the autonomy venv. Same deps as flight-controller (pymavlink) plus
# numpy for vector math the planner needs. The autonomy subsystem imports
# bridge.py from ../flight-controller via sys.path manipulation; this script
# does not duplicate or vendor the bridge.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
BRIDGE_DIR="$(cd "$SCRIPT_DIR/../flight-controller" && pwd)"

echo "==> autonomy setup: building venv at $VENV_DIR"
echo "    (imports bridge from $BRIDGE_DIR)"

if [ ! -d "$BRIDGE_DIR" ] || [ ! -f "$BRIDGE_DIR/bridge.py" ]; then
    echo "error: bridge.py not found at $BRIDGE_DIR — run scripts/flight-controller/setup.sh first" >&2
    exit 1
fi

PYTHON_BIN="$(command -v python3.11 || command -v python3)"
echo "==> Using $PYTHON_BIN"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel >/dev/null
pip install "pymavlink>=2.4.40" "numpy>=1.24,<3"

echo
echo "==> Verifying imports"
PYTHONPATH="$BRIDGE_DIR:$PYTHONPATH" python -c "
import sys
sys.path.insert(0, '$BRIDGE_DIR')
from bridge import Vehicle
import numpy as np
print('bridge.Vehicle OK; numpy', np.__version__)
"

echo
echo "==> Copying config template if config.json doesn't exist"
if [ ! -f "$SCRIPT_DIR/config.json" ] && [ -f "$SCRIPT_DIR/config.example.json" ]; then
    cp "$SCRIPT_DIR/config.example.json" "$SCRIPT_DIR/config.json"
    echo "    Created config.json from template."
fi

echo
echo "==> Setup complete."
echo "    Activate the venv with:  source $VENV_DIR/bin/activate"
echo "    State-machine smoke test (with SITL up):  python $SCRIPT_DIR/test_state_machine.py"
