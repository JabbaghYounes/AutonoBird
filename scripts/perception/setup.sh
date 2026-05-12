#!/usr/bin/env bash
# =============================================================================
# AutonoBird perception subsystem — venv setup
# =============================================================================
# Creates a dedicated venv under scripts/perception/venv with the right
# version pins for the AutonoBird + Hailo-8 stack, without touching
# Benchy's venv.
#
# Version pins (all interlocked, do not bump independently):
#   - hailort  4.23  (system install, requires numpy<2)
#   - numpy    <2    (forced by hailort)
#   - opencv-python <4.11 (4.11+ requires numpy>=2, conflicts with hailort)
#
# The hailo_platform Python module is sourced by symlink from Benchy's venv
# rather than re-installed — Benchy already has a working build pinned to
# the system's libhailort.so.4.23.0, and reproducing that install requires
# the original Hailo SDK wheel which isn't tracked in this repo.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
BENCHY_VENV_HAILO="$HOME/Documents/Benchy/venv/lib/python3.11/site-packages/hailo_platform"

echo "================================================================"
echo " AutonoBird perception venv setup"
echo "================================================================"
echo "Target venv: $VENV_DIR"
echo ""

if [ -d "$VENV_DIR" ]; then
    echo "[WARN] venv already exists at $VENV_DIR"
    read -r -p "Delete and recreate? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Aborting."
        exit 1
    fi
fi

echo "[1/4] Creating venv..."
python3 -m venv "$VENV_DIR"

echo "[2/4] Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo "[3/4] Installing pinned OpenCV (<4.11) and NumPy (<2)..."
# Pinned because:
#   - hailort 4.23 requires numpy<2
#   - opencv-python 4.11+ requires numpy>=2
# Both constraints together pin us to opencv-python<4.11 + numpy<2.
"$VENV_DIR/bin/pip" install "opencv-python<4.11" "numpy<2"

echo "[4/4] Linking hailo_platform from Benchy's venv..."
if [ ! -d "$BENCHY_VENV_HAILO" ]; then
    echo "[ERROR] Benchy's hailo_platform not found at:"
    echo "        $BENCHY_VENV_HAILO"
    echo ""
    echo "        Either:"
    echo "          (a) install Benchy and its venv first, or"
    echo "          (b) install hailo_platform into this venv manually:"
    echo "              source $VENV_DIR/bin/activate"
    echo "              pip install /path/to/hailort-4.23.0-...-aarch64.whl"
    exit 1
fi

SITE_PACKAGES="$("$VENV_DIR/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
ln -sf "$BENCHY_VENV_HAILO" "$SITE_PACKAGES/hailo_platform"

# Also link the dist-info so pip recognises the install
BENCHY_SITE="$(dirname "$BENCHY_VENV_HAILO")"
for dist in "$BENCHY_SITE"/hailort-*.dist-info; do
    [ -e "$dist" ] && ln -sf "$dist" "$SITE_PACKAGES/" && break
done

echo ""
echo "================================================================"
echo " Verification"
echo "================================================================"
"$VENV_DIR/bin/python3" - <<'PY'
import cv2, numpy, hailo_platform
print(f"  cv2            : {cv2.__version__}")
print(f"  numpy          : {numpy.__version__}")
print(f"  hailo_platform : {hailo_platform.__version__}")
# Smoke-test cv2 GUI capability
try:
    cv2.namedWindow("setup_test", cv2.WINDOW_NORMAL)
    cv2.destroyAllWindows()
    print("  cv2 GUI        : OK")
except cv2.error as e:
    print(f"  cv2 GUI        : FAILED ({e})")
    print("                   The opencv-python aarch64 wheel may lack GUI support.")
    print("                   Fall back: sudo apt install python3-opencv  and recreate")
    print("                   this venv with --system-site-packages to inherit it.")
PY

echo ""
echo "Setup complete."
echo "Activate: source $VENV_DIR/bin/activate"
echo "Run:      python3 yolo_detect.py"
