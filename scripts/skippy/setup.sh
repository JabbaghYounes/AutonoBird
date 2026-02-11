#!/bin/bash
#
# Setup script for Skippy Voice Assistant
# Run this on your Raspberry Pi: sudo bash setup.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="skippy.service"
ACTUAL_USER="${SUDO_USER:-$(whoami)}"
VOICES_DIR="$SCRIPT_DIR/voices"
PIPER_VOICE="en_US-lessac-medium"
PIPER_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

echo "=== Skippy Voice Assistant Setup ==="
echo ""

# Check if running as root for service installation
if [ "$EUID" -ne 0 ]; then
    echo "Note: Run with sudo to install systemd service"
    echo ""
fi

# Step 1: System packages
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    python3-pyaudio \
    libopenblas-dev \
    libasound2-dev \
    alsa-utils \
    wget
echo "  System packages installed"

# Step 2: Python virtual environment and pip packages
echo ""
echo "[2/6] Setting up Python environment..."
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    sudo -u "$ACTUAL_USER" python3 -m venv "$SCRIPT_DIR/venv"
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi
sudo -u "$ACTUAL_USER" "$SCRIPT_DIR/venv/bin/pip" install --upgrade pip -q
sudo -u "$ACTUAL_USER" "$SCRIPT_DIR/venv/bin/pip" install -q \
    "numpy<2" \
    openwakeword \
    faster-whisper \
    google-genai \
    piper-tts \
    pyaudio
echo "  Python packages installed"

# Step 3: Download Piper voice model
echo ""
echo "[3/6] Downloading Piper voice model..."
mkdir -p "$VOICES_DIR"
chown "$ACTUAL_USER:$ACTUAL_USER" "$VOICES_DIR"
if [ ! -f "$VOICES_DIR/${PIPER_VOICE}.onnx" ]; then
    sudo -u "$ACTUAL_USER" wget -q -O "$VOICES_DIR/${PIPER_VOICE}.onnx" \
        "${PIPER_BASE_URL}/${PIPER_VOICE}.onnx"
    sudo -u "$ACTUAL_USER" wget -q -O "$VOICES_DIR/${PIPER_VOICE}.onnx.json" \
        "${PIPER_BASE_URL}/${PIPER_VOICE}.onnx.json"
    echo "  Voice model downloaded"
else
    echo "  Voice model already exists"
fi

# Step 4: Download openWakeWord pre-trained models
echo ""
echo "[4/6] Downloading openWakeWord models..."
sudo -u "$ACTUAL_USER" "$SCRIPT_DIR/venv/bin/python3" -c \
    "import openwakeword; openwakeword.utils.download_models()"
echo "  Wake word models downloaded"

# Step 5: Configuration
echo ""
echo "[5/6] Checking configuration..."
if [ ! -f "$SCRIPT_DIR/config.json" ]; then
    cp "$SCRIPT_DIR/config.example.json" "$SCRIPT_DIR/config.json"
    chown "$ACTUAL_USER:$ACTUAL_USER" "$SCRIPT_DIR/config.json"
    echo "  Created config.json from template."
    echo ""
    echo "  !!! IMPORTANT: Edit config.json with your Gemini API key !!!"
    echo "  File location: $SCRIPT_DIR/config.json"
    echo ""
    echo "  Get your free API key at: https://aistudio.google.com"
else
    echo "  config.json already exists"
fi

# Step 6: Install systemd service
echo ""
echo "[6/6] Installing systemd service..."
chmod +x "$SCRIPT_DIR/skippy.py"
if [ "$EUID" -eq 0 ]; then
    sed -e "s|INSTALL_DIR|$SCRIPT_DIR|g" \
        -e "s|INSTALL_USER|$ACTUAL_USER|g" \
        "$SCRIPT_DIR/$SERVICE_FILE" > /etc/systemd/system/$SERVICE_FILE

    systemctl daemon-reload
    systemctl enable $SERVICE_FILE
    echo "  Service installed and enabled!"
    echo ""
    echo "  Commands:"
    echo "    Start:   sudo systemctl start $SERVICE_FILE"
    echo "    Stop:    sudo systemctl stop $SERVICE_FILE"
    echo "    Status:  sudo systemctl status $SERVICE_FILE"
    echo "    Logs:    journalctl -u $SERVICE_FILE -f"
else
    echo "  Skipped (run with sudo to install service)"
    echo ""
    echo "  To install manually:"
    echo "    sudo cp $SCRIPT_DIR/$SERVICE_FILE /etc/systemd/system/"
    echo "    sudo systemctl daemon-reload"
    echo "    sudo systemctl enable $SERVICE_FILE"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit config.json with your Gemini API key"
echo "  2. Get a free key at: https://aistudio.google.com"
echo "  3. Test with: $SCRIPT_DIR/venv/bin/python3 $SCRIPT_DIR/skippy.py"
echo "  4. Start service: sudo systemctl start $SERVICE_FILE"
echo "  5. View logs: journalctl -u $SERVICE_FILE -f"
echo ""
