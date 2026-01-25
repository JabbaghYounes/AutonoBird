#!/bin/bash
#
# Setup script for Raspberry Pi 5 Discord IP Notifier
# Run this on your Raspberry Pi after copying the files
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="send-ip-discord.service"
INSTALL_DIR="/home/pi/AutonoBird/scripts"

echo "=== Discord IP Notifier Setup ==="
echo ""

# Check if running as root for service installation
if [ "$EUID" -ne 0 ]; then
    echo "Note: Run with sudo to install systemd service"
    echo ""
fi

# Install Python dependency
echo "[1/4] Installing discord.py..."
pip3 install discord.py --break-system-packages 2>/dev/null || pip3 install discord.py

# Check for config file
echo ""
echo "[2/4] Checking configuration..."
if [ ! -f "$SCRIPT_DIR/config.json" ]; then
    echo "  Creating config.json from template..."
    cp "$SCRIPT_DIR/config.example.json" "$SCRIPT_DIR/config.json"
    echo ""
    echo "  !!! IMPORTANT: Edit config.json with your Discord credentials !!!"
    echo "  File location: $SCRIPT_DIR/config.json"
    echo ""
    echo "  You need to set:"
    echo "    - bot_token: Your Discord bot token"
    echo "    - discord_user_id: Your Discord user ID"
else
    echo "  config.json already exists"
fi

# Make main script executable
echo ""
echo "[3/4] Setting permissions..."
chmod +x "$SCRIPT_DIR/send_ip_discord.py"

# Install systemd service
echo ""
echo "[4/4] Installing systemd service..."
if [ "$EUID" -eq 0 ]; then
    # Update paths in service file for current install location
    sed "s|/home/pi/AutonoBird/scripts|$SCRIPT_DIR|g" "$SCRIPT_DIR/$SERVICE_FILE" > /etc/systemd/system/$SERVICE_FILE

    systemctl daemon-reload
    systemctl enable $SERVICE_FILE
    echo "  Service installed and enabled!"
    echo ""
    echo "  Commands:"
    echo "    Test now:      sudo systemctl start $SERVICE_FILE"
    echo "    Check status:  sudo systemctl status $SERVICE_FILE"
    echo "    View logs:     journalctl -u $SERVICE_FILE"
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
echo "  1. Create a Discord bot at https://discord.com/developers/applications"
echo "  2. Copy the bot token to config.json"
echo "  3. Get your Discord user ID and add to config.json"
echo "  4. Invite bot to a server you share (for DMs to work)"
echo "  5. Test with: python3 $SCRIPT_DIR/send_ip_discord.py"
echo ""
