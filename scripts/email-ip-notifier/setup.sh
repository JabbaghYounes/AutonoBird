#!/bin/bash
#
# Setup script for Raspberry Pi 5 Email IP Notifier
# Run this on your Raspberry Pi after copying the files
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="send-ip-email.service"

echo "=== Email IP Notifier Setup ==="
echo ""

# Check if running as root for service installation
if [ "$EUID" -ne 0 ]; then
    echo "Note: Run with sudo to install systemd service"
    echo ""
fi

# No pip dependencies needed - uses Python stdlib only
echo "[1/3] Checking configuration..."
if [ ! -f "$SCRIPT_DIR/config.json" ]; then
    cp "$SCRIPT_DIR/config.example.json" "$SCRIPT_DIR/config.json"
    echo "  Created config.json from template."
    echo ""
    echo "  !!! IMPORTANT: Edit config.json with your email credentials !!!"
    echo "  File location: $SCRIPT_DIR/config.json"
    echo ""
    echo "  You need to set:"
    echo "    - smtp_user:        Your email address"
    echo "    - smtp_password:    Your app password (NOT your real password)"
    echo "    - recipient_email:  Where to send the notification"
else
    echo "  config.json already exists"
fi

# Make main script executable
echo ""
echo "[2/3] Setting permissions..."
chmod +x "$SCRIPT_DIR/send_ip_email.py"

# Install systemd service
echo ""
echo "[3/3] Installing systemd service..."
if [ "$EUID" -eq 0 ]; then
    sed "s|/home/pi/AutonoBird/scripts/email-ip-notifier|$SCRIPT_DIR|g" "$SCRIPT_DIR/$SERVICE_FILE" > /etc/systemd/system/$SERVICE_FILE

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
echo "  1. Edit config.json with your SMTP credentials"
echo "  2. For Gmail: generate an app password at https://myaccount.google.com/apppasswords"
echo "  3. Test with: python3 $SCRIPT_DIR/send_ip_email.py"
echo ""
