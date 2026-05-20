#!/bin/bash
#
# Setup script for Raspberry Pi 5 Email IP Notifier
#
# Usage:
#   sudo bash setup.sh [install]    Install or update (regenerates the
#                                   systemd unit file with the current
#                                   directory baked in).
#   sudo bash setup.sh uninstall    Stop, disable, and remove the systemd
#                                   service. Leaves config.json and the
#                                   Python script files alone.
#        bash setup.sh help         Print usage and exit.
#
# If the notifier directory is moved after a previous install, just
# re-run "sudo bash setup.sh" from the new location -- the script
# detects the stale path in the existing service file and re-stamps it
# with the current one.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="send-ip-email.service"
SERVICE_INSTALL_PATH="/etc/systemd/system/$SERVICE_FILE"
ACTUAL_USER="${SUDO_USER:-$(whoami)}"
COMMAND="${1:-install}"

print_usage() {
    cat <<EOF
Email IP Notifier setup

Usage:
  sudo bash setup.sh [install]    Install or update. Regenerates the
                                  systemd unit file with the current
                                  directory baked into ExecStart and
                                  WorkingDirectory.
  sudo bash setup.sh uninstall    Stop, disable, and remove the systemd
                                  service. Leaves config.json and the
                                  Python script files alone.
       bash setup.sh help         Print this message.

If the email-ip-notifier directory is moved after a previous install,
re-run "sudo bash setup.sh" from the new location -- it detects the
stale path and re-stamps the service file.
EOF
}

# Read WorkingDirectory= from the installed service file. Echoes empty
# string if no service is installed.
detect_installed_dir() {
    if [ -f "$SERVICE_INSTALL_PATH" ]; then
        grep -E "^WorkingDirectory=" "$SERVICE_INSTALL_PATH" | head -1 | cut -d= -f2-
    fi
}

do_install() {
    echo "=== Email IP Notifier Setup ==="
    echo ""

    if [ "$EUID" -ne 0 ]; then
        echo "Note: Run with sudo to install the systemd service"
        echo ""
    fi

    # No pip dependencies needed by default - stdlib only.
    # Optional: if you set "include_battery": true in config.json to read
    # the Waveshare UPS HAT (B), install smbus2 first:
    #     sudo apt install python3-smbus2     # or: pip install smbus2
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
        OLD_DIR=$(detect_installed_dir)
        if [ -n "$OLD_DIR" ] && [ "$OLD_DIR" != "$SCRIPT_DIR" ]; then
            echo "  Detected previous install at: $OLD_DIR"
            echo "  Current location is:          $SCRIPT_DIR"
            echo "  Re-stamping service with the new path."
        fi

        sed -e "s|INSTALL_DIR|$SCRIPT_DIR|g" -e "s|INSTALL_USER|$ACTUAL_USER|g" \
            "$SCRIPT_DIR/$SERVICE_FILE" > "$SERVICE_INSTALL_PATH"

        systemctl daemon-reload
        systemctl enable "$SERVICE_FILE"
        echo "  Service installed and enabled!"
        echo ""
        echo "  Commands:"
        echo "    Test now:      sudo systemctl start $SERVICE_FILE"
        echo "    Check status:  sudo systemctl status $SERVICE_FILE"
        echo "    View logs:     journalctl -u $SERVICE_FILE"
        echo "    Uninstall:     sudo bash $0 uninstall"
    else
        echo "  Skipped (run with sudo to install service)"
        echo ""
        echo "  To install:"
        echo "    sudo bash $0"
    fi

    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "Next steps:"
    echo "  1. Edit config.json with your SMTP credentials"
    echo "  2. For Gmail: generate an app password at https://myaccount.google.com/apppasswords"
    echo "  3. Test with: python3 $SCRIPT_DIR/send_ip_email.py"
    echo ""
}

do_uninstall() {
    echo "=== Email IP Notifier Uninstall ==="
    echo ""

    if [ "$EUID" -ne 0 ]; then
        echo "Error: uninstall requires sudo"
        echo "Re-run with: sudo bash $0 uninstall"
        exit 1
    fi

    if [ ! -f "$SERVICE_INSTALL_PATH" ]; then
        echo "No service file at $SERVICE_INSTALL_PATH -- nothing to remove."
        exit 0
    fi

    OLD_DIR=$(detect_installed_dir)
    echo "Removing service installed from: ${OLD_DIR:-<unknown>}"

    systemctl stop "$SERVICE_FILE" 2>/dev/null || true
    systemctl disable "$SERVICE_FILE" 2>/dev/null || true
    rm -f "$SERVICE_INSTALL_PATH"
    systemctl daemon-reload

    echo "Service stopped, disabled, and removed."
    echo ""
    echo "Left intact (delete manually if you want a full teardown):"
    echo "  $SCRIPT_DIR/config.json"
    echo "  $SCRIPT_DIR/send_ip_email.py"
}

case "$COMMAND" in
    install|"")
        do_install
        ;;
    uninstall|remove)
        do_uninstall
        ;;
    help|-h|--help)
        print_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        print_usage
        exit 1
        ;;
esac
