#!/usr/bin/env python3
"""
Raspberry Pi 5 Startup Script - Send IP Address via Email

This script sends the Pi's IP address to a specified email address
when the Pi boots up. Uses Python's built-in smtplib (no external dependencies).

Setup:
1. Create config.json from config.example.json
2. Set your SMTP credentials (e.g. Gmail app password)
3. Set up as systemd service (see setup.sh)
"""

import json
import socket
import subprocess
import smtplib
import time
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path


def get_local_ip():
    """Get the local IP address of the Pi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def get_all_ips():
    """Get all network interface IPs."""
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip().split()
    except Exception:
        return []


def get_hostname():
    """Get the hostname of the Pi."""
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def get_wifi_ssid():
    """Get the name of the connected WiFi network."""
    try:
        result = subprocess.run(
            ["iwgetid", "-r"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def get_external_ip():
    """Get external/public IP address."""
    try:
        result = subprocess.run(
            ["curl", "-s", "https://api.ipify.org"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def load_config():
    """Load configuration from config.json."""
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print("Please create config.json based on config.example.json")
        exit(1)

    with open(config_path, "r") as f:
        return json.load(f)


def send_ip_email(config):
    """Send IP address notification via email."""
    hostname = get_hostname()
    local_ip = get_local_ip()
    all_ips = get_all_ips()
    wifi_ssid = get_wifi_ssid()
    external_ip = get_external_ip() if config.get("include_external_ip", False) else None
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"Raspberry Pi Boot Notification\n"
    body += f"{'=' * 40}\n\n"
    body += f"Hostname:    {hostname}\n"
    body += f"Time:        {timestamp}\n"
    body += f"Local IP:    {local_ip or 'Not available'}\n"

    if wifi_ssid:
        body += f"WiFi:        {wifi_ssid}\n"

    if len(all_ips) > 1:
        body += f"All IPs:     {', '.join(all_ips)}\n"

    if external_ip:
        body += f"External IP: {external_ip}\n"

    body += f"\nSSH: ssh {config.get('ssh_user', 'pi')}@{local_ip}\n"

    msg = MIMEText(body)
    device_name = config.get("device_name", "")
    if device_name:
        msg["Subject"] = f"{device_name} ({hostname}) booted - {local_ip}"
    else:
        msg["Subject"] = f"Pi ({hostname}) booted - {local_ip}"
    msg["From"] = config["smtp_user"]
    msg["To"] = config["recipient_email"]

    with smtplib.SMTP(config["smtp_host"], config.get("smtp_port", 587)) as server:
        server.starttls()
        server.login(config["smtp_user"], config["smtp_password"])
        server.send_message(msg)

    print(f"Successfully sent IP notification to {config['recipient_email']}")


def main():
    """Main entry point."""
    print("Starting IP notification script...")

    # Wait for network to be ready (important for startup)
    time.sleep(5)

    config = load_config()

    required_keys = ["smtp_host", "smtp_user", "smtp_password", "recipient_email"]
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing '{key}' in config.json")
            exit(1)

    send_ip_email(config)


if __name__ == "__main__":
    main()
