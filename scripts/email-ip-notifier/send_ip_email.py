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


def get_battery_status():
    """Read battery status from a Waveshare UPS HAT (B) over I2C.

    Returns a dict with keys 'percent', 'voltage' (V), 'current' (A),
    'state' ('charging' / 'discharging' / 'unknown'), or None if smbus2
    isn't installed, the I2C bus is unavailable, or the HAT doesn't
    respond at the expected address.

    Caller is expected to gate this behind the include_battery config
    flag so Pis without the HAT incur no overhead.
    """
    try:
        from smbus2 import SMBus
    except ImportError:
        return None

    # INA219 registers + config for the UPS HAT (B). Values mirror the
    # Waveshare datasheet + the working UPSentinel implementation.
    REG_CONFIG = 0x00
    REG_SHUNT_VOLTAGE = 0x01
    REG_BUS_VOLTAGE = 0x02
    REG_CALIBRATION = 0x05
    CONFIG_VALUE = 0x399F           # 32 V bus, +/- 320 mV shunt, 12-bit ADC, continuous
    CAL_VALUE = 4096                # for 0.1 mA current LSB with 0.1 ohm shunt
    SHUNT_RESISTANCE = 0.1          # ohms
    I2C_BUS = 1
    I2C_ADDR = 0x42

    # 2S Li-ion discharge curve (volts -> percent). Piecewise-linear.
    VOLTAGE_CURVE = [
        (6.0, 0), (6.4, 5), (6.8, 10), (7.0, 20),
        (7.2, 40), (7.4, 60), (7.6, 80), (7.9, 90),
        (8.2, 95), (8.4, 100),
    ]
    CURRENT_NOISE_THRESHOLD = 0.005  # 5 mA below which state is "unknown"

    def _read_word(bus, addr, reg):
        # INA219 sends big-endian; smbus2 returns little-endian
        data = bus.read_word_data(addr, reg)
        return ((data & 0xFF) << 8) | ((data >> 8) & 0xFF)

    def _write_word(bus, addr, reg, value):
        swapped = ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)
        bus.write_word_data(addr, reg, swapped)

    def _voltage_to_percent(v):
        if v <= VOLTAGE_CURVE[0][0]:
            return 0
        if v >= VOLTAGE_CURVE[-1][0]:
            return 100
        for i in range(1, len(VOLTAGE_CURVE)):
            v_low, p_low = VOLTAGE_CURVE[i - 1]
            v_high, p_high = VOLTAGE_CURVE[i]
            if v <= v_high:
                ratio = (v - v_low) / (v_high - v_low)
                return int(p_low + ratio * (p_high - p_low))
        return 100

    try:
        with SMBus(I2C_BUS) as bus:
            _write_word(bus, I2C_ADDR, REG_CONFIG, CONFIG_VALUE)
            _write_word(bus, I2C_ADDR, REG_CALIBRATION, CAL_VALUE)

            # Bus voltage register: top 13 bits are the value, LSB = 4 mV
            raw_bus = _read_word(bus, I2C_ADDR, REG_BUS_VOLTAGE)
            voltage = (raw_bus >> 3) * 0.004

            # Shunt voltage: signed 16-bit, LSB = 10 uV
            raw_shunt = _read_word(bus, I2C_ADDR, REG_SHUNT_VOLTAGE)
            if raw_shunt & 0x8000:
                raw_shunt -= 0x10000
            current = (raw_shunt * 0.00001) / SHUNT_RESISTANCE

            percent = _voltage_to_percent(voltage)
            if current > CURRENT_NOISE_THRESHOLD:
                state = "charging"
            elif current < -CURRENT_NOISE_THRESHOLD:
                state = "discharging"
            else:
                state = "unknown"

            return {
                "percent": percent,
                "voltage": round(voltage, 2),
                "current": round(current, 3),
                "state": state,
            }
    except (FileNotFoundError, OSError):
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
    battery = get_battery_status() if config.get("include_battery", False) else None
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

    if battery:
        body += f"Battery:     {battery['percent']}% ({battery['voltage']} V, {battery['state']})\n"

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
