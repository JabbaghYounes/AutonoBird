#!/usr/bin/env python3
"""
Raspberry Pi 5 Startup Script - Send IP Address via Discord Bot

This script sends the Pi's IP address to a specified Discord user via DM
when the Pi boots up.

Setup:
1. Create a Discord bot at https://discord.com/developers/applications
2. Enable "Message Content Intent" in Bot settings
3. Copy the bot token
4. Get your Discord user ID (Enable Developer Mode, right-click your name, Copy ID)
5. Create config.json with your credentials (see config.example.json)
6. Install dependency: pip3 install discord.py
7. Set up as systemd service (see setup instructions below)
"""

import json
import socket
import subprocess
import asyncio
import os
from datetime import datetime
from pathlib import Path

try:
    import discord
except ImportError:
    print("Error: discord.py not installed. Run: pip3 install discord.py")
    exit(1)


def get_local_ip():
    """Get the local IP address of the Pi."""
    try:
        # Connect to external server to determine local IP
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


async def send_ip_notification(config):
    """Send IP address notification via Discord bot DM."""

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Bot logged in as {client.user}")

        try:
            # Get the user to DM
            user_id = int(config["discord_user_id"])
            user = await client.fetch_user(user_id)

            if user is None:
                print(f"Error: Could not find user with ID {user_id}")
                await client.close()
                return

            # Gather IP information
            hostname = get_hostname()
            local_ip = get_local_ip()
            all_ips = get_all_ips()
            external_ip = get_external_ip() if config.get("include_external_ip", False) else None
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Build the message
            message = f"**Raspberry Pi Boot Notification**\n"
            message += f"```\n"
            message += f"Hostname:    {hostname}\n"
            message += f"Time:        {timestamp}\n"
            message += f"Local IP:    {local_ip or 'Not available'}\n"

            if len(all_ips) > 1:
                message += f"All IPs:     {', '.join(all_ips)}\n"

            if external_ip:
                message += f"External IP: {external_ip}\n"

            message += f"```\n"
            message += f"SSH: `ssh {config.get('ssh_user', 'pi')}@{local_ip}`"

            # Send DM
            await user.send(message)
            print(f"Successfully sent IP notification to user {user_id}")

        except discord.Forbidden:
            print("Error: Bot cannot DM this user. Make sure:")
            print("  1. The bot shares a server with the user")
            print("  2. The user has DMs enabled for that server")
        except Exception as e:
            print(f"Error sending message: {e}")
        finally:
            await client.close()

    try:
        await client.start(config["bot_token"])
    except discord.LoginFailure:
        print("Error: Invalid bot token")
        exit(1)


def main():
    """Main entry point."""
    print("Starting IP notification script...")

    # Wait for network to be ready (important for startup)
    import time
    time.sleep(5)

    config = load_config()

    # Validate config
    required_keys = ["bot_token", "discord_user_id"]
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing '{key}' in config.json")
            exit(1)

    asyncio.run(send_ip_notification(config))


if __name__ == "__main__":
    main()
