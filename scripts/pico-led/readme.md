# Pico LED bridge

Status indicator subsystem. A Raspberry Pi Pico 2 W (running MicroPython)
drives a WS2812 NeoPixel array; the Pi 5 publishes flight-state events
from `scripts/autonomy/state_machine.StateMachine` over USB-serial, and
the Pico paints the right colour + animation per state.

```
┌────────────┐      MAVLink       ┌────────────┐
│  Pixhawk / │  ───────────────▶  │   Pi 5     │
│  SITL      │                    │   bridge   │
└────────────┘                    │  + FSM     │
                                  └─────┬──────┘
                                        │ STATE <name>\n  (USB-serial)
                                        ▼
                                  ┌────────────┐
                                  │  Pico 2 W  │
                                  │  + WS2812  │
                                  └────────────┘
```

## Files

| File | Purpose |
|---|---|
| `main.py` | MicroPython firmware — flashed onto the Pico. Reads STATE / OFF / BRIGHTNESS / PING commands from USB-serial; renders animations on the WS2812 array (GPIO6). |
| `led_bridge.py` | Pi-side driver. Pure Python with one runtime dep (`pyserial`). Exposes a `LedBridge` class for the orchestrator + a CLI for standalone testing. |
| `setup.sh` | Builds the Pi-side venv (just pyserial) and seeds config.json. |
| `setup-guide.md` | One-time Pico flashing walkthrough (BOOTSEL → UF2 → MicroPython REPL). |
| `config.example.json` | Defaults — serial port, baud, brightness. |

## One-time setup

### Pi-side venv

```bash
cd scripts/pico-led
bash setup.sh
```

Creates `venv/` with `pyserial>=3.5` and seeds `config.json` from the
template.

### Pico-side firmware

1. Follow `setup-guide.md` once to flash MicroPython onto the Pico.
2. Copy `main.py` onto the Pico's filesystem:

```bash
mpremote connect /dev/ttyACM0 fs cp main.py :main.py
mpremote connect /dev/ttyACM0 soft-reset
```

After the soft reset the Pico runs `main.py` automatically on every boot.
It prints `READY` on the USB-serial channel, then idles on the
`DISCONNECTED` state (LEDs off; the onboard green LED blinks at 1 Hz as a
liveness indicator).

## State → colour mapping

The Pico knows the nine FSM states defined by
`scripts/autonomy/state_machine.FlightState`:

| State | Colour | Animation | Reading |
|---|---|---|---|
| `DISCONNECTED` | — | off | no autopilot link |
| `NO_FIX` | red | 1 Hz blink | waiting for GPS |
| `PREARMED` | green | steady | safe + ready to arm |
| `ARMED_ON_GROUND` | red | steady | armed, hot |
| `ASCENDING` | cyan | scrolling up | climb in progress |
| `AIRBORNE` | blue | steady | in flight |
| `DESCENDING` | cyan | scrolling down | descent in progress |
| `DISARMED_POSTFLIGHT` | green | slow pulse | safely landed |
| `FAULT` | red | 4 Hz blink | crash / failsafe alert |

Brightness scales the whole palette. Default 48/255; bump higher for
daylight. Set via `led.set_brightness(value)` or the `BRIGHTNESS <0-255>`
serial command.

## Usage

### Standalone CLI (no SITL / FSM needed)

```bash
source venv/bin/activate

# Cycle through all 9 states, 2 s each. Confirms firmware + cable.
python led_bridge.py --test

# Health check — should print PONG.
python led_bridge.py --ping

# Force one state and exit.
python led_bridge.py --state ASCENDING

# Lights off.
python led_bridge.py --off
```

### Live FSM mirror (needs SITL or real Pixhawk)

With SITL running (or the real Pixhawk connected):

```bash
python led_bridge.py --watch-fsm
```

Subscribes to `scripts/autonomy/state_machine.StateMachine` via
`sys.path` injection (same cross-subsystem pattern the other autonomy
tests use), mirrors every transition to the Pico, runs until Ctrl-C.

### Imported by the orchestrator

```python
from led_bridge import LedBridge

with LedBridge(port="/dev/ttyACM0") as led:
    led.set_brightness(64)
    fsm.subscribe(lambda old, new, reason: led.set_state(new.name))
    # ... main loop ...
```

## Protocol

Line-based ASCII over USB-serial (115200, but USB CDC ignores baud). One
command per line, `\n` terminator.

| Pi → Pico | Pico response |
|---|---|
| `STATE <name>\n` | (none — applied silently) |
| `OFF\n` | (none) |
| `BRIGHTNESS <0-255>\n` | (none) |
| `PING\n` | `PONG\n` |
| (any malformed line) | silently ignored |

Pico → Pi unsolicited messages:

| Message | When |
|---|---|
| `READY\n` | once at boot, after `main.py` finishes startup |

The protocol is intentionally human-readable so you can drive the Pico
from any serial terminal (`screen /dev/ttyACM0 115200`, `minicom`, etc.)
for debugging without the bridge in the loop.

## Debugging

- **`led_bridge.py --ping` returns "(no response)"**: the firmware isn't
  running. Check `mpremote connect /dev/ttyACM0` opens a REPL, and that
  `main.py` is present on the Pico (`mpremote fs ls :`).
- **LEDs stuck on magenta steady**: the Pico received an unknown state
  name. Verify spelling — must match a row in the table above exactly.
- **Pico shows up at a different `/dev/ttyACM*`**: the Pixhawk USB also
  enumerates as `/dev/ttyACM*` on the Pi. Use a stable
  `/dev/serial/by-id/...` symlink in `config.json` to avoid collisions
  when both are plugged in.
- **Pico onboard green LED not blinking**: `main.py` crashed or never
  ran. Reconnect via `mpremote connect /dev/ttyACM0` and inspect the
  REPL output / re-run with `mpremote run main.py` to see the traceback.
