"""
scripts/pico-led/main.py — Pico 2 W MicroPython firmware

Drives the WS2812 NeoPixel array (GPIO6) from FSM state commands sent over
USB-serial by the Pi-side `led_bridge.py`. Protocol is line-based ASCII:

    STATE <flight_state>\\n        set the LED pattern for the given FSM state
    OFF\\n                         all LEDs off
    BRIGHTNESS <0-255>\\n          set max brightness scaling factor
    PING\\n                        Pico replies PONG\\n  (health check)

Pico replies on boot with `READY\\n` so the Pi-side bridge can confirm the
firmware is alive before sending commands.

States mirror scripts/autonomy/state_machine.FlightState. Color + animation
choices are aviation-leaning and glance-readable:

    DISCONNECTED          off
    NO_FIX                red, slow blink (1 Hz) — waiting for GPS
    PREARMED              green, steady — safe / ready
    ARMED_ON_GROUND       red, steady — hot / armed
    ASCENDING             cyan, scrolling up — climb in progress
    AIRBORNE              blue, steady — in flight
    DESCENDING            cyan, scrolling down — descent in progress
    DISARMED_POSTFLIGHT   green, slow pulse — safely landed
    FAULT                 red, fast blink (4 Hz) — alert

Place this file at the root of the Pico filesystem (the MicroPython REPL's
`/`). It runs at boot.
"""

import sys
import time
import uselect
import machine
import neopixel


# ----------------- configuration ----------------- #

LED_COUNT = 4
GPIO_PIN = 6
DEFAULT_BRIGHTNESS = 48   # /255 — comfortable indoor
TICK_MS = 33              # ~30 fps animation tick

# Built-in LED for "I'm alive" heartbeat.
ONBOARD_LED = machine.Pin("LED", machine.Pin.OUT)


# ----------------- LED helpers ----------------- #

np = neopixel.NeoPixel(machine.Pin(GPIO_PIN), LED_COUNT)
brightness = DEFAULT_BRIGHTNESS


def scale(rgb):
    r, g, b = rgb
    return (
        (r * brightness) // 255,
        (g * brightness) // 255,
        (b * brightness) // 255,
    )


def fill(rgb):
    s = scale(rgb)
    for i in range(LED_COUNT):
        np[i] = s
    np.write()


def clear():
    for i in range(LED_COUNT):
        np[i] = (0, 0, 0)
    np.write()


# ----------------- state machine ----------------- #

current_state = "DISCONNECTED"
anim_t0 = time.ticks_ms()


def render(state, elapsed_ms):
    """Update LEDs for `state` given `elapsed_ms` since this state's start."""
    if state == "DISCONNECTED":
        clear()

    elif state == "NO_FIX":
        # red, 1 Hz blink (500 ms on, 500 ms off)
        on = (elapsed_ms // 500) % 2 == 0
        fill((255, 0, 0) if on else (0, 0, 0))

    elif state == "PREARMED":
        fill((0, 255, 0))

    elif state == "ARMED_ON_GROUND":
        fill((255, 0, 0))

    elif state == "ASCENDING":
        # cyan, scrolling up — one bright LED moves from bottom to top, ~250 ms / LED
        clear()
        pos = (elapsed_ms // 200) % LED_COUNT
        np[pos] = scale((0, 200, 255))
        # leave a dim trail one below
        trail = (pos - 1) % LED_COUNT
        np[trail] = scale((0, 60, 80))
        np.write()

    elif state == "AIRBORNE":
        fill((0, 80, 255))

    elif state == "DESCENDING":
        # cyan, scrolling down — opposite of ASCENDING
        clear()
        pos = (LED_COUNT - 1 - (elapsed_ms // 200) % LED_COUNT)
        np[pos] = scale((0, 200, 255))
        trail = (pos + 1) % LED_COUNT
        np[trail] = scale((0, 60, 80))
        np.write()

    elif state == "DISARMED_POSTFLIGHT":
        # green, slow pulse 0.25 Hz — 4 s cycle
        phase = (elapsed_ms % 4000) / 4000.0
        # triangle wave
        if phase < 0.5:
            level = phase * 2
        else:
            level = (1.0 - phase) * 2
        v = int(level * 255)
        fill((0, v, 0))

    elif state == "FAULT":
        # red, fast blink 4 Hz (125 ms on, 125 ms off)
        on = (elapsed_ms // 125) % 2 == 0
        fill((255, 0, 0) if on else (0, 0, 0))

    else:
        # unknown state — magenta steady (visible "huh?" signal)
        fill((200, 0, 200))


def set_state(new_state):
    global current_state, anim_t0
    if new_state != current_state:
        current_state = new_state
        anim_t0 = time.ticks_ms()


# ----------------- serial command parser ----------------- #

poll = uselect.poll()
poll.register(sys.stdin, uselect.POLLIN)


def handle_line(line):
    line = line.strip()
    if not line:
        return
    parts = line.split(None, 1)
    cmd = parts[0].upper()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "STATE":
        set_state(arg.strip().upper())
    elif cmd == "OFF":
        set_state("DISCONNECTED")
    elif cmd == "BRIGHTNESS":
        global brightness
        try:
            v = int(arg)
            if 0 <= v <= 255:
                brightness = v
        except ValueError:
            pass
    elif cmd == "PING":
        print("PONG")
    # silently ignore unknown commands so a casual REPL user can't break us


# ----------------- main loop ----------------- #


def main():
    print("READY")
    last_blink = time.ticks_ms()
    last_tick = time.ticks_ms()

    while True:
        now = time.ticks_ms()

        # non-blocking line read
        if poll.poll(0):
            try:
                line = sys.stdin.readline()
                if line:
                    handle_line(line)
            except Exception:
                pass  # robustness: don't die on input quirks

        # animation tick
        if time.ticks_diff(now, last_tick) >= TICK_MS:
            elapsed = time.ticks_diff(now, anim_t0)
            render(current_state, elapsed)
            last_tick = now

        # onboard LED heartbeat (1 Hz) so we can tell at a glance that
        # main.py is alive even if the NeoPixels are off (DISCONNECTED).
        if time.ticks_diff(now, last_blink) >= 1000:
            ONBOARD_LED.toggle()
            last_blink = now

        time.sleep_ms(5)


try:
    main()
except KeyboardInterrupt:
    # Allow mpremote Ctrl-C to drop back to the REPL during development.
    clear()
    raise
