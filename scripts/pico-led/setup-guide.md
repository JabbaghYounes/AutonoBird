# Pico 2 W + Waveshare RGB Mini LED Array Setup

## Step 1: Verify USB Connection

A brand-new Pico ships with no firmware — nothing will light up until you flash it.
The most common issue is a **charge-only USB cable** (no data lines).

1. **Unplug** the Pico from USB
2. **Hold the BOOTSEL button** (small white button on the Pico board)
3. **While holding BOOTSEL**, plug the USB cable back in
4. **Release BOOTSEL** after ~1 second

This forces the Pico into mass storage / bootloader mode. Verify detection:

```bash
lsusb | grep -i 2e8a
```

- **Not detected** → swap for a known data-capable USB cable and retry
- **Detected** → a `RPI-RP2` drive should mount automatically

## Step 2: Flash MicroPython

1. Download the MicroPython UF2 for **Pico 2 W** from:
   https://micropython.org/download/RPI_PICO2_W/
2. Copy the `.uf2` file onto the `RPI-RP2` mounted drive
3. The Pico will reboot automatically
4. After reboot it should appear as `/dev/ttyACM0`

## Step 3: Connect via Serial

```bash
# install mpremote if needed
pip install mpremote

# connect to the Pico REPL
mpremote connect /dev/ttyACM0
```

## Step 4: Test the LEDs

The Waveshare Pico RGB Mini LED array uses **WS2812 (NeoPixel)** LEDs on **GPIO6**
(verify the pin on your board's silkscreen/docs).

From the MicroPython REPL:

```python
import machine, neopixel

np = neopixel.NeoPixel(machine.Pin(6), 4)  # 4 LEDs on GPIO6
np[0] = (255, 0, 0)    # red
np[1] = (0, 255, 0)    # green
np[2] = (0, 0, 255)    # blue
np[3] = (255, 255, 0)  # yellow
np.write()
```
