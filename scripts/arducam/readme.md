# Stereo Depth Mapping — Raspberry Pi 4 + Arducam IMX519

Stereo depth estimation using two IMX519 16MP cameras on the Arducam Synchronized Quad-Camera Kit, spaced 75mm apart.

## Hardware Setup

- Raspberry Pi 4 (4GB+ RAM recommended)
- Arducam IMX519 16MP Autofocus Synchronized Quad-Camera Kit
- Two cameras connected to **Channel 0** (left) and **Channel 1** (right)
- Cameras mounted **75mm apart**, parallel, lenses facing the same direction
- A printed **checkerboard pattern** (9×6 inner corners, 25mm squares) for calibration

## Software Setup

```bash
# System dependencies
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv i2c-tools python3-smbus2

# Python packages (if not using system OpenCV)
pip3 install opencv-contrib-python-headless numpy --break-system-packages

# Enable I2C (needed for camera multiplexer)
sudo raspi-config nonint do_i2c 0

# Verify cameras are detected
libcamera-hello --list-cameras
```

### Important: opencv-contrib

The WLS disparity filter requires `opencv-contrib`. If you get an error about `cv2.ximgproc`, either:
- Install `opencv-contrib-python-headless`, or
- Set `"wls_filter": False` in the CONFIG dict

## Usage

### Step 1 — Calibrate

Print a checkerboard (9×6 inner corners, 25mm squares). Hold it in front of both cameras at various angles and distances:

```bash
python3 stereo_depth.py calibrate
```

- Press **SPACE** when the board is detected in both views (green overlay)
- Capture at least **15-20 pairs** at different positions, angles, and distances
- Fill the frame — get the board in corners and edges, not just the center
- Vary the distance from ~30cm to ~1.5m

### Step 2 — Compute Calibration

```bash
python3 stereo_depth.py compute_cal
```

This computes intrinsics, extrinsics, and rectification maps. Look for:
- **Stereo RMS error < 0.5** — excellent
- **RMS 0.5–1.0** — acceptable
- **RMS > 1.0** — recapture calibration images

### Step 3 — Run Depth Mapping

```bash
python3 stereo_depth.py depth
```

| Key | Action |
|-----|--------|
| `S` | Save current frame + depth map |
| `D` | Toggle disparity / depth view |
| `R` | Toggle epipolar lines (verify rectification) |
| `+`/`-` | Adjust max depth range |
| `Q`/`ESC` | Quit |

### Single Capture

```bash
python3 stereo_depth.py capture
```

Saves raw images, rectified images, colorized depth, and raw depth as `.npy`.

## Loading Depth Data in Your Own Code

```python
import numpy as np

# Load depth map (values in millimeters)
depth_mm = np.load("depth_output/depth_mm_20250210_143022.npy")

# Get depth at a specific pixel
x, y = 640, 480
distance = depth_mm[y, x]  # in mm
print(f"Distance at ({x},{y}): {distance:.0f}mm = {distance/1000:.2f}m")

# Find all points within 1 meter
close_mask = (depth_mm > 0) & (depth_mm < 1000)
```

## Tuning Guide

Edit the `CONFIG` dict at the top of `stereo_depth.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_disparities` | 128 | Higher = detects farther objects (must be ×16) |
| `block_size` | 9 | Larger = smoother but less detail (must be odd) |
| `use_sgbm` | True | SGBM is slower but much better quality than BM |
| `sgbm_mode` | "hh" | "hh" = best quality, "3way" = faster |
| `wls_filter` | True | Smooths depth, fills holes (needs opencv-contrib) |
| `wls_lambda` | 8000 | Higher = smoother depth |
| `wls_sigma` | 1.5 | Edge sensitivity (1.0–2.0) |
| `uniqueness_ratio` | 10 | Higher = fewer false matches, more holes |
| `speckle_window_size` | 100 | Removes small noise blobs |

### Tips for Better Depth

- **Good lighting** is critical — even, diffuse light works best
- **Textured scenes** produce better matches than blank walls
- The **effective range** with 75mm baseline is roughly 0.3m–5m
- For closer objects, reduce `num_disparities`; for farther, increase it
- If depth is noisy, increase `block_size` or enable WLS filter
- Verify rectification with the `R` key — horizontal lines should align across both images

## Adapter I2C Configuration

If camera switching fails, you may need to adjust the I2C bus and address in the `_switch_channel` method. Common values:

| Adapter Version | I2C Bus | Address |
|----------------|---------|---------|
| Quad-Camera HAT | 10 | 0x24 |
| UC-512 Rev B | 1 | 0x24 |
| Older models | 0 | 0x70 |

Check with: `i2cdetect -y 10` (or `-y 1`, `-y 0`)

## File Structure

```
stereo_depth/
├── stereo_depth.py            # Main program
├── stereo_calibration.json    # Calibration parameters (generated)
├── stereo_rectify_maps.npz    # Rectification lookup tables (generated)
├── calibration_images/        # Checkerboard captures (generated)
│   ├── stereo_000_left.png
│   ├── stereo_000_right.png
│   └── ...
└── depth_output/              # Saved depth frames (generated)
    ├── depth_color_*.png
    ├── depth_mm_*.npy
    └── ...
```
