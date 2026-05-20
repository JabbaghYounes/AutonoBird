# AR0144 Stereo Calibration — Testing Both Methods

## Prerequisites (shared)

1. **Checkerboard** — print a 7x5 inner corners pattern with 26 mm squares (the board AutonoBird was actually calibrated against — a print-shop scaled the A3 design down to A4). Mount on something rigid (cardboard, clipboard). `SQUARE_SIZE_MM = 26.0` in both calibration scripts; do not change without re-printing.
2. **Camera** — Waveshare AR0144 stereo USB camera plugged in
3. **Dependencies**: this subsystem runs on system Python (no venv) and depends on the apt-installed `python3-opencv` package plus `numpy`. If you need to install manually: `pip install opencv-python numpy`.
4. **Verify camera is detected**:
```bash
cd ~/Documents/AutonoBird/scripts/ar0144
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
```

If that prints `FAIL`, try index `1` or `2` and update `CAMERA_INDEX` in both scripts.

## Method 1: Manual (`basic_calibration.py`)

```bash
python3 basic_calibration.py capture      # freeform — you decide where to hold the board
python3 basic_calibration.py calibrate    # compute calibration
python3 basic_calibration.py depth        # live depth viewer
```

During capture:
- You see a combined left|right preview
- Position the checkerboard so it's detected in **both** cameras (green corners)
- Press **SPACE** to save a pair — aim for 15-20 pairs
- Vary position, angle, distance, and corners of the frame yourself
- Press **Q** when done

## Method 2: Guided (`guided_calibration.py`)

```bash
python3 guided_calibration.py capture      # follows 40-pose guided sequence
python3 guided_calibration.py calibrate    # compute calibration
python3 guided_calibration.py depth        # live depth viewer
```

During capture:
- The screen shows a **target zone** (orange rectangle) telling you where to hold the board
- Move the board into the zone at the right distance — it turns **green** when you're in position
- Hold steady ~1 second and it **auto-captures**
- Press **S** to skip a pose, **Q** to quit early
- It walks through all 4 zones (NEAR → MID-NEAR → MID → FAR)

## Comparing Results

Both scripts save to the same `stereo_calibration_data/` directory and file, so clear between runs:

```bash
rm -rf stereo_calibration_data/    # wipe previous calibration
```

The key metric to compare is the **RMS reprojection error** printed after calibration:
- **< 0.5** — excellent
- **0.5–1.0** — good
- **> 1.0** — poor, recapture

Then in the depth viewer, check whether the epipolar lines align across left/right and whether the depth map looks reasonable.
