#!/usr/bin/env python3
"""
=============================================================================
Waveshare AR0144 Stereo Camera — GUIDED Calibration (Face-Scan Style)
=============================================================================

Instead of the user guessing where to hold the checkerboard, this script
guides them through a sequence of poses — similar to how phone apps guide
you during facial recognition setup.

The screen shows:
    - A target zone (rectangle) where the checkerboard should appear
    - An arrow/instruction telling the user which direction to move
    - A progress ring showing completion
    - Auto-capture when the board is detected in the right zone

40 poses across 4 distance zones (10 poses each), optimized for
drone obstacle avoidance at 0.3–4 m operational range:

    Zone 1: NEAR    (0.3–0.6 m)  — precision close-range depth
    Zone 2: MID-NEAR (0.6–1.2 m) — moderate-speed obstacle avoidance
    Zone 3: MID     (1.2–2.5 m)  — critical navigation range
    Zone 4: FAR     (2.5–4.0 m)  — early obstacle detection

Each zone includes flat, yaw, pitch, roll, position, and combo poses
for strong 3D parameter coverage across the full depth range.

Usage:
    python guided_calibration.py capture     # Guided capture
    python guided_calibration.py calibrate   # Run calibration
    python guided_calibration.py depth       # Live depth viewer
    python guided_calibration.py all         # Full pipeline

Requirements:
    pip install opencv-python numpy
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import glob
import time
import math
import subprocess

# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkerboard inner corners (cols x rows). For a 10x7-square board,
# inner corners are 9x6. Match this to whichever printout you use.
CHECKERBOARD_COLS = 9
CHECKERBOARD_ROWS = 6
CHECKERBOARD = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
# Set to whatever you actually measured on the printed board.
# Reference values: A2 50mm board=50.0, A3 40mm board=40.0,
# A3-design auto-scaled-to-A4 (printer "fit to page")=25.0.
SQUARE_SIZE_MM = 26.0

CAMERA_INDEX = 0
CALIB_DIR = "stereo_calibration_data"
SESSIONS_DIR = os.path.join(CALIB_DIR, "sessions")
CALIB_FILE = os.path.join(CALIB_DIR, "stereo_calibration.npz")
BASELINE_M = 0.052

# SGBM parameters
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 128
SGBM_BLOCK_SIZE = 7

# =============================================================================
# GUIDED POSE DEFINITIONS
# =============================================================================
# Each pose defines:
#   - name: display instruction
#   - region: (x%, y%, w%, h%) of the LEFT image where board should appear
#   - icon: arrow direction for the user
#   - detail: extra tip shown on screen
#   - min_area_pct / max_area_pct: expected board size as % of frame area
#     (controls near/far guidance)

POSES = [
    # =========================================================================
    # ZONE 1: NEAR (0.3–0.6 m) — precision tuning for close-range depth
    # =========================================================================
    {
        "name": "[NEAR] Centered, Flat",
        "instruction": "Hold board in CENTER, flat to camera",
        "detail": "About 30-60 cm away",
        "region": (0.10, 0.10, 0.80, 0.80),
        "icon": "center",
        "min_area_pct": 25.0,
        "max_area_pct": 80.0,
    },
    {
        "name": "[NEAR] Tilted Upward ~20°",
        "instruction": "Hold CLOSE, tilt TOP of board AWAY from camera",
        "detail": "About 30-60 cm — top edge further than bottom",
        "region": (0.15, 0.05, 0.70, 0.80),
        "icon": "tilt-up",
        "min_area_pct": 25.0,
        "max_area_pct": 75.0,
    },
    {
        "name": "[NEAR] Tilted Downward ~20°",
        "instruction": "Hold CLOSE, tilt BOTTOM of board AWAY from camera",
        "detail": "About 30-60 cm — bottom edge further than top",
        "region": (0.15, 0.15, 0.70, 0.80),
        "icon": "tilt-down",
        "min_area_pct": 25.0,
        "max_area_pct": 75.0,
    },
    {
        "name": "[NEAR] Yaw Left ~25°",
        "instruction": "Hold CLOSE, rotate LEFT edge toward camera",
        "detail": "About 30-60 cm — left side closer to camera",
        "region": (0.10, 0.10, 0.75, 0.80),
        "icon": "tilt-left",
        "min_area_pct": 20.0,
        "max_area_pct": 70.0,
    },
    {
        "name": "[NEAR] Yaw Right ~25°",
        "instruction": "Hold CLOSE, rotate RIGHT edge toward camera",
        "detail": "About 30-60 cm — right side closer to camera",
        "region": (0.15, 0.10, 0.75, 0.80),
        "icon": "tilt-right",
        "min_area_pct": 20.0,
        "max_area_pct": 70.0,
    },
    {
        "name": "[NEAR] Roll Clockwise ~20°",
        "instruction": "Hold CLOSE, ROTATE board clockwise ~20°",
        "detail": "About 30-60 cm — tilt the board like a clock hand",
        "region": (0.10, 0.10, 0.80, 0.80),
        "icon": "center",
        "min_area_pct": 20.0,
        "max_area_pct": 70.0,
    },
    {
        "name": "[NEAR] Roll Counter-Clockwise ~20°",
        "instruction": "Hold CLOSE, ROTATE board counter-clockwise ~20°",
        "detail": "About 30-60 cm — opposite rotation from last pose",
        "region": (0.10, 0.10, 0.80, 0.80),
        "icon": "center",
        "min_area_pct": 20.0,
        "max_area_pct": 70.0,
    },
    {
        "name": "[NEAR] Top-Left of Frame",
        "instruction": "Hold CLOSE, position board in TOP-LEFT",
        "detail": "About 30-60 cm — board in upper-left area",
        "region": (0.0, 0.0, 0.55, 0.55),
        "icon": "top-left",
        "min_area_pct": 20.0,
        "max_area_pct": 65.0,
    },
    {
        "name": "[NEAR] Bottom-Right of Frame",
        "instruction": "Hold CLOSE, position board in BOTTOM-RIGHT",
        "detail": "About 30-60 cm — board in lower-right area",
        "region": (0.45, 0.45, 0.55, 0.55),
        "icon": "bottom-right",
        "min_area_pct": 20.0,
        "max_area_pct": 65.0,
    },
    {
        "name": "[NEAR] Very Close — Fill Frame",
        "instruction": "Move board VERY CLOSE to fill ~80% of the frame",
        "detail": "About 20-30 cm — as close as possible while detected",
        "region": (0.05, 0.05, 0.90, 0.90),
        "icon": "forward",
        "min_area_pct": 50.0,
        "max_area_pct": 90.0,
    },
    # =========================================================================
    # ZONE 2: MID-NEAR (0.6–1.2 m) — obstacle avoidance at moderate speed
    # =========================================================================
    {
        "name": "[MID-NEAR] Centered, Flat",
        "instruction": "Hold board in CENTER, flat to camera",
        "detail": "About 60-120 cm away — roughly arm's length",
        "region": (0.20, 0.15, 0.60, 0.70),
        "icon": "center",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Yaw Left + Slight Up Tilt",
        "instruction": "Yaw board LEFT and tilt TOP slightly away",
        "detail": "About 60-120 cm — combined left rotation + upward tilt",
        "region": (0.10, 0.05, 0.65, 0.70),
        "icon": "top-left",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Yaw Right + Slight Down Tilt",
        "instruction": "Yaw board RIGHT and tilt BOTTOM slightly away",
        "detail": "About 60-120 cm — combined right rotation + downward tilt",
        "region": (0.25, 0.25, 0.65, 0.70),
        "icon": "bottom-right",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Roll + Yaw Combo",
        "instruction": "ROTATE board ~15° AND yaw it slightly right",
        "detail": "About 60-120 cm — combine roll with yaw",
        "region": (0.15, 0.15, 0.70, 0.70),
        "icon": "tilt-right",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Board HIGH in Frame",
        "instruction": "Position board in the UPPER half of the frame",
        "detail": "About 60-120 cm — board near top of view",
        "region": (0.20, 0.0, 0.60, 0.50),
        "icon": "tilt-up",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Board LOW in Frame",
        "instruction": "Position board in the LOWER half of the frame",
        "detail": "About 60-120 cm — board near bottom of view",
        "region": (0.20, 0.50, 0.60, 0.50),
        "icon": "tilt-down",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Board FAR LEFT Edge",
        "instruction": "Position board at the FAR LEFT of the frame",
        "detail": "About 60-120 cm — board at left edge",
        "region": (0.0, 0.15, 0.40, 0.70),
        "icon": "left",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Board FAR RIGHT Edge",
        "instruction": "Position board at the FAR RIGHT of the frame",
        "detail": "About 60-120 cm — board at right edge",
        "region": (0.60, 0.15, 0.40, 0.70),
        "icon": "right",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Diagonal Tilt (TL to BR)",
        "instruction": "Tilt board diagonally — TOP-LEFT corner closer",
        "detail": "About 60-120 cm — diagonal perspective",
        "region": (0.15, 0.15, 0.70, 0.70),
        "icon": "bottom-right",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    {
        "name": "[MID-NEAR] Opposite Diagonal (TR to BL)",
        "instruction": "Tilt board diagonally — TOP-RIGHT corner closer",
        "detail": "About 60-120 cm — opposite diagonal perspective",
        "region": (0.15, 0.15, 0.70, 0.70),
        "icon": "bottom-left",
        "min_area_pct": 8.0,
        "max_area_pct": 30.0,
    },
    # =========================================================================
    # ZONE 3: MID (1.2–2.5 m) — critical for navigation
    # =========================================================================
    {
        "name": "[MID] Center, Flat",
        "instruction": "Hold board in CENTER, flat to camera",
        "detail": "About 1.2-2.5 m away",
        "region": (0.25, 0.20, 0.50, 0.60),
        "icon": "center",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Large Yaw ~30°",
        "instruction": "Hold at MID distance, YAW board ~30° left",
        "detail": "About 1.2-2.5 m — strong left rotation",
        "region": (0.20, 0.15, 0.60, 0.70),
        "icon": "tilt-left",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Large Pitch ~30°",
        "instruction": "Hold at MID distance, PITCH board ~30° upward",
        "detail": "About 1.2-2.5 m — strong upward tilt",
        "region": (0.20, 0.10, 0.60, 0.70),
        "icon": "tilt-up",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Yaw + Roll Combo",
        "instruction": "Hold at MID distance, YAW right + ROLL slightly",
        "detail": "About 1.2-2.5 m — combined rotation",
        "region": (0.20, 0.15, 0.60, 0.70),
        "icon": "tilt-right",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Upper Third of Frame",
        "instruction": "Position board in the UPPER THIRD of the frame",
        "detail": "About 1.2-2.5 m — board near top",
        "region": (0.25, 0.0, 0.50, 0.40),
        "icon": "tilt-up",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Lower Third of Frame",
        "instruction": "Position board in the LOWER THIRD of the frame",
        "detail": "About 1.2-2.5 m — board near bottom",
        "region": (0.25, 0.60, 0.50, 0.40),
        "icon": "tilt-down",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Extreme Left Edge",
        "instruction": "Position board at EXTREME LEFT edge of frame",
        "detail": "About 1.2-2.5 m — board at far left",
        "region": (0.0, 0.20, 0.35, 0.60),
        "icon": "left",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Extreme Right Edge",
        "instruction": "Position board at EXTREME RIGHT edge of frame",
        "detail": "About 1.2-2.5 m — board at far right",
        "region": (0.65, 0.20, 0.35, 0.60),
        "icon": "right",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Perspective Skew",
        "instruction": "Hold at MID distance, one CORNER closer to camera",
        "detail": "About 1.2-2.5 m — creates perspective distortion",
        "region": (0.20, 0.15, 0.60, 0.70),
        "icon": "center",
        "min_area_pct": 2.0,
        "max_area_pct": 12.0,
    },
    {
        "name": "[MID] Small Board — ~30% Coverage",
        "instruction": "Move FURTHER back so board is SMALL in frame",
        "detail": "Near 2-2.5 m — board covers roughly 30% of target zone",
        "region": (0.30, 0.25, 0.40, 0.50),
        "icon": "backward",
        "min_area_pct": 2.0,
        "max_area_pct": 8.0,
    },
    # =========================================================================
    # ZONE 4: FAR (2.5–4 m) — long-range disparity calibration
    # =========================================================================
    {
        "name": "[FAR] Center, Flat",
        "instruction": "Hold board in CENTER, flat to camera",
        "detail": "About 2.5-4 m away — board will look small",
        "region": (0.30, 0.25, 0.40, 0.50),
        "icon": "center",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Yaw Left",
        "instruction": "Hold FAR, YAW board to the LEFT",
        "detail": "About 2.5-4 m — rotate left edge toward camera",
        "region": (0.20, 0.20, 0.50, 0.60),
        "icon": "tilt-left",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Yaw Right",
        "instruction": "Hold FAR, YAW board to the RIGHT",
        "detail": "About 2.5-4 m — rotate right edge toward camera",
        "region": (0.30, 0.20, 0.50, 0.60),
        "icon": "tilt-right",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Pitch Up",
        "instruction": "Hold FAR, TILT top of board away from camera",
        "detail": "About 2.5-4 m — upward pitch",
        "region": (0.25, 0.10, 0.50, 0.50),
        "icon": "tilt-up",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Pitch Down",
        "instruction": "Hold FAR, TILT bottom of board away from camera",
        "detail": "About 2.5-4 m — downward pitch",
        "region": (0.25, 0.40, 0.50, 0.50),
        "icon": "tilt-down",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Upper-Left Corner",
        "instruction": "Hold FAR, position board in UPPER-LEFT",
        "detail": "About 2.5-4 m — small board in top-left area",
        "region": (0.02, 0.02, 0.40, 0.40),
        "icon": "top-left",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Upper-Right Corner",
        "instruction": "Hold FAR, position board in UPPER-RIGHT",
        "detail": "About 2.5-4 m — small board in top-right area",
        "region": (0.58, 0.02, 0.40, 0.40),
        "icon": "top-right",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Lower-Left Corner",
        "instruction": "Hold FAR, position board in LOWER-LEFT",
        "detail": "About 2.5-4 m — small board in bottom-left area",
        "region": (0.02, 0.58, 0.40, 0.40),
        "icon": "bottom-left",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Lower-Right Corner",
        "instruction": "Hold FAR, position board in LOWER-RIGHT",
        "detail": "About 2.5-4 m — small board in bottom-right area",
        "region": (0.58, 0.58, 0.40, 0.40),
        "icon": "bottom-right",
        "min_area_pct": 1.0,
        "max_area_pct": 5.0,
    },
    {
        "name": "[FAR] Smallest Detectable Board",
        "instruction": "Move as FAR as possible while board is still detected",
        "detail": "About 3.5-4 m — smallest visible board",
        "region": (0.30, 0.25, 0.40, 0.50),
        "icon": "backward",
        "min_area_pct": 0.5,
        "max_area_pct": 3.0,
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_session_dir(name=None):
    """Create or reuse a session folder. Returns (path, label, existing_pair_count)."""
    if name is None or name.strip() == "":
        name = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(SESSIONS_DIR, name)
    os.makedirs(path, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(path, "left_*.png")))
    return path, name, len(existing)


def collect_capture_pairs(session=None, latest=False):
    """
    Return ([(left_path, right_path), ...], {source_label: count}).

      session=None, latest=False  -> all sessions + legacy loose files (default)
      session="a,b"               -> only those named sessions
      latest=True                 -> only the most recent session by mtime
    """
    pairs = []
    counts = {}

    def add_dir(dirpath, label):
        lefts = sorted(glob.glob(os.path.join(dirpath, "left_*.png")))
        rights = sorted(glob.glob(os.path.join(dirpath, "right_*.png")))
        n = min(len(lefts), len(rights))
        for i in range(n):
            pairs.append((lefts[i], rights[i]))
        if n > 0:
            counts[label] = n

    if session:
        for name in [s.strip() for s in session.split(",") if s.strip()]:
            spath = os.path.join(SESSIONS_DIR, name)
            if not os.path.isdir(spath):
                print(f"[WARN] Session not found: {name}")
                continue
            add_dir(spath, f"sessions/{name}")
        return pairs, counts

    if latest:
        if not os.path.isdir(SESSIONS_DIR):
            return pairs, counts
        candidates = [
            os.path.join(SESSIONS_DIR, d)
            for d in os.listdir(SESSIONS_DIR)
            if os.path.isdir(os.path.join(SESSIONS_DIR, d))
        ]
        if not candidates:
            return pairs, counts
        newest = max(candidates, key=os.path.getmtime)
        add_dir(newest, f"sessions/{os.path.basename(newest)}")
        return pairs, counts

    if os.path.isdir(SESSIONS_DIR):
        for d in sorted(os.listdir(SESSIONS_DIR)):
            spath = os.path.join(SESSIONS_DIR, d)
            if os.path.isdir(spath):
                add_dir(spath, f"sessions/{d}")
    add_dir(CALIB_DIR, "(legacy loose files)")
    return pairs, counts


def split_stereo_frame(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


def open_camera(index=CAMERA_INDEX):
    # OpenCV's cap.set request to MJPG/2560x720 is unreliable on UVC drivers
    # (silently falls back to 1280x720 on some firmwares). Pre-set with v4l2-ctl
    # so the format actually sticks before OpenCV opens the device.
    device = f"/dev/video{index}"
    try:
        subprocess.run(
            ["v4l2-ctl", "--device", device,
             "--set-fmt-video=width=2560,height=720,pixelformat=MJPG"],
            check=False, capture_output=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # v4l2-ctl missing or stuck — fall back to OpenCV's own request

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {index}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1)
    return cap


def get_board_center_and_area(corners, img_shape):
    """Get the centroid and bounding area of detected corners."""
    pts = corners.reshape(-1, 2)
    cx, cy = np.mean(pts, axis=0)
    # Bounding rect area as percentage of image area
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    board_area = (x_max - x_min) * (y_max - y_min)
    img_area = img_shape[0] * img_shape[1]
    area_pct = (board_area / img_area) * 100
    return cx, cy, area_pct


def is_in_region(cx, cy, area_pct, region, img_w, img_h, pose):
    """Check if the board center is within the target region and size range."""
    rx, ry, rw, rh = region
    x1 = rx * img_w
    y1 = ry * img_h
    x2 = (rx + rw) * img_w
    y2 = (ry + rh) * img_h

    in_region = (x1 <= cx <= x2) and (y1 <= cy <= y2)
    in_size = pose["min_area_pct"] <= area_pct <= pose["max_area_pct"]

    return in_region and in_size


def draw_target_region(img, region, color, thickness=2):
    """Draw the target zone rectangle."""
    h, w = img.shape[:2]
    rx, ry, rw, rh = region
    x1, y1 = int(rx * w), int(ry * h)
    x2, y2 = int((rx + rw) * w), int((ry + rh) * h)

    # Draw dashed rectangle effect with rounded corners
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_arrow_icon(img, icon, cx, cy, size=40, color=(255, 255, 255)):
    """Draw a directional arrow/icon on the image."""
    arrows = {
        "center":      [(0, 0)],
        "forward":     [(0, -1)],
        "backward":    [(0, 1)],
        "left":        [(-1, 0)],
        "right":       [(1, 0)],
        "top-left":    [(-1, -1)],
        "top-right":   [(1, -1)],
        "bottom-left": [(-1, 1)],
        "bottom-right":[(1, 1)],
        "tilt-left":   [(-1, 0)],
        "tilt-right":  [(1, 0)],
        "tilt-up":     [(0, -1)],
        "tilt-down":   [(0, 1)],
    }

    if icon == "center":
        # Draw crosshair
        cv2.circle(img, (cx, cy), size // 2, color, 2)
        cv2.line(img, (cx - size, cy), (cx + size, cy), color, 1)
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, 1)
        return

    dirs = arrows.get(icon, [(0, 0)])
    for dx, dy in dirs:
        ex = cx + dx * size
        ey = cy + dy * size
        cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 3, tipLength=0.4)


def draw_progress_arc(img, progress, cx, cy, radius=50, thickness=4):
    """Draw a progress arc (like a loading ring)."""
    # Background circle
    cv2.circle(img, (cx, cy), radius, (60, 60, 60), thickness)

    # Progress arc
    angle = int(360 * progress)
    if angle > 0:
        color = (0, 255, 100) if progress >= 1.0 else (0, 200, 255)
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, angle, color, thickness + 2)

    # Percentage text
    pct_text = f"{int(progress * 100)}%"
    text_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(img, pct_text,
                (cx - text_size[0] // 2, cy + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_hold_steady_bar(img, hold_frames, required_frames, y_pos):
    """Draw a 'hold steady' progress bar that fills as user holds position."""
    h, w = img.shape[:2]
    bar_w = 300
    bar_h = 12
    x1 = (w - bar_w) // 2
    x2 = x1 + bar_w
    y1 = y_pos
    y2 = y1 + bar_h

    # Background
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Fill
    fill_pct = min(hold_frames / required_frames, 1.0)
    fill_x = int(x1 + (bar_w * fill_pct))
    color = (0, 255, 100) if fill_pct >= 1.0 else (0, 180, 255)
    cv2.rectangle(img, (x1, y1), (fill_x, y2), color, -1)

    # Label
    label = "HOLD STEADY..." if fill_pct < 1.0 else "CAPTURED!"
    cv2.putText(img, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# =============================================================================
# STEP 0: PRE-CALIBRATION CAMERA PREVIEW
# =============================================================================

def preview():
    """Live camera preview for pre-calibration sanity check.

    Shows the side-by-side stream split into left/right, prints negotiated
    resolution + FPS, and overlays checkerboard corners when detected. Use
    this to confirm both lenses see the world, focus is OK, and the board
    pattern is being recognized before starting the 40-pose run.
    """
    print("=" * 60)
    print("CAMERA PREVIEW (pre-calibration sanity check)")
    print("=" * 60)
    print("You should see two distinct camera views (left | right).")
    print("If the checkerboard is in view, corners will be drawn on it.")
    print("Press Q to quit.\n")

    cap = open_camera()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Negotiated: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")
    if actual_w != 2560 or actual_h != 720:
        print(f"[WARN] Expected 2560x720, got {actual_w}x{actual_h}.")
        print("       Side-by-side split may be wrong. Check v4l2-ctl --list-formats-ext.")
    print()

    # Force a resizable window at a sensible size. The combined frame is
    # 2560x720 — too wide for most screens. WINDOW_NORMAL lets the user
    # drag-resize and Qt scales the contents to fit.
    WINDOW_NAME = "Camera Preview (Left | Right)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 480)

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0
    frame_idx = 0
    # Run corner detection only every Nth frame — at 1280x720 it is expensive
    # on a Pi 5 and would otherwise pin display FPS to ~2.
    DETECT_EVERY = 10
    foundL = foundR = False
    cornersL = cornersR = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame")
            continue

        left, right = split_stereo_frame(frame)
        frame_idx += 1

        if frame_idx % DETECT_EVERY == 0:
            grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
            foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

        if foundL and cornersL is not None:
            cv2.drawChessboardCorners(left, CHECKERBOARD, cornersL, foundL)
        if foundR and cornersR is not None:
            cv2.drawChessboardCorners(right, CHECKERBOARD, cornersR, foundR)

        combined = np.hstack([left, right])

        fps_count += 1
        if fps_count >= 10:
            now = time.time()
            fps_display = fps_count / (now - fps_t0)
            fps_t0 = now
            fps_count = 0

        status = (f"{actual_w}x{actual_h} @ {fps_display:4.1f} fps  |  "
                  f"board L={'YES' if foundL else 'no '}  R={'YES' if foundR else 'no '}")
        cv2.rectangle(combined, (0, 0), (combined.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(combined, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Q = quit", (10, combined.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# STEP 1: GUIDED CAPTURE
# =============================================================================

def guided_capture(session_name=None):
    """
    Walks the user through each pose, showing target zones and instructions.
    Auto-captures when the board is held in the right position for ~1 second.

    Each capture run writes to its own folder under stereo_calibration_data/sessions/
    so multiple sessions accumulate without overwriting. If session_name matches an
    existing session, capture resumes there (numbering continues past existing pairs).
    """
    print("=" * 60)
    print("GUIDED STEREO CALIBRATION CAPTURE")
    print("=" * 60)
    session_path, session_label, existing_count = make_session_dir(session_name)
    print(f"Session: {session_label}")
    print(f"Folder:  {session_path}")
    if existing_count > 0:
        print(f"Resuming — {existing_count} pair(s) already in this session.")
    print(f"Total poses: {len(POSES)}")
    print(f"Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners, {SQUARE_SIZE_MM} mm squares")
    print()
    print("The screen will guide you through each pose.")
    print("Hold the checkerboard steady when it turns GREEN.")
    print("Auto-capture happens after ~1 second of stable detection.")
    print()
    print("Press S to SKIP a pose, Q to QUIT")
    print()

    cap = open_camera()

    # How many frames the board must be in-zone before auto-capture
    HOLD_REQUIRED = 20  # ~0.7 sec at 30fps

    # Force a resizable window at a sensible size. OpenCV's default
    # WINDOW_AUTOSIZE renders as a tiny ~400px box on some VNC + Qt builds.
    WINDOW_NAME = "Guided Calibration - Follow the instructions"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    pose_idx = 0
    pair_count = existing_count
    hold_count = 0

    # Flash effect timer
    flash_timer = 0

    while pose_idx < len(POSES):
        ret, frame = cap.read()
        if not ret:
            continue

        left, right = split_stereo_frame(frame)
        grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        pose = POSES[pose_idx]
        img_h, img_w = left.shape[:2]

        # Detect corners
        foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
        foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

        # Build display image (we show left camera as the guide view)
        display = left.copy()

        # Check if board is in target zone
        in_zone = False
        if foundL and foundR:
            cx, cy, area_pct = get_board_center_and_area(cornersL, grayL.shape)
            in_zone = is_in_region(cx, cy, area_pct, pose["region"],
                                    img_w, img_h, pose)

            # Draw detected corners
            corner_color = (0, 255, 0) if in_zone else (0, 165, 255)
            cv2.drawChessboardCorners(display, CHECKERBOARD, cornersL, foundL)

        if in_zone:
            hold_count += 1
            zone_color = (0, 255, 0)  # Green = in position
        else:
            hold_count = 0
            zone_color = (0, 140, 255)  # Orange = move to target

        # Draw target region
        draw_target_region(display, pose["region"], zone_color, 2)

        # Draw arrow icon in center of target region
        rx, ry, rw, rh = pose["region"]
        arrow_cx = int((rx + rw / 2) * img_w)
        arrow_cy = int((ry + rh / 2) * img_h)
        if not in_zone:
            draw_arrow_icon(display, pose["icon"], arrow_cx, arrow_cy, 50, zone_color)

        # --- Top bar: instruction ---
        bar_h = 80
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        cv2.putText(display, f"Pose {pose_idx + 1}/{len(POSES)}: {pose['name']}",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, pose["instruction"],
                     (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(display, pose["detail"],
                     (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # --- Progress ring (top-right) ---
        progress = (pose_idx) / len(POSES)
        draw_progress_arc(display, progress, img_w - 60, 45, 35, 3)

        # --- Hold steady bar ---
        if in_zone:
            draw_hold_steady_bar(display, hold_count, HOLD_REQUIRED, img_h - 40)

        # --- Status at bottom ---
        if not foundL and not foundR:
            status = "No checkerboard detected — adjust position"
            status_color = (0, 0, 200)
        elif not foundR:
            status = "Board seen in LEFT camera only — also need RIGHT"
            status_color = (0, 165, 255)
        elif not foundL:
            status = "Board seen in RIGHT camera only — also need LEFT"
            status_color = (0, 165, 255)
        elif not in_zone:
            status = "Board detected — move it into the target zone"
            status_color = (0, 165, 255)
        else:
            status = "HOLD STEADY for auto-capture..."
            status_color = (0, 255, 0)

        cv2.putText(display, status,
                     (10, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     status_color, 1)

        # --- Flash effect after capture ---
        if flash_timer > 0:
            alpha = flash_timer / 10.0
            white = np.ones_like(display) * 255
            cv2.addWeighted(white, alpha * 0.5, display, 1.0, 0, display)
            flash_timer -= 1

        # --- Auto-capture ---
        if hold_count >= HOLD_REQUIRED:
            pair_count += 1
            idx = pair_count
            lpath = os.path.join(session_path, f"left_{idx:03d}.png")
            rpath = os.path.join(session_path, f"right_{idx:03d}.png")
            cv2.imwrite(lpath, left)
            cv2.imwrite(rpath, right)
            print(f"  ✓ Pose {pose_idx + 1} captured: pair #{idx}")

            hold_count = 0
            flash_timer = 10  # Flash effect
            pose_idx += 1
            time.sleep(0.3)  # Brief pause
            continue

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"  → Skipped pose {pose_idx + 1}: {pose['name']}")
            pose_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nGuided capture complete!")
    print(f"  Captured: {pair_count}/{len(POSES)} poses")

    if pair_count < 25:
        print(f"  ⚠ {pair_count} pairs may not be enough. Consider re-running for skipped poses.")
    else:
        print(f"  ✓ Excellent coverage for calibration!")

    return pair_count >= 5


# =============================================================================
# STEP 2: CALIBRATION (same as before)
# =============================================================================

def calibrate(session=None, latest=False):
    print("=" * 60)
    print("STEREO CALIBRATION")
    print("=" * 60)

    pairs, counts = collect_capture_pairs(session=session, latest=latest)

    if not pairs:
        print("[ERROR] No calibration images found.")
        if session:
            print(f"        Looked in sessions: {session}")
        elif latest:
            print(f"        No sessions found in {SESSIONS_DIR}")
        else:
            print("        Run 'capture' first.")
        return False

    print(f"Found {len(pairs)} stereo pair(s) across:")
    for src, n in counts.items():
        print(f"  - {src}: {n}")
    print()

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = None
    used_pairs = 0

    for i, (lpath, rpath) in enumerate(pairs):
        imgL = cv2.imread(lpath)
        imgR = cv2.imread(rpath)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = grayL.shape[::-1]

        foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
        foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

        if foundL and foundR:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
            used_pairs += 1
            print(f"  Pair {i+1}: ✓")
        else:
            print(f"  Pair {i+1}: corners not found — skipped")

    if used_pairs < 5:
        print(f"\n[ERROR] Only {used_pairs} valid pairs. Need at least 5.")
        return False

    print(f"\nCalibrating with {used_pairs} pairs...\n")

    retL, mtxL, distL, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None)
    retR, mtxR, distR, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None)

    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR,
        img_size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=criteria
    )

    print(f"RMS reprojection error: {ret:.4f}")
    if ret < 0.5:
        print("  → Excellent! ✓")
    elif ret < 1.0:
        print("  → Good ✓")
    elif ret < 2.0:
        print("  → Acceptable")
    else:
        print("  → Poor — consider recalibrating")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, img_size, R, T, alpha=0)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_size, cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_size, cv2.CV_32FC1)

    ensure_dir(CALIB_DIR)
    np.savez(CALIB_FILE,
             mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             roi1=roi1, roi2=roi2,
             mapL1=mapL1, mapL2=mapL2, mapR1=mapR1, mapR2=mapR2,
             img_size=np.array(img_size), rms_error=ret)

    print(f"\nCalibration saved to: {CALIB_FILE}")
    print(f"  Left focal:  fx={mtxL[0,0]:.1f}, fy={mtxL[1,1]:.1f} px")
    print(f"  Right focal: fx={mtxR[0,0]:.1f}, fy={mtxR[1,1]:.1f} px")
    print(f"  Baseline:    [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}] mm")
    return True


# =============================================================================
# STEP 3: LIVE DEPTH VIEWER (same as previous script)
# =============================================================================

def depth_viewer():
    print("=" * 60)
    print("LIVE DEPTH VIEWER")
    print("=" * 60)

    if not os.path.exists(CALIB_FILE):
        print(f"[ERROR] No calibration found: {CALIB_FILE}")
        return

    data = np.load(CALIB_FILE)
    mapL1, mapL2 = data['mapL1'], data['mapL2']
    mapR1, mapR2 = data['mapR1'], data['mapR2']
    Q = data['Q']

    print("Calibration loaded. Starting depth viewer...")
    print("  Q/ESC = quit | +/- = numDisp | [/] = blockSize | W = warning toggle")

    cap = open_camera()
    num_disp = SGBM_NUM_DISP
    block_size = SGBM_BLOCK_SIZE
    show_warning = True
    warning_dist_m = 1.5

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        left, right = split_stereo_frame(frame)
        rectL = cv2.remap(left, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoSGBM_create(
            minDisparity=SGBM_MIN_DISP, numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32)

        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

        with np.errstate(divide='ignore', invalid='ignore'):
            focal_px = abs(Q[2, 3]) if Q[2, 3] != 0 else 500
            depth_m = np.where(disparity > 0, focal_px * BASELINE_M / disparity, 0)

        disp_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_color = cv2.applyColorMap(np.uint8(disp_display), cv2.COLORMAP_JET)

        if show_warning:
            h, w = depth_m.shape
            roi = depth_m[h//4:3*h//4, w//4:3*w//4]
            valid = roi[(roi > 0.1) & (roi < 20)]
            if len(valid) > 100:
                min_d = np.percentile(valid, 5)
                if min_d < warning_dist_m:
                    cv2.putText(disp_color, f"WARNING: {min_d:.2f}m",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    cv2.putText(disp_color, f"Clear: {min_d:.2f}m",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        info = f"numDisp={num_disp} | block={block_size}"
        cv2.putText(disp_color, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Rectified view with epipolar lines
        rect_view = np.hstack([rectL, rectR])
        for y in range(0, rect_view.shape[0], 30):
            cv2.line(rect_view, (0, y), (rect_view.shape[1], y), (0, 255, 0), 1)

        cv2.imshow("Rectified (verify alignment)", rect_view)
        cv2.imshow("Depth Map", disp_color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in (ord('+'), ord('=')):
            num_disp = min(num_disp + 16, 256)
        elif key == ord('-'):
            num_disp = max(num_disp - 16, 16)
        elif key == ord(']'):
            block_size = min(block_size + 2, 21)
        elif key == ord('['):
            block_size = max(block_size - 2, 3)
        elif key == ord('w'):
            show_warning = not show_warning

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# MAIN
# =============================================================================

def print_usage():
    print("""
Guided Stereo Calibration — Face-Scan Style
=============================================

Usage:
    python guided_calibration.py <command> [options]

Commands:
    preview     Live left/right preview with corner overlay (sanity check)
    capture     Guided pose-by-pose capture (like a face scan)
    calibrate   Run stereo calibration on captured pairs
    depth       Live depth viewer with obstacle warnings
    all         Full pipeline: capture → calibrate → depth

Options:
    --session-name NAME   capture: write into stereo_calibration_data/sessions/NAME/
                          (default: timestamp). Re-using a name resumes that session.
    --session A[,B,...]   calibrate: only use these named sessions.
    --latest              calibrate: only use the most recent session by mtime.

Default calibrate behavior combines ALL sessions (plus any legacy loose
left_*.png in stereo_calibration_data/).

The guided capture walks you through 40 poses across 4 distance zones:
    - NEAR (0.3-0.6 m)     — 10 poses for close-range precision
    - MID-NEAR (0.6-1.2 m) — 10 poses for moderate-speed avoidance
    - MID (1.2-2.5 m)      — 10 poses for navigation
    - FAR (2.5-4.0 m)      — 10 poses for early obstacle detection

Each pose auto-captures when you hold the board in position for ~1 second.
""")


def parse_args(argv):
    """Parse CLI args. Returns (cmd, opts)."""
    cmd = argv[0].lower() if argv else None
    opts = {"session_name": None, "session": None, "latest": False}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--session-name" and i + 1 < len(argv):
            opts["session_name"] = argv[i + 1]
            i += 2
        elif a == "--session" and i + 1 < len(argv):
            opts["session"] = argv[i + 1]
            i += 2
        elif a == "--latest":
            opts["latest"] = True
            i += 1
        else:
            print(f"[WARN] Unknown argument: {a}")
            i += 1
    return cmd, opts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    cmd, opts = parse_args(sys.argv[1:])

    if cmd == "preview":
        preview()
    elif cmd == "capture":
        guided_capture(session_name=opts["session_name"])
    elif cmd == "calibrate":
        calibrate(session=opts["session"], latest=opts["latest"])
    elif cmd == "depth":
        depth_viewer()
    elif cmd == "all":
        if guided_capture(session_name=opts["session_name"]):
            if calibrate(session=opts["session"], latest=opts["latest"]):
                depth_viewer()
    else:
        print(f"Unknown command: '{cmd}'")
        print_usage()
