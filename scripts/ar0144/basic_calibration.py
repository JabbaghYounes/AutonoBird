#!/usr/bin/env python3
"""
=============================================================================
Waveshare AR0144 Stereo Camera — Complete Calibration & Depth Viewer
=============================================================================

This is a ONE-STOP script that handles the full stereo calibration workflow:

    Step 1: Capture stereo image pairs (checkerboard)
    Step 2: Calibrate the stereo camera
    Step 3: Save calibration to disk
    Step 4: Run a live rectified depth viewer

Usage:
    python stereo_calibration.py capture     # Step 1: Capture checkerboard pairs
    python stereo_calibration.py calibrate   # Step 2: Run calibration on saved pairs
    python stereo_calibration.py depth       # Step 3: Live depth viewer
    python stereo_calibration.py all         # Run capture → calibrate → depth in sequence

Requirements:
    pip install opencv-python numpy

Hardware:
    - Waveshare AR0144 2MP Stereo USB Camera (52mm baseline)
    - Printed checkerboard on rigid, flat surface

Notes:
    - The AR0144 stereo module typically appears as a single USB device
      outputting a SIDE-BY-SIDE frame (left + right stitched horizontally).
    - This script splits that combined frame into left/right halves.
    - If your module exposes two separate /dev/video devices instead,
      adjust the capture section accordingly (see comments below).
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import glob
import json
import time

# =============================================================================
# CONFIGURATION — Adjust these to match your setup
# =============================================================================

# Checkerboard inner corners (columns x rows of INNER corners, not squares)
# For example, a 10x7 grid of squares has 9x6 inner corners.
# Count the internal corners on YOUR printout and set these values.
CHECKERBOARD_COLS = 9   # inner corners horizontally
CHECKERBOARD_ROWS = 6   # inner corners vertically
CHECKERBOARD = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)

# Size of each square in millimeters — MEASURE THIS on your actual printout.
# Reference values: A2 50mm board=50.0, A3 40mm board=40.0,
# A3-design auto-scaled-to-A4 (printer "fit to page")=25.0.
SQUARE_SIZE_MM = 26.0

# Camera device index (usually 0 for the first USB camera)
CAMERA_INDEX = 0

# How many stereo pairs to capture for calibration
NUM_PAIRS = 20

# Folder to save calibration images and results
CALIB_DIR = "stereo_calibration_data"
SESSIONS_DIR = os.path.join(CALIB_DIR, "sessions")
CALIB_FILE = os.path.join(CALIB_DIR, "stereo_calibration.npz")

# Baseline in meters (fixed for your module)
BASELINE_M = 0.052

# SGBM depth parameters (tuned for indoor/outdoor drone use)
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 128      # Must be divisible by 16
SGBM_BLOCK_SIZE = 7       # Odd number, 3-11 works well
SGBM_P1 = 8 * 3 * SGBM_BLOCK_SIZE ** 2
SGBM_P2 = 32 * 3 * SGBM_BLOCK_SIZE ** 2

# =============================================================================
# HELPER FUNCTIONS
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
    """
    Split a side-by-side stereo frame into left and right images.
    The AR0144 module typically outputs both cameras in one wide frame.
    """
    h, w = frame.shape[:2]
    mid = w // 2
    left = frame[:, :mid]
    right = frame[:, mid:]
    return left, right


def open_camera(index=CAMERA_INDEX):
    """Open the stereo camera and return the VideoCapture object."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {index}.")
        print("        Try changing CAMERA_INDEX (0, 1, 2...) or check USB connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Let camera warm up
    time.sleep(1)
    return cap


# =============================================================================
# STEP 1: CAPTURE STEREO IMAGE PAIRS
# =============================================================================

def capture_pairs(session_name=None):
    """
    Opens the stereo camera, displays a live preview, and lets you
    press SPACE to save a stereo pair when the checkerboard is visible.
    Press Q to quit early.

    Each capture run writes to its own folder under stereo_calibration_data/sessions/
    so multiple sessions accumulate without overwriting. If session_name matches an
    existing session, capture resumes there (numbering continues past existing pairs).
    """
    print("=" * 60)
    print("STEP 1: CAPTURE STEREO PAIRS")
    print("=" * 60)
    session_path, session_label, existing_count = make_session_dir(session_name)
    print(f"Session: {session_label}")
    print(f"Folder:  {session_path}")
    if existing_count > 0:
        print(f"Resuming — {existing_count} pair(s) already in this session.")
    print(f"Target: {NUM_PAIRS} pairs")
    print(f"Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners, {SQUARE_SIZE_MM} mm squares")
    print()
    print("Instructions:")
    print("  - Hold the checkerboard in front of the camera")
    print("  - Move it to different positions, angles, and distances")
    print("  - Press SPACE when both cameras can see the full board")
    print("  - Press Q when done (or after enough pairs captured)")
    print()

    cap = open_camera()
    pair_count = existing_count

    while pair_count < NUM_PAIRS:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame, retrying...")
            continue

        left, right = split_stereo_frame(frame)
        grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Try to find checkerboard corners for visual feedback
        foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
        foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

        # Draw corners on preview (green = found, red overlay = not found)
        previewL = left.copy()
        previewR = right.copy()

        if foundL:
            cv2.drawChessboardCorners(previewL, CHECKERBOARD, cornersL, foundL)
        if foundR:
            cv2.drawChessboardCorners(previewR, CHECKERBOARD, cornersR, foundR)

        # Status bar
        status = f"Pairs: {pair_count}/{NUM_PAIRS} | "
        if foundL and foundR:
            status += "BOTH DETECTED - Press SPACE to save"
            color = (0, 255, 0)
        elif foundL:
            status += "Left only - move board into right camera view"
            color = (0, 165, 255)
        elif foundR:
            status += "Right only - move board into left camera view"
            color = (0, 165, 255)
        else:
            status += "No board detected - adjust position"
            color = (0, 0, 255)

        # Combine left/right for display
        combined = np.hstack([previewL, previewR])
        cv2.putText(combined, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, color, 2)
        cv2.imshow("Stereo Capture (Left | Right) - SPACE=save, Q=quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if foundL and foundR:
                idx = pair_count + 1
                lpath = os.path.join(session_path, f"left_{idx:03d}.png")
                rpath = os.path.join(session_path, f"right_{idx:03d}.png")
                cv2.imwrite(lpath, left)
                cv2.imwrite(rpath, right)
                pair_count += 1
                print(f"  Saved pair #{idx}: {lpath}, {rpath}")
            else:
                print("  [SKIP] Board not detected in both cameras. Reposition and try again.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture complete. {pair_count} pairs in '{session_path}/'.")
    return pair_count > 0


# =============================================================================
# STEP 2: STEREO CALIBRATION
# =============================================================================

def calibrate(session=None, latest=False):
    """
    Runs stereo calibration on the captured image pairs.
    Saves all calibration matrices to a .npz file for reuse.
    """
    print("=" * 60)
    print("STEP 2: STEREO CALIBRATION")
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
    print(f"Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print()

    # Prepare 3D object points (real-world coordinates of checkerboard corners)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # Scale to real-world mm

    objpoints = []        # 3D points in world space
    imgpoints_left = []   # 2D points in left image
    imgpoints_right = []  # 2D points in right image

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = None
    used_pairs = 0

    for i, (lpath, rpath) in enumerate(pairs):
        imgL = cv2.imread(lpath)
        imgR = cv2.imread(rpath)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = grayL.shape[::-1]  # (width, height)

        foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
        foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

        if foundL and foundR:
            # Refine corner positions to sub-pixel accuracy
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
            used_pairs += 1
            print(f"  Pair {i+1}: corners found ✓")
        else:
            print(f"  Pair {i+1}: corners NOT found in {'left' if not foundL else 'right'} — skipped")

    if used_pairs < 5:
        print(f"\n[ERROR] Only {used_pairs} valid pairs. Need at least 5 for reliable calibration.")
        return False

    print(f"\nCalibrating with {used_pairs} valid pairs...")
    print("(This may take a moment...)\n")

    # --- Individual camera calibration first (better initial estimates) ---
    retL, mtxL, distL, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None
    )
    retR, mtxR, distR, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None
    )

    # --- Stereo calibration ---
    flags = cv2.CALIB_USE_INTRINSIC_GUESS  # Use individual calibration as starting point
    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR,
        img_size,
        flags=flags,
        criteria=criteria
    )

    print(f"Stereo calibration RMS reprojection error: {ret:.4f}")
    if ret < 0.5:
        print("  → Excellent calibration! ✓")
    elif ret < 1.0:
        print("  → Good calibration ✓")
    elif ret < 2.0:
        print("  → Acceptable, but could be improved with better images")
    else:
        print("  → Poor calibration. Consider recapturing with more care.")

    # --- Stereo rectification ---
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR,
        img_size, R, T,
        alpha=0  # 0 = crop to valid pixels only; 1 = keep all pixels
    )

    # --- Compute rectification maps (precomputed for speed) ---
    mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_size, cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_size, cv2.CV_32FC1)

    # --- Save everything ---
    ensure_dir(CALIB_DIR)
    np.savez(CALIB_FILE,
             mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             roi1=roi1, roi2=roi2,
             mapL1=mapL1, mapL2=mapL2,
             mapR1=mapR1, mapR2=mapR2,
             img_size=np.array(img_size),
             rms_error=ret)

    print(f"\nCalibration saved to: {CALIB_FILE}")
    print(f"\nKey results:")
    print(f"  Left focal length:  fx={mtxL[0,0]:.1f}, fy={mtxL[1,1]:.1f} px")
    print(f"  Right focal length: fx={mtxR[0,0]:.1f}, fy={mtxR[1,1]:.1f} px")
    print(f"  Baseline (T):       [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}] mm")
    print(f"  Image size:         {img_size[0]}x{img_size[1]}")
    return True


# =============================================================================
# STEP 3: LIVE DEPTH VIEWER
# =============================================================================

def depth_viewer():
    """
    Loads calibration, opens the stereo camera, and displays:
    - Rectified left/right images (with horizontal lines to verify alignment)
    - Live disparity / depth map
    - Obstacle proximity warning

    Controls:
        Q       = quit
        +/-     = adjust numDisparities
        [/]     = adjust blockSize
        W       = toggle obstacle warning overlay
    """
    print("=" * 60)
    print("STEP 3: LIVE DEPTH VIEWER")
    print("=" * 60)

    if not os.path.exists(CALIB_FILE):
        print(f"[ERROR] Calibration file not found: {CALIB_FILE}")
        print("        Run 'calibrate' first.")
        return

    # Load calibration
    data = np.load(CALIB_FILE)
    mapL1 = data['mapL1']
    mapL2 = data['mapL2']
    mapR1 = data['mapR1']
    mapR2 = data['mapR2']
    Q = data['Q']
    roi1 = tuple(data['roi1'])
    roi2 = tuple(data['roi2'])

    print("Calibration loaded.")
    print()
    print("Controls:")
    print("  Q       = quit")
    print("  +/-     = adjust numDisparities (range)")
    print("  [/]     = adjust blockSize (smoothness)")
    print("  W       = toggle obstacle warning")
    print("  R       = toggle rectification lines")
    print()

    cap = open_camera()

    num_disp = SGBM_NUM_DISP
    block_size = SGBM_BLOCK_SIZE
    show_warning = True
    show_lines = True
    warning_dist_m = 1.5  # Obstacle warning threshold in meters

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        left, right = split_stereo_frame(frame)

        # --- Rectify ---
        rectL = cv2.remap(left, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)

        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        # --- Compute disparity with SGBM ---
        stereo = cv2.StereoSGBM_create(
            minDisparity=SGBM_MIN_DISP,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

        # --- Convert disparity to depth in meters ---
        # Depth = f * B / disparity  (Q matrix encodes this)
        # Or use reprojectImageTo3D for full 3D
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_m = np.where(disparity > 0,
                               (Q[2, 3] * BASELINE_M * 1000) / (disparity + Q[3, 3]),
                               0)
            # Simpler: focal_length * baseline / disparity
            focal_px = Q[2, 3]  # focal length from Q matrix
            if focal_px != 0:
                depth_m = np.where(disparity > 0,
                                   abs(focal_px) * BASELINE_M / disparity,
                                   0)

        # --- Colorize disparity for display ---
        disp_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_display = np.uint8(disp_display)
        disp_color = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)

        # --- Obstacle warning ---
        if show_warning:
            h, w = depth_m.shape
            # Check center region (where the drone is heading)
            roi_y1, roi_y2 = h // 4, 3 * h // 4
            roi_x1, roi_x2 = w // 4, 3 * w // 4
            center_depth = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depth = center_depth[(center_depth > 0.1) & (center_depth < 20)]

            if len(valid_depth) > 100:
                min_dist = np.percentile(valid_depth, 5)  # 5th percentile = closest obstacles
                avg_dist = np.mean(valid_depth)

                # Draw warning on disparity map
                if min_dist < warning_dist_m:
                    cv2.putText(disp_color, f"WARNING: {min_dist:.2f}m",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.rectangle(disp_color, (roi_x1, roi_y1), (roi_x2, roi_y2),
                                  (0, 0, 255), 3)
                else:
                    cv2.putText(disp_color, f"Clear: {min_dist:.2f}m",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.rectangle(disp_color, (roi_x1, roi_y1), (roi_x2, roi_y2),
                                  (0, 255, 0), 1)

        # --- Rectification verification lines ---
        rect_combined = np.hstack([rectL, rectR])
        if show_lines:
            for y in range(0, rect_combined.shape[0], 30):
                cv2.line(rect_combined, (0, y), (rect_combined.shape[1], y), (0, 255, 0), 1)

        # --- Display info ---
        info = f"numDisp={num_disp} | blockSize={block_size} | +/- and [/] to adjust"
        cv2.putText(disp_color, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Rectified Stereo (verify horizontal alignment)", rect_combined)
        cv2.imshow("Depth Map (JET colormap)", disp_color)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            num_disp = min(num_disp + 16, 256)
        elif key == ord('-'):
            num_disp = max(num_disp - 16, 16)
        elif key == ord(']'):
            block_size = min(block_size + 2, 21)
        elif key == ord('['):
            block_size = max(block_size - 2, 3)
        elif key == ord('w'):
            show_warning = not show_warning
        elif key == ord('r'):
            show_lines = not show_lines

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# MAIN — Command-line interface
# =============================================================================

def print_usage():
    print("""
Waveshare AR0144 Stereo Camera — Calibration & Depth Tool
==========================================================

Usage:
    python basic_calibration.py <command> [options]

Commands:
    capture     Capture stereo checkerboard image pairs
    calibrate   Run stereo calibration on captured pairs
    depth       Live depth viewer (requires calibration)
    all         Run full pipeline: capture → calibrate → depth

Options:
    --session-name NAME   capture: write into stereo_calibration_data/sessions/NAME/
                          (default: timestamp). Re-using a name resumes that session.
    --session A[,B,...]   calibrate: only use these named sessions.
    --latest              calibrate: only use the most recent session by mtime.

Default calibrate behavior combines ALL sessions (plus any legacy loose
left_*.png in stereo_calibration_data/).

Workflow:
    1. Print a checkerboard, mount it on something rigid
    2. Run 'capture' to save 15-20 stereo pairs
    3. Run 'calibrate' to compute and save calibration
    4. Run 'depth' to see live depth maps

Configuration:
    Edit the CONFIGURATION section at the top of this script to match
    your checkerboard size, square dimensions, and camera index.

    Current settings:
        Checkerboard:  {cols}x{rows} inner corners
        Square size:   {sq} mm
        Camera index:  {cam}
        Target pairs:  {pairs}
""".format(
        cols=CHECKERBOARD_COLS,
        rows=CHECKERBOARD_ROWS,
        sq=SQUARE_SIZE_MM,
        cam=CAMERA_INDEX,
        pairs=NUM_PAIRS
    ))


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

    command, opts = parse_args(sys.argv[1:])

    if command == "capture":
        capture_pairs(session_name=opts["session_name"])

    elif command == "calibrate":
        calibrate(session=opts["session"], latest=opts["latest"])

    elif command == "depth":
        depth_viewer()

    elif command == "all":
        print("Running full pipeline: capture → calibrate → depth\n")
        if capture_pairs(session_name=opts["session_name"]):
            if calibrate(session=opts["session"], latest=opts["latest"]):
                depth_viewer()
            else:
                print("\nCalibration failed. Fix issues and try again.")
        else:
            print("\nNo pairs captured. Cannot continue.")

    else:
        print(f"Unknown command: '{command}'")
        print_usage()
