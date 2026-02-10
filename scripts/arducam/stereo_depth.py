#!/usr/bin/env python3
"""
Stereo Depth Mapping for Raspberry Pi 4
========================================
Arducam IMX519 16MP Autofocus Synchronized Quad-Camera Kit
Two cameras, 75mm baseline separation

Usage:
    python3 stereo_depth.py calibrate    # Capture calibration images with checkerboard
    python3 stereo_depth.py compute_cal  # Compute calibration from saved images
    python3 stereo_depth.py depth        # Run real-time depth mapping
    python3 stereo_depth.py capture      # Just capture and save stereo pairs

Requirements:
    pip3 install opencv-python-headless numpy
    sudo apt install -y libcamera-apps python3-picamera2
"""

import sys
import os
import time
import json
import glob
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

CONFIG = {
    # Camera settings
    "camera_ids": [0, 1],          # Arducam adapter channels for left/right
    "resolution": (2328, 1748),    # IMX519 modes: (4656,3496), (2328,1748), (1920,1080)
    "preview_resolution": (1280, 720),  # Resolution for live depth preview
    "baseline_mm": 75.0,           # Distance between camera centers in mm

    # Stereo matching
    "num_disparities": 128,        # Must be divisible by 16 (higher = farther range)
    "block_size": 9,               # Odd number, matching block size
    "min_disparity": 0,
    "uniqueness_ratio": 10,
    "speckle_window_size": 100,
    "speckle_range": 32,
    "disp12_max_diff": 1,
    "pre_filter_cap": 63,
    "pre_filter_type": 1,          # 0=NORMALIZED, 1=XSOBEL
    "use_sgbm": True,              # True=SGBM (better quality), False=BM (faster)
    "sgbm_mode": "hh",             # "sgbm", "hh" (full 8-dir), "3way"
    "wls_filter": True,            # Use WLS filter for smoother depth
    "wls_lambda": 8000,
    "wls_sigma": 1.5,

    # Calibration
    "checkerboard_size": (9, 6),   # Inner corners (cols, rows)
    "square_size_mm": 25.0,        # Actual square size on printed board
    "calibration_frames": 20,      # Number of stereo pairs to collect
    "calibration_dir": "calibration_images",
    "calibration_file": "stereo_calibration.json",

    # Output
    "output_dir": "depth_output",
    "colormap": cv2.COLORMAP_MAGMA,  # Depth visualization colormap
}


# ──────────────────────────────────────────────────────────────────
# Camera Interface (Arducam Quad-Camera via libcamera / Picamera2)
# ──────────────────────────────────────────────────────────────────

class ArducamStereoCamera:
    """
    Interface for the Arducam synchronized quad-camera adapter.
    Uses Picamera2 with channel switching for the multiplexed cameras.
    """

    def __init__(self, cam_ids=(0, 1), resolution=(2328, 1748)):
        self.cam_ids = cam_ids
        self.resolution = resolution
        self.picam2 = None
        self._init_camera()

    def _init_camera(self):
        """Initialize Picamera2 with the Arducam adapter."""
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("ERROR: picamera2 not found. Install with:")
            print("  sudo apt install -y python3-picamera2")
            sys.exit(1)

        self.picam2 = Picamera2()

        # Configure for still capture at desired resolution
        config = self.picam2.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"},
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)  # Warm-up

    def _switch_channel(self, channel):
        """
        Switch the Arducam multiplexer to the specified camera channel.
        Uses i2c commands for the Arducam quad-camera hat.
        """
        import subprocess

        # Arducam quad-camera adapter I2C channel switching
        # Adapter board address is typically 0x24
        # Channel mapping: 0=A, 1=B, 2=C, 3=D
        i2c_bus = 10  # Default for Arducam adapter on Pi 4
        adapter_addr = 0x24

        # Try using Arducam's built-in switching utility first
        try:
            subprocess.run(
                ["i2ctransfer", "-y", str(i2c_bus), f"w2@0x{adapter_addr:02x}",
                 "0x24", f"0x{(1 << channel):02x}"],
                check=True, capture_output=True, timeout=2
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try the arducam channel switch script
            try:
                subprocess.run(
                    ["/usr/local/bin/channel_switch", str(channel)],
                    check=True, capture_output=True, timeout=2
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Final fallback: write via Python smbus2
                try:
                    import smbus2
                    bus = smbus2.SMBus(i2c_bus)
                    bus.write_byte_data(adapter_addr, 0x24, 1 << channel)
                    bus.close()
                except Exception as e:
                    print(f"WARNING: Could not switch camera channel: {e}")
                    print("You may need to configure the I2C bus/address for your adapter.")
                    print("See: https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamHAT/")

        time.sleep(0.15)  # Allow sensor to stabilize after switch

    def capture_stereo_pair(self):
        """Capture a synchronized stereo image pair (left, right)."""
        # Capture left camera
        self._switch_channel(self.cam_ids[0])
        time.sleep(0.05)  # Short settle time
        left = self.picam2.capture_array()

        # Capture right camera
        self._switch_channel(self.cam_ids[1])
        time.sleep(0.05)
        right = self.picam2.capture_array()

        # Convert from RGB to BGR for OpenCV
        left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

        return left, right

    def close(self):
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()


class FileStereoCamera:
    """Fallback: load stereo pairs from image files (for testing)."""

    def __init__(self, left_path, right_path):
        self.left = cv2.imread(left_path)
        self.right = cv2.imread(right_path)
        if self.left is None or self.right is None:
            raise FileNotFoundError(f"Could not load images: {left_path}, {right_path}")

    def capture_stereo_pair(self):
        return self.left.copy(), self.right.copy()

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────
# Stereo Calibration
# ──────────────────────────────────────────────────────────────────

class StereoCalibrator:
    """Handles stereo camera calibration using a checkerboard pattern."""

    def __init__(self, config):
        self.config = config
        self.board_size = config["checkerboard_size"]
        self.square_size = config["square_size_mm"]
        self.cal_dir = Path(config["calibration_dir"])
        self.cal_dir.mkdir(parents=True, exist_ok=True)

        # Prepare object points (3D points in real-world coordinates)
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[
            0:self.board_size[0], 0:self.board_size[1]
        ].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Termination criteria for cornerSubPix
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5
        )

    def capture_calibration_images(self, camera):
        """Interactive calibration image capture session."""
        target = self.config["calibration_frames"]
        count = 0

        print(f"\n{'='*60}")
        print(f"  STEREO CALIBRATION IMAGE CAPTURE")
        print(f"{'='*60}")
        print(f"  Checkerboard: {self.board_size[0]}x{self.board_size[1]} inner corners")
        print(f"  Square size:  {self.square_size}mm")
        print(f"  Target:       {target} stereo pairs")
        print(f"  Save to:      {self.cal_dir}/")
        print(f"")
        print(f"  Controls:")
        print(f"    SPACE  - Capture if checkerboard detected")
        print(f"    Q/ESC  - Quit")
        print(f"{'='*60}\n")

        while count < target:
            left, right = camera.capture_stereo_pair()

            # Resize for display
            display_w = 640
            scale = display_w / left.shape[1]
            left_small = cv2.resize(left, None, fx=scale, fy=scale)
            right_small = cv2.resize(right, None, fx=scale, fy=scale)

            # Detect checkerboard
            gray_l = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found_l, corners_l = cv2.findChessboardCorners(gray_l, self.board_size, flags)
            found_r, corners_r = cv2.findChessboardCorners(gray_r, self.board_size, flags)

            # Draw detection results
            vis_l = left_small.copy()
            vis_r = right_small.copy()

            if found_l:
                cv2.drawChessboardCorners(vis_l, self.board_size, corners_l, found_l)
            if found_r:
                cv2.drawChessboardCorners(vis_r, self.board_size, corners_r, found_r)

            # Status bar
            status = f"Captured: {count}/{target}"
            both_found = found_l and found_r
            if both_found:
                status += " | CHECKERBOARD DETECTED - Press SPACE"
                color = (0, 255, 0)
            else:
                status += " | Searching..."
                color = (0, 0, 255)

            cv2.putText(vis_l, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis_r, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            combined = np.hstack([vis_l, vis_r])
            cv2.putText(combined, status, (10, combined.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Stereo Calibration", combined)
            key = cv2.waitKey(100) & 0xFF

            if key == ord(' ') and both_found:
                # Save full-resolution images
                fname = f"stereo_{count:03d}"
                cv2.imwrite(str(self.cal_dir / f"{fname}_left.png"), left)
                cv2.imwrite(str(self.cal_dir / f"{fname}_right.png"), right)
                count += 1
                print(f"  Saved pair {count}/{target}")
                time.sleep(0.5)  # Avoid double captures

            elif key in (ord('q'), 27):
                print(f"\nCapture ended early. Got {count} pairs.")
                break

        cv2.destroyAllWindows()
        print(f"\nCalibration images saved to: {self.cal_dir}/")
        return count

    def compute_calibration(self):
        """Compute stereo calibration from saved image pairs."""
        print(f"\n{'='*60}")
        print(f"  COMPUTING STEREO CALIBRATION")
        print(f"{'='*60}\n")

        # Find all stereo pairs
        left_files = sorted(glob.glob(str(self.cal_dir / "*_left.png")))
        right_files = sorted(glob.glob(str(self.cal_dir / "*_right.png")))

        if len(left_files) == 0:
            print("ERROR: No calibration images found!")
            print(f"  Expected in: {self.cal_dir}/")
            print(f"  Run: python3 stereo_depth.py calibrate")
            return None

        if len(left_files) != len(right_files):
            print(f"WARNING: Mismatched pairs ({len(left_files)} left, {len(right_files)} right)")

        pairs = min(len(left_files), len(right_files))
        print(f"  Found {pairs} stereo pairs")

        obj_points = []   # 3D points
        img_points_l = [] # 2D points in left images
        img_points_r = [] # 2D points in right images
        img_size = None

        for i in range(pairs):
            left = cv2.imread(left_files[i], cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(right_files[i], cv2.IMREAD_GRAYSCALE)

            if img_size is None:
                img_size = (left.shape[1], left.shape[0])

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found_l, corners_l = cv2.findChessboardCorners(left, self.board_size, flags)
            found_r, corners_r = cv2.findChessboardCorners(right, self.board_size, flags)

            if found_l and found_r:
                # Refine to sub-pixel accuracy
                corners_l = cv2.cornerSubPix(left, corners_l, (11, 11), (-1, -1), self.criteria)
                corners_r = cv2.cornerSubPix(right, corners_r, (11, 11), (-1, -1), self.criteria)

                obj_points.append(self.objp)
                img_points_l.append(corners_l)
                img_points_r.append(corners_r)
                print(f"  Pair {i+1}: OK")
            else:
                print(f"  Pair {i+1}: SKIPPED (detection failed: L={found_l}, R={found_r})")

        usable = len(obj_points)
        print(f"\n  Usable pairs: {usable}/{pairs}")

        if usable < 5:
            print("ERROR: Need at least 5 usable pairs for reliable calibration.")
            return None

        # Step 1: Calibrate each camera individually
        print("\n  Calibrating left camera...")
        ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            obj_points, img_points_l, img_size, None, None,
            flags=cv2.CALIB_FIX_K3
        )
        print(f"    RMS reprojection error: {ret_l:.4f}")

        print("  Calibrating right camera...")
        ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            obj_points, img_points_r, img_size, None, None,
            flags=cv2.CALIB_FIX_K3
        )
        print(f"    RMS reprojection error: {ret_r:.4f}")

        # Step 2: Stereo calibration
        print("  Running stereo calibration...")
        flags = (
            cv2.CALIB_FIX_INTRINSIC  # Use individual calibrations as starting point
        )

        ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_l, img_points_r,
            K_l, D_l, K_r, D_r, img_size,
            criteria=self.criteria,
            flags=flags
        )
        print(f"    Stereo RMS error: {ret_stereo:.4f}")

        # Step 3: Stereo rectification
        print("  Computing rectification transforms...")
        R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0  # 0 = crop to valid pixels, 1 = keep all pixels
        )

        # Compute rectification maps
        map_l1, map_l2 = cv2.initUndistortRectifyMap(
            K_l, D_l, R1, P1, img_size, cv2.CV_32FC1
        )
        map_r1, map_r2 = cv2.initUndistortRectifyMap(
            K_r, D_r, R2, P2, img_size, cv2.CV_32FC1
        )

        # Package calibration data
        calibration = {
            "image_size": list(img_size),
            "baseline_mm": self.config["baseline_mm"],
            "stereo_rms_error": ret_stereo,
            "K_left": K_l.tolist(),
            "D_left": D_l.tolist(),
            "K_right": K_r.tolist(),
            "D_right": D_r.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q": Q.tolist(),
            "roi_left": list(roi_l),
            "roi_right": list(roi_r),
        }

        # Save calibration JSON
        cal_file = self.config["calibration_file"]
        with open(cal_file, 'w') as f:
            json.dump(calibration, f, indent=2)

        # Save rectification maps as numpy arrays (faster to load)
        np.savez("stereo_rectify_maps.npz",
                 map_l1=map_l1, map_l2=map_l2,
                 map_r1=map_r1, map_r2=map_r2)

        print(f"\n  Calibration saved: {cal_file}")
        print(f"  Rectification maps: stereo_rectify_maps.npz")
        print(f"\n  Baseline: {np.linalg.norm(T):.1f}mm")
        print(f"  Focal length (left):  {K_l[0,0]:.1f}px")
        print(f"  Focal length (right): {K_r[0,0]:.1f}px")

        if ret_stereo > 1.0:
            print(f"\n  WARNING: RMS error {ret_stereo:.2f} is high.")
            print(f"  Consider recapturing calibration images with the board")
            print(f"  at varied angles and distances.")

        return calibration


# ──────────────────────────────────────────────────────────────────
# Depth Estimation Engine
# ──────────────────────────────────────────────────────────────────

class StereoDepthEstimator:
    """Computes depth maps from rectified stereo image pairs."""

    def __init__(self, config):
        self.config = config
        self.calibration = None
        self.map_l1 = self.map_l2 = None
        self.map_r1 = self.map_r2 = None
        self.Q = None
        self.stereo_matcher = None
        self.wls_filter = None
        self._load_calibration()
        self._init_matcher()

    def _load_calibration(self):
        """Load stereo calibration and rectification maps."""
        cal_file = self.config["calibration_file"]
        maps_file = "stereo_rectify_maps.npz"

        if not os.path.exists(cal_file):
            print(f"ERROR: Calibration file not found: {cal_file}")
            print(f"  Run: python3 stereo_depth.py calibrate")
            print(f"  Then: python3 stereo_depth.py compute_cal")
            sys.exit(1)

        with open(cal_file, 'r') as f:
            self.calibration = json.load(f)

        self.Q = np.array(self.calibration["Q"])

        if os.path.exists(maps_file):
            maps = np.load(maps_file)
            self.map_l1 = maps["map_l1"]
            self.map_l2 = maps["map_l2"]
            self.map_r1 = maps["map_r1"]
            self.map_r2 = maps["map_r2"]
        else:
            # Recompute from calibration data
            img_size = tuple(self.calibration["image_size"])
            K_l = np.array(self.calibration["K_left"])
            D_l = np.array(self.calibration["D_left"])
            K_r = np.array(self.calibration["K_right"])
            D_r = np.array(self.calibration["D_right"])
            R1 = np.array(self.calibration["R1"])
            R2 = np.array(self.calibration["R2"])
            P1 = np.array(self.calibration["P1"])
            P2 = np.array(self.calibration["P2"])

            self.map_l1, self.map_l2 = cv2.initUndistortRectifyMap(
                K_l, D_l, R1, P1, img_size, cv2.CV_32FC1)
            self.map_r1, self.map_r2 = cv2.initUndistortRectifyMap(
                K_r, D_r, R2, P2, img_size, cv2.CV_32FC1)

        print(f"  Calibration loaded (RMS: {self.calibration['stereo_rms_error']:.3f})")

    def _init_matcher(self):
        """Initialize the stereo block matcher."""
        cfg = self.config
        num_disp = cfg["num_disparities"]
        block_size = cfg["block_size"]

        if cfg["use_sgbm"]:
            mode_map = {
                "sgbm": cv2.StereoSGBM_MODE_SGBM,
                "hh": cv2.StereoSGBM_MODE_HH,
                "3way": cv2.StereoSGBM_MODE_SGBM_3WAY,
            }
            mode = mode_map.get(cfg["sgbm_mode"], cv2.StereoSGBM_MODE_HH)

            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=cfg["min_disparity"],
                numDisparities=num_disp,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,
                P2=32 * 3 * block_size ** 2,
                disp12MaxDiff=cfg["disp12_max_diff"],
                uniquenessRatio=cfg["uniqueness_ratio"],
                speckleWindowSize=cfg["speckle_window_size"],
                speckleRange=cfg["speckle_range"],
                preFilterCap=cfg["pre_filter_cap"],
                mode=mode,
            )
            print(f"  Matcher: SGBM ({cfg['sgbm_mode']}) | "
                  f"Disparities: {num_disp} | Block: {block_size}")
        else:
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=num_disp,
                blockSize=block_size,
            )
            self.stereo_matcher.setPreFilterType(cfg["pre_filter_type"])
            self.stereo_matcher.setPreFilterCap(cfg["pre_filter_cap"])
            self.stereo_matcher.setUniquenessRatio(cfg["uniqueness_ratio"])
            self.stereo_matcher.setSpeckleWindowSize(cfg["speckle_window_size"])
            self.stereo_matcher.setSpeckleRange(cfg["speckle_range"])
            self.stereo_matcher.setDisp12MaxDiff(cfg["disp12_max_diff"])
            print(f"  Matcher: BM | Disparities: {num_disp} | Block: {block_size}")

        # WLS (Weighted Least Squares) filter for smoother depth
        if cfg["wls_filter"]:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self.stereo_matcher
            )
            self.wls_filter.setLambda(cfg["wls_lambda"])
            self.wls_filter.setSigmaColor(cfg["wls_sigma"])
            print(f"  WLS filter: ON (λ={cfg['wls_lambda']}, σ={cfg['wls_sigma']})")

    def rectify(self, left, right):
        """Apply stereo rectification to an image pair."""
        rect_l = cv2.remap(left, self.map_l1, self.map_l2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(right, self.map_r1, self.map_r2, cv2.INTER_LINEAR)
        return rect_l, rect_r

    def compute_disparity(self, left_rect, right_rect):
        """Compute disparity map from rectified stereo pair."""
        # Convert to grayscale for matching
        if len(left_rect.shape) == 3:
            gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        else:
            gray_l, gray_r = left_rect, right_rect

        if self.wls_filter is not None:
            # Compute both left and right disparities for WLS filtering
            disp_l = self.stereo_matcher.compute(gray_l, gray_r)
            disp_r = self.right_matcher.compute(gray_r, gray_l)
            disparity = self.wls_filter.filter(disp_l, gray_l, None, disp_r)
        else:
            disparity = self.stereo_matcher.compute(gray_l, gray_r)

        return disparity

    def disparity_to_depth(self, disparity):
        """Convert disparity map to depth in millimeters."""
        # Disparity from OpenCV is in fixed-point (divide by 16)
        disp_float = disparity.astype(np.float32) / 16.0

        # Reproject to 3D using Q matrix
        # depth = baseline * focal_length / disparity
        points_3d = cv2.reprojectImageTo3D(disp_float, self.Q)
        depth_map = points_3d[:, :, 2]

        # Clean up invalid values
        depth_map[disp_float <= 0] = 0
        depth_map[depth_map < 0] = 0
        depth_map[depth_map > 10000] = 0  # Cap at 10m

        return depth_map

    def colorize_depth(self, depth_map, max_depth_mm=3000):
        """Create a colored visualization of the depth map."""
        # Normalize to 0-255 range
        depth_vis = np.clip(depth_map, 0, max_depth_mm)
        depth_vis = (depth_vis / max_depth_mm * 255).astype(np.uint8)
        depth_vis = 255 - depth_vis  # Invert so closer = brighter

        # Apply colormap
        colored = cv2.applyColorMap(depth_vis, self.config["colormap"])

        # Black out invalid regions
        colored[depth_map <= 0] = 0

        return colored

    def process_frame(self, left, right, preview_scale=None):
        """Full pipeline: rectify → disparity → depth → visualization."""
        # Optionally downscale for speed
        if preview_scale and preview_scale != 1.0:
            left = cv2.resize(left, None, fx=preview_scale, fy=preview_scale)
            right = cv2.resize(right, None, fx=preview_scale, fy=preview_scale)

        # Rectify
        rect_l, rect_r = self.rectify(left, right)

        # Compute disparity
        disparity = self.compute_disparity(rect_l, rect_r)

        # Convert to depth
        depth_map = self.disparity_to_depth(disparity)

        # Colorize
        depth_colored = self.colorize_depth(depth_map)

        return rect_l, rect_r, disparity, depth_map, depth_colored


# ──────────────────────────────────────────────────────────────────
# Application Modes
# ──────────────────────────────────────────────────────────────────

def run_calibration_capture(config):
    """Interactively capture calibration images."""
    camera = ArducamStereoCamera(
        cam_ids=config["camera_ids"],
        resolution=config["resolution"]
    )
    calibrator = StereoCalibrator(config)

    try:
        count = calibrator.capture_calibration_images(camera)
        if count >= 5:
            print("\nReady to compute calibration:")
            print("  python3 stereo_depth.py compute_cal")
    finally:
        camera.close()


def run_compute_calibration(config):
    """Compute calibration from previously captured images."""
    calibrator = StereoCalibrator(config)
    cal = calibrator.compute_calibration()
    if cal:
        print("\nReady for depth mapping:")
        print("  python3 stereo_depth.py depth")


def run_depth_mapping(config):
    """Real-time stereo depth mapping with live preview."""
    camera = ArducamStereoCamera(
        cam_ids=config["camera_ids"],
        resolution=config["preview_resolution"]
    )
    estimator = StereoDepthEstimator(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute scale factor between calibration resolution and preview resolution
    cal_size = tuple(estimator.calibration["image_size"])
    preview_size = config["preview_resolution"]
    scale = preview_size[0] / cal_size[0]

    print(f"\n{'='*60}")
    print(f"  REAL-TIME STEREO DEPTH MAPPING")
    print(f"{'='*60}")
    print(f"  Resolution: {preview_size[0]}x{preview_size[1]}")
    print(f"  Scale from calibration: {scale:.2f}x")
    print(f"")
    print(f"  Controls:")
    print(f"    S      - Save current depth frame")
    print(f"    D      - Toggle disparity/depth view")
    print(f"    R      - Toggle rectification lines")
    print(f"    +/-    - Adjust max depth range")
    print(f"    Q/ESC  - Quit")
    print(f"{'='*60}\n")

    show_lines = False
    show_disparity = False
    max_depth = 3000  # mm
    frame_count = 0
    fps_time = time.time()

    try:
        while True:
            t0 = time.time()

            left, right = camera.capture_stereo_pair()

            # Process
            rect_l, rect_r, disparity, depth_map, depth_colored = \
                estimator.process_frame(left, right, preview_scale=scale)

            dt = time.time() - t0

            # FPS calculation
            frame_count += 1
            if time.time() - fps_time > 1.0:
                fps = frame_count / (time.time() - fps_time)
                frame_count = 0
                fps_time = time.time()
            else:
                fps = 0

            # --- Build display ---
            display_h = 360
            aspect = rect_l.shape[1] / rect_l.shape[0]
            display_w = int(display_h * aspect)

            vis_left = cv2.resize(rect_l, (display_w, display_h))

            if show_disparity:
                # Show raw disparity
                disp_vis = cv2.normalize(disparity, None, 0, 255,
                                         cv2.NORM_MINMAX, cv2.CV_8U)
                disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                vis_right = cv2.resize(disp_vis, (display_w, display_h))
            else:
                vis_right = cv2.resize(depth_colored, (display_w, display_h))

            # Draw epipolar lines on left image to verify rectification
            if show_lines:
                for y in range(0, display_h, 30):
                    cv2.line(vis_left, (0, y), (display_w, y), (0, 255, 0), 1)
                    cv2.line(vis_right, (0, y), (display_w, y), (0, 255, 0), 1)

            # Labels
            cv2.putText(vis_left, "Rectified Left", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            label = "Disparity" if show_disparity else f"Depth (max {max_depth/1000:.1f}m)"
            cv2.putText(vis_right, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            combined = np.hstack([vis_left, vis_right])

            # Info bar
            if fps > 0:
                info = f"FPS: {fps:.1f} | Processing: {dt*1000:.0f}ms"
                cv2.putText(combined, info, (10, combined.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Stereo Depth", combined)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(output_dir / f"depth_{ts}.png"), depth_colored)
                cv2.imwrite(str(output_dir / f"left_{ts}.png"), rect_l)
                np.save(str(output_dir / f"depth_mm_{ts}.npy"), depth_map)
                print(f"  Saved frame: {ts}")
            elif key == ord('d'):
                show_disparity = not show_disparity
            elif key == ord('r'):
                show_lines = not show_lines
            elif key in (ord('+'), ord('=')):
                max_depth = min(max_depth + 500, 10000)
            elif key == ord('-'):
                max_depth = max(max_depth - 500, 500)

    finally:
        cv2.destroyAllWindows()
        camera.close()


def run_capture(config):
    """Capture and save a single stereo pair with depth."""
    camera = ArducamStereoCamera(
        cam_ids=config["camera_ids"],
        resolution=config["resolution"]
    )
    estimator = StereoDepthEstimator(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Capturing stereo pair...")
        left, right = camera.capture_stereo_pair()

        # Full resolution processing
        rect_l, rect_r, disparity, depth_map, depth_colored = \
            estimator.process_frame(left, right)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(str(output_dir / f"left_{ts}.png"), left)
        cv2.imwrite(str(output_dir / f"right_{ts}.png"), right)
        cv2.imwrite(str(output_dir / f"rect_left_{ts}.png"), rect_l)
        cv2.imwrite(str(output_dir / f"rect_right_{ts}.png"), rect_r)
        cv2.imwrite(str(output_dir / f"depth_color_{ts}.png"), depth_colored)
        np.save(str(output_dir / f"depth_mm_{ts}.npy"), depth_map)
        np.save(str(output_dir / f"disparity_{ts}.npy"), disparity)

        print(f"\nSaved to {output_dir}/:")
        print(f"  left_{ts}.png          - Raw left image")
        print(f"  right_{ts}.png         - Raw right image")
        print(f"  rect_left_{ts}.png     - Rectified left")
        print(f"  rect_right_{ts}.png    - Rectified right")
        print(f"  depth_color_{ts}.png   - Colorized depth map")
        print(f"  depth_mm_{ts}.npy      - Depth in mm (numpy array)")
        print(f"  disparity_{ts}.npy     - Raw disparity (numpy array)")

        # Print some depth stats
        valid = depth_map[depth_map > 0]
        if len(valid) > 0:
            print(f"\n  Depth statistics:")
            print(f"    Min:    {valid.min():.0f}mm ({valid.min()/1000:.2f}m)")
            print(f"    Max:    {valid.max():.0f}mm ({valid.max()/1000:.2f}m)")
            print(f"    Median: {np.median(valid):.0f}mm ({np.median(valid)/1000:.2f}m)")
            print(f"    Valid:  {len(valid)}/{depth_map.size} pixels "
                  f"({100*len(valid)/depth_map.size:.1f}%)")

    finally:
        camera.close()


# ──────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────

def print_usage():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Stereo Depth Mapping — Arducam IMX519 Quad-Camera Kit     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Step 1: Capture calibration images (hold checkerboard)      ║
║    $ python3 stereo_depth.py calibrate                       ║
║                                                              ║
║  Step 2: Compute stereo calibration                          ║
║    $ python3 stereo_depth.py compute_cal                     ║
║                                                              ║
║  Step 3: Run depth mapping                                   ║
║    $ python3 stereo_depth.py depth                           ║
║                                                              ║
║  Other:                                                      ║
║    $ python3 stereo_depth.py capture    (single frame)       ║
║                                                              ║
║  Config: Edit CONFIG dict at top of stereo_depth.py          ║
║  Cameras: Channels 0 (left) and 1 (right) on adapter        ║
║  Baseline: 75mm                                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    mode = sys.argv[1].lower()

    if mode == "calibrate":
        run_calibration_capture(CONFIG)
    elif mode == "compute_cal":
        run_compute_calibration(CONFIG)
    elif mode == "depth":
        run_depth_mapping(CONFIG)
    elif mode == "capture":
        run_capture(CONFIG)
    else:
        print(f"Unknown mode: {mode}")
        print_usage()
        sys.exit(1)
