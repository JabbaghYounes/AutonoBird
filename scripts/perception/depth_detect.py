#!/usr/bin/env python3
"""
=============================================================================
Handheld perception: stereo depth + YOLO detection on Hailo-8
=============================================================================

Full handheld-flight perception pipeline. Reuses the YOLO bring-up from
yolo_detect.py and layers SGBM stereo depth on top so each detected object
is annotated with "<class> <score> @ <distance>m".

Pipeline per frame:
  1. Read 2560x720 stereo frame from AR0144
  2. Split into left/right halves (1280x720 each)
  3. Rectify both halves using the calibration .npz
  4. SGBM disparity at HALF resolution (640x360) for compute headroom
  5. Disparity -> depth in metres using Q matrix focal length + baseline
  6. YOLO on the rectified left at 640x640 (letterboxed)
  7. For each detection, look up median depth in a 5x5 patch around the
     bbox centroid; annotate the box with class + score + distance
  8. Display + heartbeat

Gated on a valid stereo calibration .npz produced by:
    scripts/ar0144/guided_calibration.py capture --session-name <name>
    scripts/ar0144/guided_calibration.py calibrate
Default calibration path is the one those scripts write to.

Usage:
    python3 depth_detect.py [--hef PATH] [--calib PATH]
                            [--threshold 0.4] [--show-depth]

The --show-depth flag opens a second window with the colourised disparity
map alongside the annotated detections — useful for verifying calibration
quality during handheld bring-up.
=============================================================================
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Reuse the YOLO scaffolding from the sibling script. yolo_detect.py is
# import-safe (its main() is behind an __name__ == "__main__" guard).
from yolo_detect import (
    HailoYOLO,
    open_camera,
    letterbox,
    unmap_box,
    color_for_class,
    decode_detections,
    COCO_CLASSES,
    INPUT_SIZE,
    SCRIPT_DIR,
    DEFAULT_HEF,
    DEFAULT_CONFIDENCE,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Calibration .npz path defaults to where guided_calibration.py writes it
DEFAULT_CALIB = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "ar0144",
                 "stereo_calibration_data", "stereo_calibration.npz")
)

# SGBM (same params as guided_calibration.py's depth viewer, halved disparity
# range since we run at half resolution)
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 64        # must be divisible by 16; 64 at half-res ~ 128 at full-res
SGBM_BLOCK_SIZE = 7

# Run SGBM at this scale-down factor of the rectified halves. SGBM cost is
# roughly quadratic in pixel count, so half-res is ~4x faster with mild
# accuracy loss (acceptable for obstacle-class depth).
DEPTH_SCALE = 2

# Median-filter patch around the bbox centroid when looking up depth (px,
# at the half-res depth map's scale). Rejects depth holes / mismatches.
DEPTH_PATCH = 5

# Physical baseline of the AR0144 stereo module (metres)
BASELINE_M = 0.052

# Depth values outside this band are treated as invalid (camera-near jitter
# and far-field nonsense)
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 10.0


# =============================================================================
# CALIBRATION + SGBM
# =============================================================================

def load_calibration(path):
    """Load stereo calibration .npz produced by guided_calibration.py."""
    if not os.path.exists(path):
        print(f"[ERROR] Calibration file not found: {path}")
        print("        Run scripts/ar0144/guided_calibration.py capture + calibrate first.")
        sys.exit(1)

    data = np.load(path)
    required = ["mapL1", "mapL2", "mapR1", "mapR2", "Q"]
    missing = [k for k in required if k not in data.files]
    if missing:
        print(f"[ERROR] Calibration .npz missing keys: {missing}")
        sys.exit(1)

    cal = {
        "mapL1": data["mapL1"], "mapL2": data["mapL2"],
        "mapR1": data["mapR1"], "mapR2": data["mapR2"],
        "Q": data["Q"],
        "rms_error": float(data["rms_error"]) if "rms_error" in data.files else float("nan"),
    }
    # Use the calibration's own translation vector for baseline if present,
    # else fall back to the AR0144 nominal spec. T is in millimetres in the
    # convention guided_calibration.py uses (since objpoints are in mm).
    if "T" in data.files:
        cal["baseline_m"] = float(abs(data["T"][0, 0])) / 1000.0
    else:
        cal["baseline_m"] = BASELINE_M
    return cal


def make_stereo_matcher():
    """SGBM with parameters validated against the calibration script's depth viewer."""
    return cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP,
        numDisparities=SGBM_NUM_DISP,
        blockSize=SGBM_BLOCK_SIZE,
        P1=8 * 3 * SGBM_BLOCK_SIZE ** 2,
        P2=32 * 3 * SGBM_BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )


def lookup_depth(depth_map_half, cx_full, cy_full, patch=DEPTH_PATCH):
    """Return median depth (m) at the bbox centroid, sampled from the half-res
    depth map. Returns NaN if no valid samples are in the patch.
    """
    cx_half = int(cx_full / DEPTH_SCALE)
    cy_half = int(cy_full / DEPTH_SCALE)
    h, w = depth_map_half.shape
    half = patch // 2
    x1 = max(0, cx_half - half)
    x2 = min(w, cx_half + half + 1)
    y1 = max(0, cy_half - half)
    y2 = min(h, cy_half + half + 1)
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    region = depth_map_half[y1:y2, x1:x2]
    valid = region[(region > DEPTH_MIN_M) & (region < DEPTH_MAX_M)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default=DEFAULT_HEF,
                        help=f"Path to compiled HEF (default: {DEFAULT_HEF})")
    parser.add_argument("--calib", default=DEFAULT_CALIB,
                        help=f"Path to stereo calibration .npz (default: {DEFAULT_CALIB})")
    parser.add_argument("--source", type=int, default=0,
                        help="V4L2 camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"YOLO confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--show-depth", action="store_true",
                        help="Open a second window with the colourised disparity map")
    args = parser.parse_args()

    print("=" * 60)
    print("HANDHELD PERCEPTION: STEREO DEPTH + YOLO ON HAILO-8")
    print("=" * 60)

    calib = load_calibration(args.calib)
    print(f"[INFO] Calibration loaded: {args.calib}")
    if not np.isnan(calib["rms_error"]):
        print(f"[INFO] Calibration RMS reprojection error: {calib['rms_error']:.4f} px")
    print(f"[INFO] Baseline: {calib['baseline_m'] * 1000:.2f} mm")

    model = HailoYOLO(args.hef)
    cap = open_camera(args.source)
    matcher = make_stereo_matcher()

    WINDOW_NAME = "Handheld Perception (Stereo Depth + YOLO) - Q to quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    if args.show_depth:
        DEPTH_WIN = "Disparity (colourised)"
        cv2.namedWindow(DEPTH_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEPTH_WIN, 640, 360)

    # Q-matrix derived focal length in rectified pixels. Q[2, 3] = -f
    # (sign depends on convention; abs() is safe). At our half-res depth
    # computation, focal length scales down by DEPTH_SCALE.
    Q = calib["Q"]
    focal_px_full = abs(float(Q[2, 3])) if Q[2, 3] != 0 else 500.0
    focal_px_half = focal_px_full / DEPTH_SCALE
    baseline_m = calib["baseline_m"]

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0
    npu_ms_ema = sgbm_ms_ema = total_ms_ema = 0.0
    last_heartbeat = time.time()
    consecutive_bad_reads = 0
    MAX_BAD_READS = 60

    print("Press Q to quit.\n")

    while True:
        loop_t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_bad_reads += 1
            if consecutive_bad_reads >= MAX_BAD_READS:
                print(f"[ERROR] {consecutive_bad_reads} consecutive bad reads — "
                      "camera disconnected. Exiting.")
                break
            continue
        consecutive_bad_reads = 0

        w = frame.shape[1]
        left_raw = frame[:, : w // 2]
        right_raw = frame[:, w // 2:]

        # Rectify both halves
        rectL = cv2.remap(left_raw, calib["mapL1"], calib["mapL2"], cv2.INTER_LINEAR)
        rectR = cv2.remap(right_raw, calib["mapR1"], calib["mapR2"], cv2.INTER_LINEAR)
        orig_h, orig_w = rectL.shape[:2]

        # SGBM on half-res grayscale
        small_h, small_w = orig_h // DEPTH_SCALE, orig_w // DEPTH_SCALE
        grayL_small = cv2.resize(cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY),
                                  (small_w, small_h))
        grayR_small = cv2.resize(cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY),
                                  (small_w, small_h))

        sgbm_t0 = time.perf_counter()
        disparity = matcher.compute(grayL_small, grayR_small).astype(np.float32) / 16.0
        sgbm_ms = (time.perf_counter() - sgbm_t0) * 1000.0
        sgbm_ms_ema = 0.9 * sgbm_ms_ema + 0.1 * sgbm_ms if sgbm_ms_ema else sgbm_ms

        with np.errstate(divide="ignore", invalid="ignore"):
            depth_map = np.where(
                disparity > 0,
                focal_px_half * baseline_m / disparity,
                0.0,
            ).astype(np.float32)

        # YOLO on the rectified left, full resolution, letterboxed to 640x640
        padded, scale, pad_w, pad_h = letterbox(rectL, INPUT_SIZE)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        npu_t0 = time.perf_counter()
        raw_out = model.infer(rgb)
        npu_ms = (time.perf_counter() - npu_t0) * 1000.0
        npu_ms_ema = 0.9 * npu_ms_ema + 0.1 * npu_ms if npu_ms_ema else npu_ms

        detections = decode_detections(raw_out, args.threshold)

        # Annotate
        display = rectL.copy()
        for det in detections:
            x1, y1, x2, y2 = unmap_box(det["bbox"], scale, pad_w, pad_h,
                                        orig_w, orig_h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth_m = lookup_depth(depth_map, cx, cy)

            class_id = det["class_id"]
            label = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) \
                else f"id{class_id}"
            depth_str = f"{depth_m:.2f}m" if not np.isnan(depth_m) else "?m"
            color = color_for_class(class_id)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.circle(display, (cx, cy), 4, color, -1)
            text = f"{label} {det['score']:.2f} @ {depth_str}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(display, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Loop timing
        total_ms = (time.perf_counter() - loop_t0) * 1000.0
        total_ms_ema = 0.9 * total_ms_ema + 0.1 * total_ms if total_ms_ema else total_ms

        # FPS
        fps_count += 1
        now = time.time()
        if now - fps_t0 >= 0.5:
            fps_display = fps_count / (now - fps_t0)
            fps_t0 = now
            fps_count = 0

        # Header overlay
        strip_h = 28
        cv2.rectangle(display, (0, 0), (orig_w, strip_h), (0, 0, 0), -1)
        cv2.putText(display,
                    f"FPS: {fps_display:4.1f}  NPU: {npu_ms_ema:5.1f}ms  "
                    f"SGBM: {sgbm_ms_ema:5.1f}ms  loop: {total_ms_ema:5.1f}ms  "
                    f"det: {len(detections):2d}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(WINDOW_NAME, display)

        if args.show_depth:
            # Visualise disparity (not depth — disparity colour-maps more
            # cleanly across distance bands)
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = cv2.applyColorMap(np.uint8(disp_vis), cv2.COLORMAP_JET)
            cv2.imshow(DEPTH_WIN, disp_vis)

        # Heartbeat
        if now - last_heartbeat >= 3.0:
            print(f"  [heartbeat] fps={fps_display:.1f} npu={npu_ms_ema:.1f}ms "
                  f"sgbm={sgbm_ms_ema:.1f}ms loop={total_ms_ema:.1f}ms "
                  f"det={len(detections)}")
            last_heartbeat = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Final EMA — NPU: {npu_ms_ema:.1f} ms, "
          f"SGBM: {sgbm_ms_ema:.1f} ms, loop: {total_ms_ema:.1f} ms.")


if __name__ == "__main__":
    main()
