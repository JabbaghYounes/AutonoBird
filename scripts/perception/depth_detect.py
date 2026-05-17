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
import json
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

# Default HEF when --pose is set. The pose model is symlinked from
# ~/Documents/Benchy/resources/hefs/v8_pose_n_hailo8.hef by the Pi-side
# setup; the symlink lands at scripts/perception/models/v8_pose_n_hailo8.hef.
DEFAULT_POSE_HEF = os.path.join(SCRIPT_DIR, "models", "v8_pose_n_hailo8.hef")

# COCO 17-keypoint skeleton (pairs of indices) for visual overlay when in
# --pose mode. Matches the same indices the gesture classifier uses.
COCO_SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),          # shoulders + arms
    (11, 12), (5, 11), (6, 12),                        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),            # legs
    (0, 5), (0, 6),                                    # nose to shoulders
]

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
# POSE DECODING (--pose flag)
# =============================================================================

# YOLOv8-pose Hailo HEFs return one detection per person with bbox + 17
# keypoints. The exact output layout depends on which version of Hailo
# Model Zoo's post-process layer is compiled into the HEF:
#
#   (a) "list of per-class lists" — payload[0] is a list of length 1
#       (pose models have one class: person). Inner list elements are
#       arrays of length ≥ 5 + 17 * 3 = 56:
#         [y_min, x_min, y_max, x_max, score,
#          kp1_y, kp1_x, kp1_v, kp2_y, kp2_x, kp2_v, ..., kp17_y, kp17_x, kp17_v]
#   (b) "dict of separated arrays" — payload[0] is {"bboxes": (N,4),
#       "scores": (N,), "joints": (N,17,3)} or similar.
#   (c) "flat ndarray" — payload is (B, N, 56) or (N, 56) with the same
#       per-row layout as (a).
#
# The decoder below tries each shape in turn. Coordinates returned by
# Hailo Model Zoo's NMS layer are in normalised [0, 1] letterbox-space —
# (0,0) is the top-left of the 640×640 model input, (1,1) is bottom-right.
# Downstream code unmaps to original-frame pixels then to the autonomy
# stack's [-1, +1] frame-centred convention.
#
# UNTESTED ON LIVE HAILO: this function was drafted from the Hailo Model
# Zoo reference layout but has not been exercised against the actual
# v8_pose_n_hailo8.hef. The first Pi run is the validation. If the
# returned list is empty when a person is clearly in frame, dump the
# raw_output keys + shapes to console for diagnosis.


def decode_pose_detections(raw_output, confidence_threshold):
    """Decode YOLOv8-pose Hailo output into [{bbox, score, class_id, keypoints_norm}, ...].

    `bbox` is in normalised letterbox xyxy ([0,1] x [0,1]).
    `keypoints_norm` is a (17, 3) numpy array of (x_norm, y_norm, conf)
    in the same normalised letterbox space, ready for unmap_keypoint().

    Returns an empty list when the output shape doesn't match any known
    layout (caller can then choose to print diagnostics).
    """
    if not raw_output:
        return []

    payload = next(iter(raw_output.values()))
    detections: list[dict] = []

    def _push_row(row):
        """Per-row decode for the flat / list-of-classes layout."""
        if len(row) < 5 + 17 * 3:
            return
        score = float(row[4])
        if score < confidence_threshold:
            return
        # [y_min, x_min, y_max, x_max] → [x_min, y_min, x_max, y_max] (xyxy)
        bbox = [float(row[1]), float(row[0]), float(row[3]), float(row[2])]
        # 51 floats: (kp1_y, kp1_x, kp1_v) × 17 → (17, 3) with (x, y, v)
        kp_flat = np.asarray(row[5:5 + 17 * 3], dtype=float).reshape(17, 3)
        # Swap y,x → x,y; keep visibility/confidence column.
        kp_norm = kp_flat[:, [1, 0, 2]]
        detections.append({
            "bbox": bbox,
            "score": score,
            "class_id": 0,
            "keypoints_norm": kp_norm,
        })

    # Layout (a): list (per batch) → list (per class) → list (per detection)
    if isinstance(payload, list):
        # Hailo outputs are batched even with B=1; unwrap the outer list.
        per_image = payload[0] if payload else None

        # Layout (b): dict of separated arrays.
        if isinstance(per_image, dict):
            bboxes = per_image.get("bboxes")
            scores = per_image.get("scores")
            joints = per_image.get("joints", per_image.get("keypoints"))
            if bboxes is not None and scores is not None and joints is not None:
                for i, score in enumerate(scores):
                    if float(score) < confidence_threshold:
                        continue
                    bbox = bboxes[i]
                    detections.append({
                        # Convert [y_min, x_min, y_max, x_max] → xyxy
                        "bbox": [float(bbox[1]), float(bbox[0]),
                                 float(bbox[3]), float(bbox[2])],
                        "score": float(score),
                        "class_id": 0,
                        "keypoints_norm": np.asarray(joints[i])[:, [1, 0, 2]],
                    })
                return detections

        # Layout (a continued): nested per-class list (pose has 1 class).
        if isinstance(per_image, list):
            classes = per_image if (
                per_image and isinstance(per_image[0], (list, np.ndarray))
                and len(per_image[0]) and isinstance(per_image[0][0], (list, np.ndarray))
            ) else [per_image]
            for class_dets in classes:
                for det in class_dets:
                    _push_row(det)
            return detections

    # Layout (c): flat ndarray.
    if isinstance(payload, np.ndarray):
        arr = payload
        if arr.ndim == 3:
            arr = arr[0]  # (B, N, K) → (N, K), B=1
        if arr.ndim == 2 and arr.shape[1] >= 5 + 17 * 3:
            for row in arr:
                _push_row(row)
            return detections

    # Unknown shape — caller prints diagnostics.
    return detections


def unmap_keypoints(kp_norm, scale, pad_w, pad_h, orig_w, orig_h):
    """Convert (17, 3) letterbox-normalised keypoints to original-frame
    pixel coordinates, returning a (17, 3) array of (x_px, y_px, conf).

    Mirrors unmap_box() but for points. Coordinates outside the model's
    640×640 letterbox padding are clamped to the original image bounds.
    """
    out = np.empty_like(kp_norm)
    for i, (x_n, y_n, v) in enumerate(kp_norm):
        x_px = (x_n * INPUT_SIZE - pad_w) / scale
        y_px = (y_n * INPUT_SIZE - pad_h) / scale
        x_px = max(0.0, min(float(orig_w - 1), float(x_px)))
        y_px = max(0.0, min(float(orig_h - 1), float(y_px)))
        out[i] = (x_px, y_px, float(v))
    return out


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
    parser.add_argument("--jsonl", default=None,
                        help="If set, append one JSON object per processed frame "
                             "to this file. Format matches scripts/autonomy/"
                             "perception_source.PerceptionFrame so the autonomy "
                             "DepthDetectSource can tail it. One detection-set "
                             "per line, line-buffered for live tail.")
    parser.add_argument("--no-gui", action="store_true",
                        help="Skip cv2 window setup. Useful for headless runs "
                             "that only need JSONL emit (e.g. over SSH).")
    parser.add_argument("--pose", action="store_true",
                        help="Use the YOLOv8-pose Hailo model (default HEF "
                             f"{DEFAULT_POSE_HEF}) and emit per-detection "
                             "keypoints into the JSONL — required for the "
                             "autonomy gesture pipeline. Untested on live "
                             "Hailo at first integration; first Pi run is "
                             "the validation.")
    args = parser.parse_args()

    # If --pose is set and the user didn't override --hef, swap to the
    # pose HEF default. (Explicit --hef still wins.)
    if args.pose and args.hef == DEFAULT_HEF:
        args.hef = DEFAULT_POSE_HEF

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

    # Optional JSONL emit. One JSON object per processed frame, matching
    # scripts/autonomy/perception_source.PerceptionFrame's shape:
    #   {"t": <unix s>, "detections": [{...}], "camera_hfov_deg": 67.0, ...}
    # buffering=1 = line-buffered so the autonomy-side tail sees each line
    # as soon as it's written.
    jsonl_file = None
    if args.jsonl:
        # Append-mode: multiple sessions can target the same file without
        # clobbering earlier runs; the consumer can tail from EOF.
        jsonl_file = open(args.jsonl, "a", buffering=1)
        print(f"[INFO] Emitting JSONL detection events to {args.jsonl}")

    if not args.no_gui:
        WINDOW_NAME = "Handheld Perception (Stereo Depth + YOLO) - Q to quit"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        if args.show_depth:
            DEPTH_WIN = "Disparity (colourised)"
            cv2.namedWindow(DEPTH_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DEPTH_WIN, 640, 360)
    else:
        WINDOW_NAME = None
        DEPTH_WIN = None

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

    print("Press Q to quit (GUI mode) or Ctrl+C (any mode).\n")

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

        if args.pose:
            detections = decode_pose_detections(raw_out, args.threshold)
            # Pose models only output one class: person.
            label_override = "person"
        else:
            detections = decode_detections(raw_out, args.threshold)
            label_override = None

        # Build the per-frame record while we annotate: collect each
        # detection's class / score / unmapped bbox / depth / centroid for
        # both visualisation and JSONL emit. This avoids a second pass.
        annotated: list[dict] = []

        # Annotate
        display = rectL.copy()
        for det in detections:
            x1, y1, x2, y2 = unmap_box(det["bbox"], scale, pad_w, pad_h,
                                        orig_w, orig_h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth_m = lookup_depth(depth_map, cx, cy)

            class_id = det["class_id"]
            label = label_override if label_override is not None else (
                COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES)
                else f"id{class_id}"
            )
            depth_str = f"{depth_m:.2f}m" if not np.isnan(depth_m) else "?m"
            color = color_for_class(class_id)

            # Normalised centroid in [-1, 1] from the frame centre, matches
            # PerceptionSource.Detection.bbox_centroid_norm.
            cx_norm = (cx - orig_w / 2.0) / (orig_w / 2.0)
            cy_norm = (cy - orig_h / 2.0) / (orig_h / 2.0)

            entry = {
                "class_name": label,
                "confidence": float(det["score"]),
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "depth_m": None if np.isnan(depth_m) else float(depth_m),
                "bbox_centroid_norm": [float(cx_norm), float(cy_norm)],
            }

            # Pose: add 17-keypoint payload normalised to [-1, +1] frame-
            # centred coords (matches PerceptionSource.Detection.keypoints).
            if args.pose and "keypoints_norm" in det:
                kp_px = unmap_keypoints(det["keypoints_norm"], scale, pad_w,
                                        pad_h, orig_w, orig_h)
                kp_out: list[list[float]] = []
                for x_px, y_px, conf in kp_px:
                    x_n = (x_px - orig_w / 2.0) / (orig_w / 2.0)
                    y_n = (y_px - orig_h / 2.0) / (orig_h / 2.0)
                    kp_out.append([float(x_n), float(y_n), float(conf)])
                    # Draw the keypoint on the display if confident.
                    if conf >= 0.3:
                        cv2.circle(display, (int(x_px), int(y_px)), 3,
                                   (0, 255, 255), -1)
                entry["keypoints"] = kp_out

                # Draw the COCO skeleton in red for visual feedback.
                for a, b in COCO_SKELETON_PAIRS:
                    xa, ya, va = kp_px[a]
                    xb, yb, vb = kp_px[b]
                    if va >= 0.3 and vb >= 0.3:
                        cv2.line(display, (int(xa), int(ya)),
                                 (int(xb), int(yb)), (0, 0, 255), 2)

            annotated.append(entry)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.circle(display, (cx, cy), 4, color, -1)
            text = f"{label} {det['score']:.2f} @ {depth_str}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(display, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # JSONL emit (one line per processed frame). Always emit, even
        # when no detections fired — the autonomy consumer relies on a
        # stream of frames at the perception rate to know perception is
        # alive, not just on detection events.
        if jsonl_file is not None:
            frame_record = {
                "t": time.time(),
                "detections": annotated,
                # AR0144 measured HFOV after cal-4 (dissertation § 6.1). 4:3
                # crop -> VFOV ≈ HFOV * 3/4. Constants for now; could be
                # derived from the calibration .npz in a future revision.
                "camera_hfov_deg": 67.0,
                "camera_vfov_deg": 41.0,
            }
            try:
                jsonl_file.write(json.dumps(frame_record) + "\n")
            except Exception as e:
                print(f"[WARN] JSONL write failed: {e}", file=sys.stderr)

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

        if not args.no_gui:
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

        if not args.no_gui:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()
    if jsonl_file is not None:
        try:
            jsonl_file.close()
        except Exception:
            pass
    print(f"\nDone. Final EMA — NPU: {npu_ms_ema:.1f} ms, "
          f"SGBM: {sgbm_ms_ema:.1f} ms, loop: {total_ms_ema:.1f} ms.")


if __name__ == "__main__":
    main()
