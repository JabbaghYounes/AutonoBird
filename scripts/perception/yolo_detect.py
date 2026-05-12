#!/usr/bin/env python3
"""
=============================================================================
YOLO detection bring-up on Hailo-8 (AI HAT+) — pre-calibration test
=============================================================================

Feeds only the LEFT half of the AR0144 stereo stream to a YOLO model running
on the Hailo-8 NPU. No stereo, no depth — this validates the detection
pipeline in isolation so the stereo-depth + fusion layer can be bolted on
later without bringing up two unknowns at once.

What this script does:
  1. Opens the AR0144 in 2560x720 MJPG (same setup as guided_calibration.py)
  2. Takes the left 1280x720 half
  3. Letterboxes to 640x640 for YOLO input
  4. Runs inference on the Hailo-8 via HailoRT
  5. Decodes NMS-baked output, un-maps boxes to the original image
  6. Draws boxes + class labels + score; shows FPS and NPU latency

What this script does NOT do:
  - Stereo rectification (needs calibration)
  - Depth estimation (needs calibration)
  - "Object N metres away" claims (needs calibration)
  - Custom NMS (assumes the HEF has NMS baked in — Hailo Model Zoo standard)

Usage:
    python3 yolo_detect.py [--hef PATH] [--source CAM_INDEX] [--threshold 0.4]

Requires:
    - HailoRT and the hailo_platform Python module (already installed on this
      Pi from Benchy; verify with `hailortcli fw-control identify`).
    - A pre-compiled HEF for Hailo-8. Recommended: yolov8n.hef from the
      Hailo Model Zoo (COCO 80 classes, NMS in-network).
        https://github.com/hailo-ai/hailo_model_zoo
      Place it at ~/hailo_models/yolov8n.hef or pass --hef explicitly.
=============================================================================
"""

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

try:
    import hailo_platform as hpf
except ImportError as e:
    print("[ERROR] hailo_platform Python module is not available.")
    print("        On the calibration rig this should already be installed via")
    print("        the Hailo SDK that Benchy uses. Verify with:")
    print("            python3 -c 'import hailo_platform; print(hailo_platform.__version__)'")
    print(f"        Underlying ImportError: {e}")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HEF = os.path.join(SCRIPT_DIR, "models", "model.hef")
DEFAULT_CONFIDENCE = 0.4
INPUT_SIZE = 640  # YOLOv8/v11 standard square input
CAMERA_INDEX = 0

# COCO 80-class labels (standard for YOLO pre-trained models from Hailo Model Zoo)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


# =============================================================================
# CAMERA
# =============================================================================

def open_camera(index=CAMERA_INDEX):
    """Open AR0144 in 2560x720 MJPG — same path as calibration scripts."""
    device = f"/dev/video{index}"
    try:
        subprocess.run(
            ["v4l2-ctl", "--device", device,
             "--set-fmt-video=width=2560,height=720,pixelformat=MJPG"],
            check=False, capture_output=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {index}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(1)
    return cap


# =============================================================================
# IMAGE PREP / UN-MAP
# =============================================================================

def letterbox(image, target_size=INPUT_SIZE):
    """Resize+pad to target_size×target_size preserving aspect ratio.
    Returns (padded_image, scale, pad_w, pad_h) so detections can be unmapped.
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, pad_w, pad_h


def unmap_box(box, scale, pad_w, pad_h, orig_w, orig_h):
    """Convert a letterboxed bbox back to original-image coordinates.
    Accepts either normalised (0-1) or pixel-space coordinates.
    """
    x1, y1, x2, y2 = box
    if max(x1, y1, x2, y2) <= 1.0:
        x1 *= INPUT_SIZE; y1 *= INPUT_SIZE
        x2 *= INPUT_SIZE; y2 *= INPUT_SIZE
    x1 = max(0, int((x1 - pad_w) / scale))
    y1 = max(0, int((y1 - pad_h) / scale))
    x2 = min(orig_w, int((x2 - pad_w) / scale))
    y2 = min(orig_h, int((y2 - pad_h) / scale))
    return x1, y1, x2, y2


def color_for_class(class_id):
    """Deterministic per-class colour."""
    rng = np.random.RandomState(class_id * 7 + 13)
    return tuple(int(c) for c in rng.randint(60, 256, size=3))


# =============================================================================
# HAILO INFERENCE WRAPPER
# =============================================================================

class HailoYOLO:
    """Synchronous Hailo inference wrapper.

    Assumes the HEF was compiled with NMS in-network (Hailo Model Zoo
    convention for yolo*n.hef), outputting up to N detections as
    [x1, y1, x2, y2, score, class_id] in either normalised (0-1) or
    input-pixel space. If your HEF doesn't bake NMS, you'll need a
    separate post-processing step; see Hailo's example apps for that.
    """

    def __init__(self, hef_path):
        if not os.path.exists(hef_path):
            print(f"[ERROR] HEF not found: {hef_path}")
            print("        Download a pre-compiled YOLOv8n HEF for Hailo-8:")
            print("        https://github.com/hailo-ai/hailo_model_zoo")
            sys.exit(1)

        self.hef = hpf.HEF(hef_path)
        self.target = hpf.VDevice()
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef=self.hef, interface=hpf.HailoStreamInterface.PCIe,
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = hpf.InputVStreamParams.make(
            self.network_group, format_type=hpf.FormatType.UINT8,
        )
        self.output_vstreams_params = hpf.OutputVStreamParams.make(
            self.network_group, format_type=hpf.FormatType.FLOAT32,
        )

        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = input_info.name
        print(f"[INFO] HEF loaded: {hef_path}")
        print(f"[INFO] Input  '{self.input_name}' shape {input_info.shape}")
        for out_info in self.hef.get_output_vstream_infos():
            print(f"[INFO] Output '{out_info.name}' shape {out_info.shape}")

    def infer(self, image_hwc_uint8):
        """Run one frame. Returns dict {output_name: ndarray}."""
        input_data = {self.input_name: np.expand_dims(image_hwc_uint8, axis=0)}
        with hpf.InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        ) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                outputs = infer_pipeline.infer(input_data)
        return outputs


def decode_detections(raw_output, confidence_threshold):
    """Extract detections from an NMS-baked single-output tensor.

    Handles the common Hailo Model Zoo conventions:
      - (1, N, 6): N detections, each [x1, y1, x2, y2, score, class_id]
      - (N, 6): same without batch dim
    """
    tensor = next(iter(raw_output.values()))
    while tensor.ndim > 2:
        tensor = tensor[0]
    detections = []
    for row in tensor:
        if len(row) < 6:
            continue
        x1, y1, x2, y2, score, class_id = row[:6]
        if score < confidence_threshold:
            continue
        detections.append({
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "score": float(score),
            "class_id": int(class_id),
        })
    return detections


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default=DEFAULT_HEF,
                        help=f"Path to compiled HEF (default: {DEFAULT_HEF})")
    parser.add_argument("--source", type=int, default=CAMERA_INDEX,
                        help=f"V4L2 camera index (default: {CAMERA_INDEX})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE})")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLO BRING-UP ON HAILO-8 (LEFT CAMERA ONLY)")
    print("=" * 60)

    model = HailoYOLO(args.hef)
    cap = open_camera(args.source)

    WINDOW_NAME = "YOLO on Hailo-8 (Left Camera) - Q to quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0
    npu_ms_ema = 0.0
    total_ms_ema = 0.0
    last_heartbeat = time.time()

    print("Press Q to quit.\n")

    while True:
        loop_t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Left half only (same split as guided_calibration.py)
        w = frame.shape[1]
        left = frame[:, : w // 2]
        orig_h, orig_w = left.shape[:2]

        # Letterbox + BGR->RGB for YOLO input
        padded, scale, pad_w, pad_h = letterbox(left, INPUT_SIZE)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Inference
        t0 = time.perf_counter()
        raw_out = model.infer(rgb)
        npu_ms = (time.perf_counter() - t0) * 1000.0
        npu_ms_ema = 0.9 * npu_ms_ema + 0.1 * npu_ms if npu_ms_ema else npu_ms

        detections = decode_detections(raw_out, args.threshold)

        # Draw on full-res left frame
        display = left.copy()
        for det in detections:
            x1, y1, x2, y2 = unmap_box(det["bbox"], scale, pad_w, pad_h,
                                        orig_w, orig_h)
            class_id = det["class_id"]
            label = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) \
                else f"id{class_id}"
            color = color_for_class(class_id)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {det['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(display, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # End-of-loop timing
        total_ms = (time.perf_counter() - loop_t0) * 1000.0
        total_ms_ema = 0.9 * total_ms_ema + 0.1 * total_ms if total_ms_ema else total_ms

        # FPS calc
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
                    f"loop: {total_ms_ema:5.1f}ms  "
                    f"det: {len(detections):2d}  thr: {args.threshold:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        cv2.imshow(WINDOW_NAME, display)

        # Console heartbeat every 3 s
        if now - last_heartbeat >= 3.0:
            print(f"  [heartbeat] fps={fps_display:.1f} "
                  f"npu={npu_ms_ema:.1f}ms loop={total_ms_ema:.1f}ms "
                  f"det={len(detections)}")
            last_heartbeat = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Final EMA: NPU {npu_ms_ema:.1f} ms, loop {total_ms_ema:.1f} ms.")


if __name__ == "__main__":
    main()
