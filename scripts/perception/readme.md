# Perception (AutonoBird)

Where AutonoBird's on-board perception pipeline lives. Currently a single
script for the **pre-calibration YOLO bring-up**; will grow into the full
capture → rectify → SGBM depth → YOLO detection → fusion → decision loop
once stereo calibration is integrated.

## What's here today

- **`yolo_detect.py`** — runs YOLO on the Hailo-8 NPU against the *left half*
  of the AR0144 stereo stream. No calibration required, no depth, no fusion.
  Validates the detection half of the perception pipeline in isolation so the
  stereo-depth layer can be added later without bringing up two unknowns at
  once.

## What's NOT here yet

- Stereo rectification — gated on completing `scripts/ar0144/guided_calibration.py`
- SGBM depth estimation — gated on calibration `.npz` output
- Depth-fused detections ("`person` at 1.4 m") — gated on the two above
- MAVLink integration / decision logic — gated on depth fusion

## Prerequisites

- HailoRT runtime and Python bindings (already installed on the Pi from
  Benchy use; verify with `hailortcli fw-control identify` and
  `python3 -c 'import hailo_platform'`).
- A compiled HEF for Hailo-8. Recommended: a YOLOv8n **detection** HEF
  (COCO 80 classes, NMS in-network) from the
  [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
  Place it at `scripts/perception/models/model.hef` (the default the script
  looks for) or pass `--hef PATH`. The `models/` folder is gitignored via
  `*.hef`.
- AR0144 stereo USB camera enumerated as `/dev/video0` (the same setup the
  calibration scripts use).
- OpenCV + numpy in the active Python environment.

## Quick start

```bash
# 1. (Once) get a Hailo-8 pre-compiled YOLOv8n detection HEF
mkdir -p ~/Documents/AutonoBird/scripts/perception/models
# Place it as models/model.hef (the default the script picks up)

# 2. Confirm Hailo runtime sees the AI HAT+
hailortcli fw-control identify

# 3. Run
cd ~/Documents/AutonoBird/scripts/perception
python3 yolo_detect.py
# Or with overrides:
python3 yolo_detect.py --hef models/model.hef --threshold 0.35
```

## What you should see

A window showing the left AR0144 view with bounding boxes + COCO class
labels + scores drawn on top. The top header strip prints live FPS, NPU
inference time, total loop time, detection count, and confidence threshold.
Console heartbeats every ~3 seconds. Press **Q** to quit.

Expected ballpark on Hailo-8 with `yolov8n.hef`: NPU latency ~10 ms,
end-to-end loop limited by camera capture (~13 fps cap on the USB 2.0
AR0144) and OpenCV display. Detection FPS will track loop FPS, not NPU FPS.

## Assumptions worth knowing

- **NMS is baked into the HEF.** Hailo Model Zoo's `yolo*n.hef` files all
  ship with NMS in-network and output `(B, N, 6)` detections. If you use a
  HEF without baked NMS, `decode_detections()` won't work — you'd need an
  external NMS step (see `hailo-rpi5-examples` for templates).
- **80-class COCO labels.** The pre-trained models detect generic objects
  (person, chair, cup, etc.) — useful for bring-up but not drone-specific.
  Custom-class fine-tuning is out of scope for the current dissertation
  iteration.
- **Left camera only.** Right camera is read by the script (as part of the
  side-by-side AR0144 frame) but discarded. Stereo work uses both halves
  via the calibration `.npz`.
