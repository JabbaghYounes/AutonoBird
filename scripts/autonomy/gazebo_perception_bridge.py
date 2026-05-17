"""
scripts/autonomy/gazebo_perception_bridge.py

Bridge from a Gazebo Harmonic gpu_lidar topic to AutonoBird's PerceptionFrame
JSONL pipe (the same one `depth_detect.py --jsonl` writes, that the autonomy
stack's `DepthDetectSource` tails).

Subscribes to a LaserScan topic via `gz topic -e --json-output`, parses each
scan, finds the closest in-range return in the forward cone, and emits one
PerceptionFrame record per scan. If no return is closer than the obstacle
window (--max-range default 10 m), an empty-detections frame is emitted
(autonomy reads "no obstacle"). The bridge is the Gazebo-side analogue of
the AR0144 + SGBM + YOLO depth-fusion pipeline — only the sensor backend
differs.

Pre-flight:
    1. Gazebo running with the iris_with_lidar model spawned (the world
       file `iris_obstacle.sdf` does this automatically).
    2. ArduPilot SITL connected over JSON.
    3. Autonomy venv active (only stdlib + nothing else needed here).

Run:
    python gazebo_perception_bridge.py --jsonl /tmp/gazebo_perception.jsonl
    # then in another terminal:
    python test_avoidance_gazebo.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--topic",
        default="/iris/forward_lidar",
        help="Gazebo LaserScan topic to subscribe to.",
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path("/tmp/gazebo_perception.jsonl"),
        help="Output JSONL file (line-buffered append).",
    )
    ap.add_argument(
        "--max-range",
        type=float,
        default=10.0,
        help="Beyond this range, scans are treated as 'no obstacle'.",
    )
    ap.add_argument(
        "--hfov-deg",
        type=float,
        default=67.0,
        help="Horizontal FOV emitted into PerceptionFrame metadata.",
    )
    ap.add_argument(
        "--vfov-deg",
        type=float,
        default=41.0,
        help="Vertical FOV emitted into PerceptionFrame metadata.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-scan stdout logging (still writes JSONL).",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------- #
# gz topic JSON parsing                                                  #
# ---------------------------------------------------------------------- #


def gz_topic_subprocess(topic: str) -> subprocess.Popen:
    """Spawn `gz topic -e --json-output -t TOPIC` with line-buffered stdout."""
    cmd = ["gz", "topic", "-e", "--json-output", "-t", topic]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )


def iter_json_messages(proc: subprocess.Popen):
    """Yield one JSON object per gz topic message.

    `gz topic --json-output` may print each message on a single line or
    spread across multiple lines. We accumulate stdout and yield once
    `{` and `}` counts balance.
    """
    assert proc.stdout is not None
    buf: list[str] = []
    depth = 0
    started = False
    for line in proc.stdout:
        if not started:
            if "{" not in line:
                continue
            started = True
        buf.append(line)
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = "".join(buf)
                    buf = []
                    started = False
                    try:
                        yield json.loads(text)
                    except json.JSONDecodeError:
                        # gz may interleave info lines; skip malformed.
                        break
                    break


# ---------------------------------------------------------------------- #
# LaserScan -> PerceptionFrame                                           #
# ---------------------------------------------------------------------- #


def scan_to_perception_frame(
    scan: dict,
    max_range: float,
    hfov_deg: float,
    vfov_deg: float,
) -> dict:
    """Convert one gz.msgs.LaserScan JSON object to a PerceptionFrame dict.

    Detection schema matches scripts/autonomy/perception_source.PerceptionFrame:
        t: unix seconds
        detections: [{class_name, confidence, bbox_xyxy, depth_m, bbox_centroid_norm}]
        camera_hfov_deg, camera_vfov_deg
    """
    angle_min = float(scan.get("angle_min", -0.585))
    angle_max = float(scan.get("angle_max", 0.585))
    ranges = scan.get("ranges", []) or []
    n = len(ranges)
    # Build the per-ray angles. angle_step may be present or derivable.
    if n > 1:
        angle_step = (angle_max - angle_min) / (n - 1)
    else:
        angle_step = 0.0
    angles = [angle_min + i * angle_step for i in range(n)]

    # Find the closest in-range return.
    closest_idx = -1
    closest_r = float("inf")
    for i, r in enumerate(ranges):
        try:
            r_f = float(r)
        except (TypeError, ValueError):
            continue
        if r_f <= 0.0 or not (r_f == r_f):  # NaN check
            continue
        if r_f < closest_r and r_f <= max_range:
            closest_r = r_f
            closest_idx = i

    detections: list[dict] = []
    if closest_idx >= 0:
        ang = angles[closest_idx]
        # Normalised horizontal centroid in [-1, +1] from frame centre, to
        # match the convention used by depth_detect.py and the synthetic
        # source (see scripts/autonomy/perception_source.py docstring). The
        # planner reads bbox_centroid_norm[0] and treats +x as "obstacle on
        # right".
        abs_max = max(abs(angle_min), abs(angle_max))
        if abs_max > 0:
            cx_norm = ang / abs_max
        else:
            cx_norm = 0.0
        cy_norm = 0.0  # lidar is single-row; no vertical info
        # bbox_xyxy is documented as image pixels (ints). Lidar has no
        # image; fake a 640x360 grid so the planner's optional bbox checks
        # have plausible values. The planner doesn't actually use bbox_xyxy.
        half_w_px = 5
        center_px = int(round((cx_norm + 1.0) * 0.5 * 640))
        detections.append(
            {
                "class_name": "obstacle",
                "confidence": 1.0,
                "bbox_xyxy": [
                    max(0, center_px - half_w_px),
                    170,
                    min(640, center_px + half_w_px),
                    190,
                ],
                "depth_m": round(closest_r, 3),
                "bbox_centroid_norm": [cx_norm, cy_norm],
            }
        )

    # Stamp: prefer the scan's header.stamp if present; else wall clock.
    header = scan.get("header") or {}
    stamp = header.get("stamp") or {}
    try:
        t = float(stamp.get("sec", 0)) + float(stamp.get("nsec", 0)) * 1e-9
        if t <= 0.0:
            t = time.time()
    except (TypeError, ValueError):
        t = time.time()

    return {
        "t": t,
        "detections": detections,
        "camera_hfov_deg": hfov_deg,
        "camera_vfov_deg": vfov_deg,
    }


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def run() -> int:
    args = parse_args()

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gazebo-bridge] topic       : {args.topic}", flush=True)
    print(f"[gazebo-bridge] jsonl       : {args.jsonl}", flush=True)
    print(f"[gazebo-bridge] max range   : {args.max_range:.1f} m", flush=True)

    proc = gz_topic_subprocess(args.topic)
    n_emitted = 0
    n_with_detection = 0
    t0 = time.time()
    try:
        with args.jsonl.open("a", buffering=1) as fout:
            for scan in iter_json_messages(proc):
                frame = scan_to_perception_frame(
                    scan,
                    max_range=args.max_range,
                    hfov_deg=args.hfov_deg,
                    vfov_deg=args.vfov_deg,
                )
                fout.write(json.dumps(frame) + "\n")
                n_emitted += 1
                if frame["detections"]:
                    n_with_detection += 1
                if not args.quiet and n_emitted % 10 == 0:
                    rate = n_emitted / max(time.time() - t0, 1e-3)
                    if frame["detections"]:
                        d = frame["detections"][0]
                        msg = (
                            f"emit {n_emitted}  rate {rate:.1f} Hz  "
                            f"obstacle @ {d['depth_m']:.2f} m bbox_x={d['bbox_centroid_norm']['x']:.2f}"
                        )
                    else:
                        msg = (
                            f"emit {n_emitted}  rate {rate:.1f} Hz  "
                            f"no obstacle in range"
                        )
                    print(f"[gazebo-bridge] {msg}", flush=True)
    except KeyboardInterrupt:
        print(
            f"[gazebo-bridge] interrupted: {n_emitted} frames "
            f"({n_with_detection} with detections)",
            flush=True,
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Surface any stderr from gz topic (mismatched topic, no publishers,
        # etc.) so users can see it.
        if proc.stderr is not None:
            err = proc.stderr.read()
            if err.strip():
                print("[gazebo-bridge] gz topic stderr:", flush=True)
                print(err, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
