"""
scripts/autonomy/make_synthetic_jsonl.py

Generate a JSONL file that mimics what `scripts/perception/depth_detect.py`
emits, without needing the perception hardware.

Useful for testing the autonomy stack's DepthDetectSource end-to-end
when a real Pi-rig capture isn't available. The output file has the
exact same schema as a real depth_detect.py run — the planner /
DepthDetectSource can't tell the difference.

Usage:
    python make_synthetic_jsonl.py --out /tmp/session.jsonl

    # Replay against SITL:
    python test_avoidance_jsonl.py --jsonl /tmp/session.jsonl

Default scenario (matches test_avoidance.py's synthetic timeline):
    0..8   s  no detections
    8..16  s  one obstacle ('person' at 0.80 m, centred ahead)
   16..24  s  no detections
Frame rate: 6.8 Hz (matches the dissertation §6.1 measured rate).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


DEFAULT_RATE_HZ = 6.8
DEFAULT_DURATION_S = 24.0
OBSTACLE_START_S = 8.0
OBSTACLE_END_S = 16.0
OBSTACLE_DEPTH_M = 0.80
OBSTACLE_CLASS = "person"
OBSTACLE_CONF = 0.87


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="Output JSONL file path")
    p.add_argument(
        "--rate-hz", type=float, default=DEFAULT_RATE_HZ,
        help=f"Frame rate to emit (default {DEFAULT_RATE_HZ:.1f} = dissertation measured)",
    )
    p.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help=f"Total duration in seconds (default {DEFAULT_DURATION_S})",
    )
    p.add_argument(
        "--obstacle-start", type=float, default=OBSTACLE_START_S,
        help=f"When the obstacle appears (s, default {OBSTACLE_START_S})",
    )
    p.add_argument(
        "--obstacle-end", type=float, default=OBSTACLE_END_S,
        help=f"When the obstacle disappears (s, default {OBSTACLE_END_S})",
    )
    p.add_argument(
        "--obstacle-depth", type=float, default=OBSTACLE_DEPTH_M,
        help=f"Obstacle distance in metres (default {OBSTACLE_DEPTH_M})",
    )
    p.add_argument(
        "--obstacle-centroid-x", type=float, default=0.0,
        help="Obstacle centroid x in [-1, +1] (default 0 = centred ahead)",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = int(args.duration * args.rate_hz)
    frame_dt = 1.0 / args.rate_hz
    # Anchor timestamps to "now" so the JSONL looks like a fresh capture.
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n_frames):
            t = t0 + i * frame_dt
            offset_s = i * frame_dt
            in_obstacle_window = args.obstacle_start <= offset_s < args.obstacle_end

            detections = []
            if in_obstacle_window:
                detections.append(
                    {
                        "class_name": OBSTACLE_CLASS,
                        "confidence": OBSTACLE_CONF,
                        "bbox_xyxy": [560, 280, 720, 440],
                        "depth_m": float(args.obstacle_depth),
                        "bbox_centroid_norm": [
                            float(args.obstacle_centroid_x),
                            0.0,
                        ],
                    }
                )

            record = {
                "t": float(t),
                "detections": detections,
                "camera_hfov_deg": 67.0,
                "camera_vfov_deg": 41.0,
            }
            f.write(json.dumps(record) + "\n")

    print(
        f"Wrote {n_frames} frames over {args.duration:.1f} s "
        f"@ {args.rate_hz:.1f} Hz to {out_path}"
    )
    print(
        f"  obstacle window: {args.obstacle_start:.1f}-{args.obstacle_end:.1f} s "
        f"(class={OBSTACLE_CLASS}, depth={args.obstacle_depth:.2f} m, "
        f"centroid_x={args.obstacle_centroid_x:+.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
