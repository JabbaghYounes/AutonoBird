"""
scripts/autonomy/perception_source.py

Perception-input abstraction for the planner.

A `PerceptionSource` is anything that produces `PerceptionFrame` objects
describing what the drone "sees" right now. The planner consumes frames
without caring where they come from. Concrete sources implemented here:

* `SyntheticPerceptionSource` — deterministic, scripted frame timeline.
  Used by `test_avoidance.py` to inject obstacles at known times so the
  closed-loop SITL test is reproducible.

* `DepthDetectSource` — *stub* for now. The intent is to tap on the
  perception subsystem's `depth_detect.py` output (a JSONL stream of
  detection events). The integration requires a small emit-as-JSONL
  addition to `depth_detect.py`; until that lands, this source returns
  empty frames so downstream code can be wired up regardless.

The data contract:

    PerceptionFrame(
        t              = 1.71e9,   # unix seconds, monotonic if possible
        detections     = [
            Detection(
                class_name = "person",
                confidence = 0.87,
                bbox_xyxy  = (320, 180, 480, 540),  # pixels, original frame
                depth_m    = 0.19,                  # metres to bbox centroid
                bbox_centroid_norm = (-0.05, 0.10), # (x, y) in [-1, 1] from frame centre
            ),
            ...
        ],
        depth_map      = None,     # optional 2D ndarray; not used by current planner
        camera_hfov_deg = 67.0,    # AR0144 measured HFOV after cal-4
        camera_vfov_deg = 41.0,    # 4:3-ish derived
    )

The planner only needs `detections[].depth_m` and `bbox_centroid_norm`
to compute an avoidance vector — the rest is reserved for future use
(sector analysis, depth-map ROI cropping, multi-object tracking).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

try:
    # numpy is optional for callers that only use detection lists.
    # SyntheticPerceptionSource doesn't need it.
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------- #
# Data classes                                                           #
# ---------------------------------------------------------------------- #


@dataclass
class Detection:
    """One YOLO detection plus its fused depth, as the planner sees it."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    depth_m: float
    # Normalised (x, y) centroid of the bbox relative to the frame centre:
    #   x = -1 at the left edge, +1 at the right edge
    #   y = -1 at the top edge,  +1 at the bottom edge
    # Computed by the source so the planner doesn't need the original frame size.
    bbox_centroid_norm: tuple[float, float]

    def __repr__(self) -> str:  # pragma: no cover
        x, y = self.bbox_centroid_norm
        return (
            f"Detection({self.class_name} {self.confidence:.2f} "
            f"@ {self.depth_m:.2f}m, centroid=({x:+.2f},{y:+.2f}))"
        )


@dataclass
class PerceptionFrame:
    """One snapshot of perception output."""

    t: float
    detections: list[Detection] = field(default_factory=list)
    depth_map: Optional["np.ndarray"] = None
    camera_hfov_deg: float = 67.0   # AR0144 measured HFOV after cal-4 (§ 6.1)
    camera_vfov_deg: float = 41.0   # derived for the 4:3 sensor crop

    @property
    def closest(self) -> Optional[Detection]:
        """The detection with the smallest depth, or None if empty."""
        if not self.detections:
            return None
        return min(self.detections, key=lambda d: d.depth_m)

    def detections_within(self, max_depth_m: float) -> list[Detection]:
        return [d for d in self.detections if d.depth_m <= max_depth_m]


# ---------------------------------------------------------------------- #
# Source base class                                                      #
# ---------------------------------------------------------------------- #


class PerceptionSource:
    """Base class. Concrete sources push frames into `self._q`."""

    def __init__(self, queue_size: int = 16) -> None:
        self._q: queue.Queue[PerceptionFrame] = queue.Queue(maxsize=queue_size)
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"perception-{type(self).__name__}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "PerceptionSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Consumer-facing API                                                #
    # ------------------------------------------------------------------ #

    def latest(self) -> Optional[PerceptionFrame]:
        """Drain the queue and return the most-recent frame, or None.

        The planner polls at its own rate (~10 Hz) and only cares about
        the freshest perception data — older frames are dropped. This is
        the right shape for reactive control.
        """
        latest: Optional[PerceptionFrame] = None
        while True:
            try:
                latest = self._q.get_nowait()
            except queue.Empty:
                break
        return latest

    def frames(self, timeout: Optional[float] = None) -> Iterator[PerceptionFrame]:
        """Iterator over frames, blocking up to `timeout` seconds each."""
        while not self._stop_flag.is_set():
            try:
                yield self._q.get(timeout=timeout)
            except queue.Empty:
                return

    # ------------------------------------------------------------------ #
    # Subclass hook                                                      #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Override in subclasses. Must respect `self._stop_flag`."""
        raise NotImplementedError

    def _publish(self, frame: PerceptionFrame) -> None:
        """Push a frame, dropping the oldest if the queue is full."""
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(frame)
            except queue.Empty:
                pass


# ---------------------------------------------------------------------- #
# Synthetic source — scripted frame timeline                             #
# ---------------------------------------------------------------------- #


# A scripted "event" the synthetic source replays. Each entry says
# "starting at offset T seconds from .start(), emit frames containing
# these detections at `rate_hz` Hz until the next event".
@dataclass
class SyntheticEvent:
    offset_s: float
    detections: list[Detection] = field(default_factory=list)


# Convenience constructor for the common "single obstacle ahead at D m" case.
def obstacle_ahead(depth_m: float, class_name: str = "person", confidence: float = 0.9) -> Detection:
    """One obstacle centred in the frame at the given depth."""
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(560, 280, 720, 440),  # centred-ish bbox on a 1280x720 frame
        depth_m=depth_m,
        bbox_centroid_norm=(0.0, 0.0),
    )


def obstacle_offset(
    depth_m: float,
    centroid_x_norm: float,
    centroid_y_norm: float = 0.0,
    class_name: str = "person",
    confidence: float = 0.9,
) -> Detection:
    """An obstacle at the given depth, offset within the frame.

    centroid_x_norm in [-1, 1]: negative = left of centre, positive = right.
    The planner uses this to decide which way to sidestep.
    """
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(0, 0, 0, 0),  # bbox is incidental for the planner
        depth_m=depth_m,
        bbox_centroid_norm=(centroid_x_norm, centroid_y_norm),
    )


class SyntheticPerceptionSource(PerceptionSource):
    """Replay a scripted timeline of obstacle events.

    Events are absolute offsets from `start()`. Between events, the source
    emits frames at `rate_hz` Hz carrying the most recently-activated
    event's detections (or an empty list before the first event).

    Example::

        src = SyntheticPerceptionSource(
            events=[
                SyntheticEvent(0.0, []),                                  # clear
                SyntheticEvent(10.0, [obstacle_ahead(0.8)]),             # block ahead at 0.8m
                SyntheticEvent(15.0, []),                                 # clear again
            ],
            rate_hz=10.0,
        )
        src.start()
        # planner reads src.latest() at its loop rate
    """

    def __init__(
        self,
        events: Sequence[SyntheticEvent],
        rate_hz: float = 10.0,
        queue_size: int = 16,
    ) -> None:
        super().__init__(queue_size=queue_size)
        if not events:
            raise ValueError("events must contain at least one entry")
        # Ensure sorted by offset.
        self._events = sorted(events, key=lambda e: e.offset_s)
        if self._events[0].offset_s > 0:
            # Prepend a "no detections" event at t=0 if not specified.
            self._events.insert(0, SyntheticEvent(0.0, []))
        self._period = 1.0 / rate_hz
        self._start_t: Optional[float] = None

    def _active_event(self, now: float) -> SyntheticEvent:
        """Return the latest event whose offset has elapsed."""
        assert self._start_t is not None
        elapsed = now - self._start_t
        active = self._events[0]
        for e in self._events:
            if e.offset_s <= elapsed:
                active = e
            else:
                break
        return active

    def _run(self) -> None:
        self._start_t = time.time()
        while not self._stop_flag.is_set():
            now = time.time()
            event = self._active_event(now)
            frame = PerceptionFrame(
                t=now,
                detections=list(event.detections),
            )
            self._publish(frame)
            self._stop_flag.wait(self._period)


# ---------------------------------------------------------------------- #
# DepthDetect source — stub                                              #
# ---------------------------------------------------------------------- #


class DepthDetectSource(PerceptionSource):
    """Tail `scripts/perception/depth_detect.py`'s JSONL output.

    depth_detect.py (run with ``--jsonl PATH``) writes one JSON record per
    processed frame to PATH. This source opens that file and yields
    PerceptionFrames as new lines appear.

    Two operating modes:

    - **Replay** (``tail_from_end=False``, the default): read from the
      start of the file. Useful for replaying a recorded handheld
      perception session into the planner against SITL.
    - **Live tail** (``tail_from_end=True``): seek to EOF and only emit
      frames written after this source started. Useful when running
      depth_detect.py concurrently — gives the planner only fresh
      detections rather than racing through a backlog.

    The source is robust to the file not yet existing on start (waits
    up to ``startup_timeout_s`` for it) and to malformed lines (skips
    them). Detections with ``depth_m=null`` (the perception side
    couldn't resolve depth for that bbox) are dropped — the planner
    only sees usable distances.
    """

    def __init__(
        self,
        jsonl_path: str,
        tail_from_end: bool = False,
        startup_timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        queue_size: int = 16,
    ) -> None:
        super().__init__(queue_size=queue_size)
        self._jsonl_path = jsonl_path
        self._tail_from_end = bool(tail_from_end)
        self._startup_timeout_s = float(startup_timeout_s)
        self._poll_interval_s = float(poll_interval_s)

    def _wait_for_file(self) -> bool:
        """Block until the JSONL file exists, or stop / timeout."""
        deadline = time.time() + self._startup_timeout_s
        while not self._stop_flag.is_set():
            if os.path.exists(self._jsonl_path):
                return True
            if time.time() > deadline:
                return False
            self._stop_flag.wait(0.1)
        return False

    def _parse_line(self, line: str) -> Optional[PerceptionFrame]:
        """Parse one JSONL record into a PerceptionFrame. None on bad input."""
        line = line.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None

        detections_raw = obj.get("detections") or []
        detections: list[Detection] = []
        for d in detections_raw:
            depth = d.get("depth_m")
            if depth is None:
                # Perception couldn't resolve depth (out-of-band, hole in
                # disparity map). Skip — planner needs a number.
                continue
            try:
                detections.append(
                    Detection(
                        class_name=str(d["class_name"]),
                        confidence=float(d["confidence"]),
                        bbox_xyxy=tuple(int(v) for v in d["bbox_xyxy"]),  # type: ignore[arg-type]
                        depth_m=float(depth),
                        bbox_centroid_norm=tuple(
                            float(v) for v in d["bbox_centroid_norm"]
                        ),  # type: ignore[arg-type]
                    )
                )
            except (KeyError, TypeError, ValueError):
                # Skip malformed detection objects but keep the frame.
                continue

        return PerceptionFrame(
            t=float(obj.get("t", time.time())),
            detections=detections,
            camera_hfov_deg=float(obj.get("camera_hfov_deg", 67.0)),
            camera_vfov_deg=float(obj.get("camera_vfov_deg", 41.0)),
        )

    def _run(self) -> None:
        if not self._wait_for_file():
            # File never appeared. Stay alive and emit empty frames so
            # the planner doesn't crash on missing perception — same
            # contract as the old stub.
            while not self._stop_flag.is_set():
                self._publish(PerceptionFrame(t=time.time()))
                self._stop_flag.wait(0.5)
            return

        try:
            f = open(self._jsonl_path, "r", encoding="utf-8")
        except OSError:
            return

        try:
            if self._tail_from_end:
                f.seek(0, os.SEEK_END)

            while not self._stop_flag.is_set():
                line = f.readline()
                if not line:
                    # EOF — wait for more lines to be appended.
                    self._stop_flag.wait(self._poll_interval_s)
                    continue
                frame = self._parse_line(line)
                if frame is not None:
                    self._publish(frame)
        finally:
            f.close()


# ---------------------------------------------------------------------- #
# Smoke-test helper — `python -m scripts.autonomy.perception_source`     #
# ---------------------------------------------------------------------- #


def _self_test() -> None:
    """Print a few frames from a SyntheticPerceptionSource for sanity."""
    events = [
        SyntheticEvent(0.0, []),
        SyntheticEvent(1.0, [obstacle_ahead(0.8)]),
        SyntheticEvent(2.0, []),
    ]
    with SyntheticPerceptionSource(events, rate_hz=5.0) as src:
        deadline = time.time() + 3.5
        while time.time() < deadline:
            f = src.latest()
            if f is not None:
                msg = f"  closest={f.closest}" if f.closest else "  no detections"
                print(f"[t+{time.time() - src._start_t:.2f}s]{msg}")
            time.sleep(0.25)


def _jsonl_roundtrip_test() -> None:
    """Smoke-test DepthDetectSource by writing a small JSONL file and
    reading it back through the source. No SITL or perception process
    needed — just verifies the parser + queueing.
    """
    import tempfile

    records = [
        # Frame 1: no detections
        {"t": 1.0, "detections": [], "camera_hfov_deg": 67.0, "camera_vfov_deg": 41.0},
        # Frame 2: one valid detection
        {
            "t": 1.1,
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.87,
                    "bbox_xyxy": [560, 280, 720, 440],
                    "depth_m": 0.85,
                    "bbox_centroid_norm": [0.0, 0.0],
                },
            ],
        },
        # Frame 3: one valid + one with null depth (should be dropped)
        {
            "t": 1.2,
            "detections": [
                {
                    "class_name": "laptop",
                    "confidence": 0.42,
                    "bbox_xyxy": [400, 300, 500, 400],
                    "depth_m": 0.67,
                    "bbox_centroid_norm": [-0.3, 0.1],
                },
                {
                    "class_name": "chair",
                    "confidence": 0.55,
                    "bbox_xyxy": [100, 100, 200, 200],
                    "depth_m": None,
                    "bbox_centroid_norm": [-0.7, -0.3],
                },
            ],
        },
        # Frame 4: malformed line — should be silently skipped (we test
        # by writing it as a raw broken line below, not in this list).
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write("this is not json\n")        # malformed line
        f.write("\n")                          # blank line
        path = f.name

    print(f"Test JSONL at {path}")
    src = DepthDetectSource(jsonl_path=path, tail_from_end=False)
    src.start()

    # Give the tailer a moment to read the whole file.
    time.sleep(0.5)
    src.stop()

    # Drain all frames (newest first via latest(), so use frames())
    frames: list[PerceptionFrame] = []
    while True:
        try:
            frames.append(src._q.get_nowait())  # type: ignore[attr-defined]
        except queue.Empty:
            break

    print(f"Got {len(frames)} frames (expected 3 — one malformed + one blank dropped):")
    for fr in frames:
        msg = (
            f"  t={fr.t:.2f} detections={len(fr.detections)}: "
            + ", ".join(f"{d.class_name}@{d.depth_m:.2f}m" for d in fr.detections)
        )
        print(msg)
    assert len(frames) == 3, f"expected 3 frames, got {len(frames)}"
    # Frame 0: no detections
    assert frames[0].detections == []
    # Frame 1: one detection, depth 0.85
    assert len(frames[1].detections) == 1
    assert frames[1].detections[0].class_name == "person"
    assert frames[1].detections[0].depth_m == 0.85
    # Frame 2: chair (null depth) dropped, laptop kept
    assert len(frames[2].detections) == 1
    assert frames[2].detections[0].class_name == "laptop"
    print("PASS: JSONL round-trip parses correctly.")

    os.unlink(path)


if __name__ == "__main__":
    import sys
    if "--jsonl" in sys.argv:
        _jsonl_roundtrip_test()
    else:
        _self_test()
