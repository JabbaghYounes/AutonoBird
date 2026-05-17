"""
scripts/autonomy/test_gesture_pipeline.py — gesture stack unit/integration test

End-to-end exercises the gesture pipeline on the dev workstation without
needing the Pi rig + Hailo + AR0144:

    keypoints  ─►  GestureClassifier  ─►  Gesture  ─►  GestureActionMap  ─►  MockOrchestrator

For each of the four supported gestures (STOP / LAND / COME / RECEDE),
this test:
    1. Builds a synthetic PerceptionFrame whose primary person detection
       has keypoints in the canonical pose for that gesture.
    2. Feeds N consecutive frames into the classifier and asserts that
       the smoothed gesture fires by frame N.
    3. Confirms the action map dispatches the gesture to the right method
       on the mock orchestrator.
    4. Resets between gestures so the smoothing history clears.

No SITL, no Hailo, no camera required. Real-perception runs against the
same pipeline once `depth_detect.py` is taught to emit pose keypoints
(scripts/perception/depth_detect.py --pose; future work).

Run:
    source venv/bin/activate
    python test_gesture_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_AUTONOMY_DIR = Path(__file__).resolve().parent
for _d in (str(_AUTONOMY_DIR),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gesture_action_map import GestureActionMap  # noqa: E402
from gesture_classifier import (  # noqa: E402
    ClassifierConfig,
    Gesture,
    GestureClassifier,
)
from perception_source import Detection, PerceptionFrame  # noqa: E402


# ---------------------------------------------------------------------- #
# Canonical keypoint layouts                                             #
# ---------------------------------------------------------------------- #


# COCO 17 indices: nose / left/right (eye, ear, shoulder, elbow, wrist,
# hip, knee, ankle). All in frame-centred [-1, +1]; confidence 0.9 unless
# otherwise stated.

def _person_base() -> list[list[float]]:
    """Standing-upright operator centred in frame. Wrists left blank
    for the gesture to fill in."""
    kp: list[list[float]] = [
        [0.0,  -0.50, 0.9],   # 0  nose
        [-0.03, -0.55, 0.9],  # 1  left eye
        [0.03,  -0.55, 0.9],  # 2  right eye
        [-0.06, -0.55, 0.9],  # 3  left ear
        [0.06,  -0.55, 0.9],  # 4  right ear
        [-0.15, -0.35, 0.9],  # 5  left shoulder
        [0.15,  -0.35, 0.9],  # 6  right shoulder
        [-0.20, -0.10, 0.9],  # 7  left elbow
        [0.20,  -0.10, 0.9],  # 8  right elbow
        [-0.20,  0.15, 0.9],  # 9  left wrist  (relaxed-at-side default)
        [0.20,   0.15, 0.9],  # 10 right wrist (relaxed-at-side default)
        [-0.10,  0.10, 0.9],  # 11 left hip
        [0.10,   0.10, 0.9],  # 12 right hip
        [-0.10,  0.35, 0.9],  # 13 left knee
        [0.10,   0.35, 0.9],  # 14 right knee
        [-0.10,  0.55, 0.9],  # 15 left ankle
        [0.10,   0.55, 0.9],  # 16 right ankle
    ]
    return kp


# Per-gesture wrist positions. Everything else stays as _person_base().
_GESTURE_WRISTS = {
    # STOP: T-pose, wrists outside shoulders at shoulder height.
    "STOP":   {"left_wrist": [-0.55, -0.35, 0.9], "right_wrist": [0.55, -0.35, 0.9]},
    # LAND: arms straight down, wrists below hips.
    "LAND":   {"left_wrist": [-0.12,  0.30, 0.9], "right_wrist": [0.12,  0.30, 0.9]},
    # COME: arms straight up, wrists above nose.
    "COME":   {"left_wrist": [-0.15, -0.75, 0.9], "right_wrist": [0.15, -0.75, 0.9]},
    # RECEDE: arms crossed at chest, wrists cross the midline.
    "RECEDE": {"left_wrist": [0.10,  -0.25, 0.9], "right_wrist": [-0.10, -0.25, 0.9]},
}


def keypoints_for(gesture_name: str) -> list[tuple[float, float, float]]:
    kp = _person_base()
    overrides = _GESTURE_WRISTS.get(gesture_name)
    if overrides is not None:
        kp[9] = overrides["left_wrist"]    # left wrist
        kp[10] = overrides["right_wrist"]  # right wrist
    return [(float(p[0]), float(p[1]), float(p[2])) for p in kp]


def make_frame(gesture_name: str, depth_m: float = 2.0) -> PerceptionFrame:
    """Wrap the canonical-keypoint detection in a PerceptionFrame."""
    det = Detection(
        class_name="person",
        confidence=0.95,
        bbox_xyxy=(280, 100, 360, 540),
        depth_m=depth_m,
        bbox_centroid_norm=(0.0, 0.0),
        keypoints=keypoints_for(gesture_name),
    )
    return PerceptionFrame(t=time.time(), detections=[det])


# ---------------------------------------------------------------------- #
# Mock orchestrator                                                      #
# ---------------------------------------------------------------------- #


class MockOrchestrator:
    """Records intent calls so the test can assert on them."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def command_hold(self) -> None:
        self.calls.append("HOLD")

    def command_resume(self) -> None:
        self.calls.append("RESUME")

    def command_land(self) -> None:
        self.calls.append("LAND")

    def command_rtl(self) -> None:
        self.calls.append("RTL")


# ---------------------------------------------------------------------- #
# Tests                                                                  #
# ---------------------------------------------------------------------- #


def _drive_classifier(
    clf: GestureClassifier,
    gesture_name: str,
    n_frames: int,
) -> list[Gesture]:
    """Feed n_frames copies of gesture's pose; return the per-frame smoothed output."""
    out: list[Gesture] = []
    for _ in range(n_frames):
        out.append(clf.update(make_frame(gesture_name)))
    return out


def test_classifier_each_gesture() -> int:
    print("=== test 1: classifier recognises each gesture ===")
    cfg = ClassifierConfig(smoothing_frames=3)
    for gname in ("STOP", "LAND", "COME", "RECEDE"):
        clf = GestureClassifier(cfg)  # fresh each gesture; no leakage
        outs = _drive_classifier(clf, gname, n_frames=5)
        # First (smoothing_frames - 1) outputs are NONE due to history fill;
        # from frame smoothing_frames onward the classifier should fire.
        head = outs[:cfg.smoothing_frames - 1]
        tail = outs[cfg.smoothing_frames - 1:]
        if any(g != Gesture.NONE for g in head):
            print(f"FAIL {gname}: classifier fired before smoothing window filled "
                  f"(head={[g.name for g in head]})")
            return 1
        expected = Gesture[gname]
        if not all(g == expected for g in tail):
            print(f"FAIL {gname}: tail not all {expected.name}: "
                  f"{[g.name for g in tail]}")
            return 1
        print(f"  PASS {gname:<7s} → smoothed output: {[g.name for g in outs]}")
    return 0


def test_classifier_rejects_noise() -> int:
    print("=== test 2: classifier rejects NONE poses + noisy alternations ===")
    cfg = ClassifierConfig(smoothing_frames=3)
    clf = GestureClassifier(cfg)
    # Build a frame with low-confidence keypoints — classifier should
    # see no valid pose, return NONE.
    bad_kps = [(0.0, 0.0, 0.1)] * 17
    bad_det = Detection(
        class_name="person",
        confidence=0.95,
        bbox_xyxy=(0, 0, 1, 1),
        depth_m=2.0,
        bbox_centroid_norm=(0.0, 0.0),
        keypoints=bad_kps,
    )
    bad_frame = PerceptionFrame(t=time.time(), detections=[bad_det])
    for _ in range(5):
        g = clf.update(bad_frame)
        if g != Gesture.NONE:
            print(f"FAIL: low-conf keypoints produced {g.name}, expected NONE")
            return 1
    # Now alternate STOP / COME — classifier should not lock onto either
    # because the smoothing window never sees two-of-the-same in a row.
    clf = GestureClassifier(cfg)
    for i in range(6):
        gname = "STOP" if i % 2 == 0 else "COME"
        g = clf.update(make_frame(gname))
        if g != Gesture.NONE:
            print(f"FAIL: alternation produced {g.name} at frame {i}")
            return 1
    print("  PASS rejects low-conf poses + alternation noise")
    return 0


def test_action_map_dispatches() -> int:
    print("=== test 3: action map dispatches each gesture exactly once ===")
    mock = MockOrchestrator()
    am = GestureActionMap(mock, cooldown_s=0.0)  # disable cooldown for the test
    expected = {
        Gesture.STOP:   "HOLD",
        Gesture.LAND:   "LAND",
        Gesture.COME:   "RESUME",
        Gesture.RECEDE: "HOLD",  # mapped to hold for the v1 vocabulary
    }
    for gesture, intent in expected.items():
        before = len(mock.calls)
        ok = am.dispatch(gesture)
        if not ok:
            print(f"FAIL {gesture.name}: dispatch returned False")
            return 1
        if mock.calls[-1] != intent:
            print(f"FAIL {gesture.name}: expected {intent}, got {mock.calls[-1]}")
            return 1
        if len(mock.calls) != before + 1:
            print(f"FAIL {gesture.name}: expected one call, got {len(mock.calls)-before}")
            return 1
    # NONE must be a no-op.
    if am.dispatch(Gesture.NONE):
        print("FAIL: NONE gesture dispatched")
        return 1
    print(f"  PASS dispatches: {mock.calls}")
    return 0


def test_action_map_cooldown() -> int:
    print("=== test 4: action map cooldown suppresses re-fires ===")
    mock = MockOrchestrator()
    am = GestureActionMap(mock, cooldown_s=0.5)
    # Two STOPs back-to-back: only the first should reach the orchestrator.
    am.dispatch(Gesture.STOP)
    am.dispatch(Gesture.STOP)
    if mock.calls != ["HOLD"]:
        print(f"FAIL: expected ['HOLD'], got {mock.calls}")
        return 1
    # After cooldown elapses, the next STOP dispatches again.
    time.sleep(0.6)
    am.dispatch(Gesture.STOP)
    if mock.calls != ["HOLD", "HOLD"]:
        print(f"FAIL: expected ['HOLD','HOLD'] after cooldown, got {mock.calls}")
        return 1
    print("  PASS cooldown blocks repeat, expires correctly")
    return 0


def test_end_to_end() -> int:
    print("=== test 5: classifier + action map glued together ===")
    cfg = ClassifierConfig(smoothing_frames=3)
    clf = GestureClassifier(cfg)
    mock = MockOrchestrator()
    am = GestureActionMap(mock, cooldown_s=0.0)
    # Feed: 5 STOP frames, 3 NONE-resetting frames, 5 LAND frames.
    for _ in range(5):
        g = clf.update(make_frame("STOP"))
        am.dispatch(g)
    # Reset with a non-person frame.
    none_frame = PerceptionFrame(t=time.time(), detections=[])
    for _ in range(3):
        clf.update(none_frame)
    for _ in range(5):
        g = clf.update(make_frame("LAND"))
        am.dispatch(g)
    if "HOLD" not in mock.calls or "LAND" not in mock.calls:
        print(f"FAIL: expected HOLD and LAND in calls, got {mock.calls}")
        return 1
    # And HOLD must appear before LAND (STOP came first).
    if mock.calls.index("HOLD") >= mock.calls.index("LAND"):
        print(f"FAIL: ordering wrong: {mock.calls}")
        return 1
    print(f"  PASS end-to-end calls: {mock.calls}")
    return 0


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def main() -> int:
    tests = [
        test_classifier_each_gesture,
        test_classifier_rejects_noise,
        test_action_map_dispatches,
        test_action_map_cooldown,
        test_end_to_end,
    ]
    failures = 0
    for t in tests:
        rc = t()
        if rc != 0:
            failures += 1
    print()
    print("=" * 64)
    if failures == 0:
        print(f"PASS: all {len(tests)} gesture pipeline tests passed")
        return 0
    print(f"FAIL: {failures}/{len(tests)} tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
