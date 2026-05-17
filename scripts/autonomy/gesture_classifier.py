"""
scripts/autonomy/gesture_classifier.py — body-pose gesture recognition

Consumes the `keypoints` field on a Detection (17 COCO body keypoints in
frame-centred [-1, +1] coordinates, produced by a YOLOv*-pose model — see
`scripts/perception/depth_detect.py --pose` when that's wired up on the
Pi rig). Classifies the operator's pose into one of a small, safety-
critical vocabulary:

    STOP        — T-pose, both arms straight out horizontal.
                  Drone holds position (planner pauses, vehicle hovers).
    LAND        — Both arms held straight down at the sides.
                  Drone descends and lands at its current XY position.
    COME        — Both arms held straight up overhead.
                  Drone moves toward the operator and resumes cruise / follow.
    RECEDE      — Arms crossed in front of chest (X pose).
                  Drone backs off — planner cruises away from the operator.

The classifier is deliberately conservative:

  * **Per-keypoint confidence gating.** Any keypoint below
    `min_keypoint_confidence` is treated as missing — gestures that rely
    on a missing keypoint return NONE rather than fire on noise.
  * **Temporal smoothing.** A gesture only "fires" once the same answer
    has been observed for `smoothing_frames` consecutive frames. Single-
    frame false positives don't reach the action map.
  * **Single primary operator.** When multiple `person` detections exist
    in a frame, only the closest (by stereo depth) is considered the
    operator. Bystanders don't trigger commands.

Why rule-based: with only four safety-critical gestures, hand-tuned rules
on body keypoints are accurate, debuggable, and don't require a
training dataset we don't have. A learned classifier (small MLP on the
keypoint vector) is a clean upgrade path if the vocabulary grows past
~10 gestures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from perception_source import Detection, PerceptionFrame


# COCO 17-keypoint indices.
class KP:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Gesture(Enum):
    NONE = "NONE"
    STOP = "STOP"
    LAND = "LAND"
    COME = "COME"
    RECEDE = "RECEDE"


@dataclass
class ClassifierConfig:
    # Per-keypoint confidence below this is treated as missing.
    min_keypoint_confidence: float = 0.5
    # Number of consecutive frames a gesture must persist before firing.
    smoothing_frames: int = 3
    # STOP: |wrist.y - shoulder.y| ≤ this counts as "horizontal arm".
    stop_vertical_tolerance: float = 0.10
    # STOP: |wrist.x - shoulder.x| ≥ this counts as "arm extended".
    stop_horizontal_extension: float = 0.15
    # LAND: wrist.y > hip.y + this counts as "arm down".
    land_below_hip_offset: float = 0.10
    # COME: nose.y - wrist.y ≥ this counts as "arm overhead".
    come_above_head_offset: float = 0.15
    # RECEDE: wrist must cross the body midline by at least this much.
    recede_crossover: float = 0.05


@dataclass
class ClassifierState:
    last_fired: Gesture = Gesture.NONE
    history: list[Gesture] = field(default_factory=list)


class GestureClassifier:
    """Frame-by-frame body-pose gesture recogniser.

    Construct once, call `update(frame)` per frame. Returns the
    *smoothed* gesture: NONE unless the same non-NONE gesture has been
    observed for `smoothing_frames` consecutive frames.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.state = ClassifierState()

    # ---- public API ---- #

    def update(self, frame: PerceptionFrame) -> Gesture:
        """Process one frame, return the smoothed gesture.

        If the frame has no person detection with usable keypoints, the
        history is appended with NONE so an old smoothed-gesture decays
        out within `smoothing_frames` rather than latching.
        """
        operator = self._select_operator(frame)
        if operator is None:
            return self._push(Gesture.NONE)
        return self._push(self._classify(operator))

    # ---- internals ---- #

    def _select_operator(self, frame: PerceptionFrame) -> Optional[Detection]:
        """Pick the closest person detection with keypoints, or None."""
        candidates = [
            d for d in frame.detections
            if d.class_name == "person"
            and d.keypoints is not None
            and len(d.keypoints) >= 17
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda d: d.depth_m)

    def _kp(self, det: Detection, idx: int) -> Optional[tuple[float, float]]:
        """Return (x, y) for keypoint idx if its confidence passes, else None."""
        if det.keypoints is None or idx >= len(det.keypoints):
            return None
        x, y, conf = det.keypoints[idx]
        if conf < self.config.min_keypoint_confidence:
            return None
        return (x, y)

    def _classify(self, det: Detection) -> Gesture:
        cfg = self.config
        ls = self._kp(det, KP.LEFT_SHOULDER)
        rs = self._kp(det, KP.RIGHT_SHOULDER)
        lw = self._kp(det, KP.LEFT_WRIST)
        rw = self._kp(det, KP.RIGHT_WRIST)
        lh = self._kp(det, KP.LEFT_HIP)
        rh = self._kp(det, KP.RIGHT_HIP)
        nose = self._kp(det, KP.NOSE)

        # All gestures need both shoulders and both wrists.
        if ls is None or rs is None or lw is None or rw is None:
            return Gesture.NONE

        # In normalised frame coords:
        #   left_shoulder.x is the operator's *right* side (mirror in
        #   the camera frame), but for our gesture geometry we just need
        #   the LR-on-screen relationship — so we use the labels as-is.

        # STOP — T-pose. Wrists at shoulder height AND extended outward.
        left_arm_horizontal = abs(lw[1] - ls[1]) <= cfg.stop_vertical_tolerance
        right_arm_horizontal = abs(rw[1] - rs[1]) <= cfg.stop_vertical_tolerance
        left_arm_extended = (ls[0] - lw[0]) >= cfg.stop_horizontal_extension
        right_arm_extended = (rw[0] - rs[0]) >= cfg.stop_horizontal_extension
        if (
            left_arm_horizontal and right_arm_horizontal
            and left_arm_extended and right_arm_extended
        ):
            return Gesture.STOP

        # COME — both wrists clearly above the head.
        if nose is not None:
            if (
                (nose[1] - lw[1]) >= cfg.come_above_head_offset
                and (nose[1] - rw[1]) >= cfg.come_above_head_offset
            ):
                return Gesture.COME

        # LAND — both wrists below both hips.
        if lh is not None and rh is not None:
            if (
                lw[1] >= lh[1] + cfg.land_below_hip_offset
                and rw[1] >= rh[1] + cfg.land_below_hip_offset
            ):
                return Gesture.LAND

        # RECEDE — wrists crossed in front of the chest (X pose).
        # Body midline ≈ midpoint of shoulders.
        midline_x = (ls[0] + rs[0]) / 2.0
        # Left wrist must cross midline to the right; right wrist to the left.
        left_crossed = (lw[0] - midline_x) >= cfg.recede_crossover
        right_crossed = (midline_x - rw[0]) >= cfg.recede_crossover
        # And wrists should be at roughly chest level (between shoulders and hips).
        wrist_at_chest = (
            ls[1] - 0.05 <= lw[1] <= ls[1] + 0.30
            and rs[1] - 0.05 <= rw[1] <= rs[1] + 0.30
        )
        if left_crossed and right_crossed and wrist_at_chest:
            return Gesture.RECEDE

        return Gesture.NONE

    def _push(self, g: Gesture) -> Gesture:
        """Append to history, return smoothed gesture."""
        self.state.history.append(g)
        # Trim — only keep the last `smoothing_frames` worth.
        n = self.config.smoothing_frames
        if len(self.state.history) > n:
            self.state.history = self.state.history[-n:]
        if len(self.state.history) < n:
            return Gesture.NONE
        # All recent equal AND not NONE → fire.
        first = self.state.history[0]
        if first != Gesture.NONE and all(g == first for g in self.state.history):
            self.state.last_fired = first
            return first
        return Gesture.NONE
