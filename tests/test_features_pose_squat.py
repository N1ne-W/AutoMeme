import pytest
from unittest.mock import MagicMock
from src.vision.features.pose_squat import SquatDetector
from src.vision.features.base import FeatureContext


def _mk(x, y):
    lm = MagicMock()
    lm.x = x; lm.y = y
    return lm

def _pose(hip, knee, ankle):
    lms = [_mk(0, 0)] * 33
    lms[23] = _mk(*hip)
    lms[25] = _mk(*knee)
    lms[27] = _mk(*ankle)
    m = MagicMock()
    m.landmark = lms
    return m


class TestSquatDetector:
    def test_standing_not_squat(self):
        det = SquatDetector()
        ctx = FeatureContext(
            pose_landmarks=_pose((0.5, 0.4), (0.5, 0.55), (0.5, 0.7)),
            left_hand_landmarks=None, right_hand_landmarks=None, face_landmarks=None)
        assert not det.detect(ctx)

    def test_squat_detected(self):
        det = SquatDetector()
        ctx = FeatureContext(
            pose_landmarks=_pose((0.5, 0.4), (0.3, 0.55), (0.5, 0.7)),
            left_hand_landmarks=None, right_hand_landmarks=None, face_landmarks=None)
        assert det.detect(ctx)

    def test_no_pose_returns_false(self):
        det = SquatDetector()
        ctx = FeatureContext(None, None, None, None)
        assert not det.detect(ctx)

    def test_category_pose(self):
        assert SquatDetector().category == "pose"

    def test_custom_threshold(self):
        det = SquatDetector(params={"angle_threshold": 100})
        ctx = FeatureContext(
            pose_landmarks=_pose((0.5, 0.3), (0.7, 0.55), (0.5, 0.7)),
            left_hand_landmarks=None, right_hand_landmarks=None, face_landmarks=None)
        assert det.detect(ctx)
