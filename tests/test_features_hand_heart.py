import pytest
from unittest.mock import MagicMock
from src.vision.features.hand_heart import HeartDetector
from src.vision.features.base import FeatureContext


def _mk(x, y):
    lm = MagicMock()
    lm.x = x; lm.y = y
    return lm

def _hand(thumb, index):
    lms = [_mk(0.5, 0.8)] * 21
    lms[4] = _mk(*thumb)
    lms[8] = _mk(*index)
    m = MagicMock()
    m.landmark = lms
    return m


class TestHeartDetector:
    def test_heart_detected(self):
        det = HeartDetector()
        ctx = FeatureContext(
            None,
            _hand((0.47, 0.5), (0.47, 0.45)),
            _hand((0.53, 0.5), (0.53, 0.45)),
            None)
        assert det.detect(ctx)

    def test_too_far_no_heart(self):
        det = HeartDetector()
        ctx = FeatureContext(
            None,
            _hand((0.1, 0.5), (0.1, 0.45)),
            _hand((0.9, 0.5), (0.9, 0.45)),
            None)
        assert not det.detect(ctx)

    def test_one_hand_no_heart(self):
        det = HeartDetector()
        ctx = FeatureContext(
            None,
            _hand((0.47, 0.5), (0.47, 0.45)),
            None, None)
        assert not det.detect(ctx)

    def test_no_hands(self):
        det = HeartDetector()
        assert not det.detect(FeatureContext(None, None, None, None))

    def test_category_hand(self):
        assert HeartDetector().category == "hand"

    def test_custom_threshold(self):
        det = HeartDetector(params={"distance_threshold": 0.5})
        ctx = FeatureContext(
            None,
            _hand((0.3, 0.5), (0.3, 0.45)),
            _hand((0.7, 0.5), (0.7, 0.45)),
            None)
        assert det.detect(ctx)
