import pytest
from unittest.mock import MagicMock
from src.vision.features.hand_victory import VictoryDetector
from src.vision.features.base import FeatureContext


def _mk(x, y):
    lm = MagicMock()
    lm.x = x; lm.y = y
    return lm

def _hand(states):
    # states: [index, middle, ring, pinky] each True=straight
    lms = [_mk(0.5, 0.8)] * 21
    lms[0] = _mk(0.5, 0.8)  # wrist
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]
    for i, straight in enumerate(states):
        if straight:
            lms[tip_ids[i]] = _mk(0.5, 0.2)
            lms[mcp_ids[i]] = _mk(0.5, 0.6)
        else:
            lms[tip_ids[i]] = _mk(0.5, 0.7)
            lms[mcp_ids[i]] = _mk(0.5, 0.5)
    m = MagicMock()
    m.landmark = lms
    return m


class TestVictoryDetector:
    def test_victory(self):
        det = VictoryDetector()
        ctx = FeatureContext(None, None, _hand([True, True, False, False]), None)
        assert det.detect(ctx)

    def test_wrong_fingers(self):
        det = VictoryDetector()
        ctx = FeatureContext(None, None, _hand([True, False, False, False]), None)
        assert not det.detect(ctx)

    def test_all_straight(self):
        det = VictoryDetector()
        ctx = FeatureContext(None, None, _hand([True, True, True, True]), None)
        assert not det.detect(ctx)

    def test_no_hands(self):
        det = VictoryDetector()
        assert not det.detect(FeatureContext(None, None, None, None))

    def test_category_hand(self):
        assert VictoryDetector().category == "hand"

    def test_left_hand_works(self):
        det = VictoryDetector()
        ctx = FeatureContext(None, _hand([True, True, False, False]), None, None)
        assert det.detect(ctx)
