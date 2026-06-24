import pytest
from src.engine.debounce import Debounce, DebounceState


class TestDebounce:
    def test_initial_counter_zero(self):
        d = Debounce(threshold=8)
        assert d.counter == 0
        assert d.progress == 0.0

    def test_detecting_before_threshold(self):
        d = Debounce(threshold=5)
        for i in range(4):
            result = d.update(True)
            assert result == DebounceState.DETECTING
        assert d.counter == 4

    def test_confirmed_at_threshold(self):
        d = Debounce(threshold=3)
        d.update(True); d.update(True)
        result = d.update(True)
        assert result == DebounceState.CONFIRMED

    def test_soft_decay_on_mismatch(self):
        d = Debounce(threshold=5)
        for _ in range(3):
            d.update(True)
        assert d.counter == 3
        d.update(False)
        assert d.counter == 2  # soft decay: decrement by 1, not reset to 0

    def test_sustained_false_eventually_zero(self):
        d = Debounce(threshold=5)
        d.counter = 5
        for _ in range(5):
            d.update(False)
        assert d.counter == 0  # sustained False drains counter to zero

    def test_threshold_one(self):
        d = Debounce(threshold=1)
        assert d.update(True) == DebounceState.CONFIRMED

    def test_threshold_zero_becomes_one(self):
        d = Debounce(threshold=0)
        assert d.update(True) == DebounceState.CONFIRMED

    def test_progress_capped_at_one(self):
        d = Debounce(threshold=3)
        for _ in range(10):
            d.update(True)
        assert d.progress <= 1.0

    def test_oscillation_tolerance(self):
        """2 True, 1 False alternating should eventually confirm (not oscillate forever)."""
        d = Debounce(threshold=8)
        # Simulate a shaky hand: True(2) False(1) pattern
        for _ in range(20):
            d.update(True)
            d.update(True)
            d.update(False)  # brief loss
        # After 20 cycles: 40 True - 20 False = +20 net -> counter should be >= 8
        assert d.counter >= 8

    def test_manual_reset(self):
        d = Debounce(threshold=8)
        for _ in range(6):
            d.update(True)
        d.reset()
        assert d.counter == 0
