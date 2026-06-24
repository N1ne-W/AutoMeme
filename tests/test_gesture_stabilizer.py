"""Tests for GestureStabilizer: time-based enter/exit debounce."""
import time
import pytest
from src.engine.gesture_stabilizer import GestureStabilizer, GestureEvent, FeatureState


class TestGestureStabilizer:
    """Core stabilizer state machine tests."""

    def test_initial_state_inactive(self):
        gs = GestureStabilizer()
        assert gs.get_state("any_feature") == FeatureState.INACTIVE

    def test_entering_state(self):
        gs = GestureStabilizer(enter_debounce_ms=200)
        now = 1000.0
        events = gs.update({"f1": True}, now)
        assert len(events) == 0  # No event yet, still entering
        assert gs.get_state("f1") == FeatureState.ENTERING

    def test_enter_debounce_confirms(self):
        gs = GestureStabilizer(enter_debounce_ms=150)
        now = 1000.0
        # First frame: start entering
        events = gs.update({"f1": True}, now)
        assert len(events) == 0
        # Advance past enter_debounce_ms
        now += 0.160  # 160ms > 150ms
        events = gs.update({"f1": True}, now)
        assert len(events) == 1
        assert events[0].event_type == "started"
        assert events[0].feature_id == "f1"
        assert gs.get_state("f1") == FeatureState.ACTIVE

    def test_enter_lost_before_confirm(self):
        gs = GestureStabilizer(enter_debounce_ms=200)
        now = 1000.0
        gs.update({"f1": True}, now)
        now += 0.05  # 50ms, not enough
        events = gs.update({"f1": False}, now)  # Lost signal
        assert len(events) == 0
        assert gs.get_state("f1") == FeatureState.INACTIVE  # Back to inactive

    def test_active_then_lost_grace(self):
        """Feature active, then raw False -> enters LOST_GRACE, no ended event yet."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=500)
        now = 1000.0
        # Enter active
        gs.update({"f1": True}, now)
        now += 0.110
        events = gs.update({"f1": True}, now)
        assert len(events) == 1 and events[0].event_type == "started"
        assert gs.get_state("f1") == FeatureState.ACTIVE
        # Now lose signal
        now += 0.05
        events = gs.update({"f1": False}, now)
        assert len(events) == 0
        assert gs.get_state("f1") == FeatureState.LOST_GRACE
        assert gs.is_active_or_grace("f1")  # Still should display

    def test_lost_grace_recovers_without_event(self):
        """Within exit_debounce_ms, recover to ACTIVE without re-trigger."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=500)
        now = 1000.0
        gs.update({"f1": True}, now)
        now += 0.110
        gs.update({"f1": True}, now)  # Now ACTIVE
        now += 0.05
        gs.update({"f1": False}, now)  # LOST_GRACE
        assert gs.get_state("f1") == FeatureState.LOST_GRACE
        # Recover within grace (200ms < 500ms)
        now += 0.200
        events = gs.update({"f1": True}, now)
        assert len(events) == 0  # No event on recovery
        assert gs.get_state("f1") == FeatureState.ACTIVE

    def test_exit_debounce_triggers_ended(self):
        """Lost grace exceeds exit_debounce_ms -> ended event."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=400)
        now = 1000.0
        gs.update({"f1": True}, now)
        now += 0.110
        gs.update({"f1": True}, now)  # ACTIVE
        now += 0.05
        gs.update({"f1": False}, now)  # LOST_GRACE
        # Advance past exit_debounce_ms
        now += 0.500  # 500ms > 400ms
        events = gs.update({"f1": False}, now)
        assert len(events) == 1
        assert events[0].event_type == "ended"
        assert events[0].feature_id == "f1"
        assert gs.get_state("f1") == FeatureState.INACTIVE
        assert not gs.is_active_or_grace("f1")

    def test_min_hold_blocks_exit(self):
        """min_hold_ms prevents exit before minimum active time."""
        gs = GestureStabilizer(enter_debounce_ms=50, exit_debounce_ms=100, min_hold_ms=300)
        now = 1000.0
        gs.update({"f1": True}, now)
        now += 0.060
        gs.update({"f1": True}, now)  # ACTIVE at ~1060
        assert gs.get_state("f1") == FeatureState.ACTIVE
        # Immediately lose signal
        now += 0.01
        gs.update({"f1": False}, now)  # LOST_GRACE
        now += 0.200  # 200ms lost, but min_hold is 300ms
        events = gs.update({"f1": False}, now)
        assert len(events) == 0  # Blocked by min_hold
        assert gs.get_state("f1") == FeatureState.LOST_GRACE
        # Wait past min_hold from active start
        now += 0.200  # Total active+grace > 300ms now
        events = gs.update({"f1": False}, now)
        assert len(events) == 1
        assert events[0].event_type == "ended"

    def test_multiple_features_independent(self):
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=300)
        now = 1000.0
        gs.update({"f1": True, "f2": True}, now)
        now += 0.110
        events = gs.update({"f1": True, "f2": True}, now)
        assert len(events) == 2  # Both started
        started_ids = {e.feature_id for e in events}
        assert started_ids == {"f1", "f2"}
        # f1 stays active, f2 lost
        now += 0.05
        events = gs.update({"f1": True, "f2": False}, now)
        assert len(events) == 0
        assert gs.get_state("f1") == FeatureState.ACTIVE
        assert gs.get_state("f2") == FeatureState.LOST_GRACE

    def test_stale_features_cleaned_up(self):
        gs = GestureStabilizer()
        now = 1000.0
        gs.update({"f1": True}, now)
        assert gs.get_state("f1") != FeatureState.INACTIVE
        # f1 no longer reported
        gs.update({}, now + 1.0)
        assert gs.get_state("f1") == FeatureState.INACTIVE

    def test_reset_clears_all(self):
        gs = GestureStabilizer(enter_debounce_ms=50)
        now = 1000.0
        gs.update({"f1": True}, now)
        now += 0.060
        gs.update({"f1": True}, now)
        assert gs.get_state("f1") == FeatureState.ACTIVE
        gs.reset()
        assert gs.get_state("f1") == FeatureState.INACTIVE

    def test_old_config_no_gesture_section(self):
        """Without gesture config, default values are used (no crash)."""
        gs = GestureStabilizer()  # All defaults
        assert gs.enter_debounce_ms == 150
        assert gs.exit_debounce_ms == 600
        now = 1000.0
        gs.update({"f1": True}, now)
        now += 0.200  # > 150ms enter
        events = gs.update({"f1": True}, now)
        assert len(events) == 1 and events[0].event_type == "started"


class TestGestureStabilizerDisplaySim:
    """Simulated display integration tests."""

    def test_visual_active_shows_image(self):
        """After enter_debounce, stabilizer says active -> display should show."""
        gs = GestureStabilizer(enter_debounce_ms=150)
        now = 1000.0
        gs.update({"is_donk": True}, now)
        now += 0.200
        events = gs.update({"is_donk": True}, now)
        assert len(events) == 1 and events[0].event_type == "started"
        # Simulate: display should be showing
        display_owner = events[0].feature_id
        assert display_owner == "is_donk"

    def test_brief_loss_does_not_clear(self):
        """Brief loss within exit_debounce_ms: image stays (LOST_GRACE, no ended)."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=600)
        now = 1000.0
        gs.update({"is_donk": True}, now)
        now += 0.110
        gs.update({"is_donk": True}, now)  # ACTIVE
        # Brief loss: 150ms < 600ms
        now += 0.150
        events = gs.update({"is_donk": False}, now)
        assert len(events) == 0  # No ended event
        assert gs.is_active_or_grace("is_donk")  # Display should stay

    def test_loss_exceeds_exit_clears(self):
        """Loss > exit_debounce_ms triggers ended -> display clear."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=400)
        now = 1000.0
        gs.update({"is_donk": True}, now)
        now += 0.110
        gs.update({"is_donk": True}, now)
        now += 0.05
        gs.update({"is_donk": False}, now)
        now += 0.500  # > 400ms exit
        events = gs.update({"is_donk": False}, now)
        assert len(events) == 1 and events[0].event_type == "ended"
        assert not gs.is_active_or_grace("is_donk")

    def test_owner_mismatch_clear_ignored(self):
        """When feature A ends but display owner is B, clear should be ignored."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=300)
        now = 1000.0
        # Feature A active, then lost and ended
        gs.update({"feat_a": True, "feat_b": True}, now)
        now += 0.110
        gs.update({"feat_a": True, "feat_b": True}, now)  # Both ACTIVE
        # Display is owned by feat_b
        now += 0.05
        gs.update({"feat_a": False, "feat_b": True}, now)  # A lost
        now += 0.400  # A ended
        events = gs.update({"feat_a": False, "feat_b": True}, now)
        ended_features = [e.feature_id for e in events if e.event_type == "ended"]
        assert "feat_a" in ended_features
        # feat_b still active - clear for feat_a should be ignored by owner check
        assert gs.get_state("feat_b") == FeatureState.ACTIVE

    def test_voice_duration_unchanged(self):
        """Voice keyword features (no visual feature) are not affected by stabilizer."""
        gs = GestureStabilizer()
        now = 1000.0
        # Voice mappings have no features, so stabilizer receives empty dict
        events = gs.update({}, now)
        assert len(events) == 0
        # Voice one-shot still uses duration_ms from mapping, unchanged
        # This test just ensures stabilizer doesnt interfere with empty feature sets

    def test_cooldown_does_not_affect_stabilizer(self):
        """Stabilizer tracks features independently of state machine cooldown."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=400)
        now = 1000.0
        gs.update({"is_donk": True}, now)
        now += 0.110
        gs.update({"is_donk": True}, now)  # ACTIVE
        # Even during "cooldown", the stabilizer still tracks the feature
        # If feature stays True, it remains ACTIVE regardless of cooldown
        now += 1.0  # Simulate cooldown time passing
        events = gs.update({"is_donk": True}, now)
        assert len(events) == 0  # No change, still ACTIVE
        assert gs.is_active_or_grace("is_donk")

    def test_no_reload_on_recovery(self):
        """When recovering from LOST_GRACE, no started event emitted (no reload)."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=500)
        now = 1000.0
        gs.update({"is_donk": True}, now)
        now += 0.110
        started_events = gs.update({"is_donk": True}, now)
        assert len(started_events) == 1  # Initial started
        # Lose and recover
        now += 0.05
        gs.update({"is_donk": False}, now)
        now += 0.200  # Within grace
        events = gs.update({"is_donk": True}, now)
        # No new started event on recovery
        assert all(e.event_type != "started" for e in events)
        assert gs.get_state("is_donk") == FeatureState.ACTIVE

    def test_re_trigger_after_full_ended(self):
        """After full ended, a new detection should trigger a fresh started."""
        gs = GestureStabilizer(enter_debounce_ms=100, exit_debounce_ms=300)
        now = 1000.0
        # First activation
        gs.update({"is_donk": True}, now)
        now += 0.110
        gs.update({"is_donk": True}, now)  # started
        # Full end
        now += 0.05
        gs.update({"is_donk": False}, now)
        now += 0.400
        gs.update({"is_donk": False}, now)  # ended
        assert gs.get_state("is_donk") == FeatureState.INACTIVE
        # New detection
        now += 0.5
        gs.update({"is_donk": True}, now)
        now += 0.110
        events = gs.update({"is_donk": True}, now)
        assert len(events) == 1 and events[0].event_type == "started"

