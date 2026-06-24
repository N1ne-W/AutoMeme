"""Tests for AudioSlot: state machine, idempotent play, fade-in/out, owner safety."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.engine.audio_slot import AudioSlot, AudioState


class FakeChannel:
    """Mock pygame Channel for testing without real audio hardware."""
    def __init__(self):
        self._volume = 0.0
        self._stopped = False
        self._busy = True

    def set_volume(self, v):
        self._volume = v

    def get_volume(self):
        return self._volume

    def stop(self):
        self._stopped = True
        self._busy = False

    def get_busy(self):
        return self._busy


class FakeSound:
    """Mock pygame Sound."""
    def play(self, loops=0, fade_ms=0):
        return FakeChannel()


@pytest.fixture
def slot():
    """Create AudioSlot with mocked pygame."""
    with patch("src.engine.audio_slot.AudioSlot._load_and_play", return_value=None):
        s = AudioSlot("fake/project/root")
        yield s


class TestAudioSlotStates:
    """Core state machine transitions."""

    def test_initial_state_idle(self, slot):
        assert slot.state == AudioState.IDLE
        assert slot.owner_id is None
        assert slot.volume == 0.0

    def test_play_transitions_to_fading_in(self, slot):
        slot.play("owner1", "sound.mp3")
        assert slot.state == AudioState.FADING_IN
        assert slot.owner_id == "owner1"

    def test_tick_fades_in_to_playing(self, slot):
        slot.play("owner1", "sound.mp3", fade_in_ms=100, max_volume=0.8)
        fc = FakeChannel()
        slot._channel = fc
        slot._volume = 0.0
        slot._state = AudioState.FADING_IN
        slot._max_volume = 0.8
        slot._fade_in_ms = 100
        slot.tick(150.0)
        assert slot.state == AudioState.PLAYING
        assert slot.volume == 0.8
        assert fc._volume == 0.8

    def test_stop_transitions_to_fading_out(self, slot):
        slot.play("owner1", "sound.mp3")
        slot._state = AudioState.PLAYING
        slot._channel = FakeChannel()
        slot.stop("owner1")
        assert slot.state == AudioState.FADING_OUT

    def test_tick_fades_out_to_idle(self, slot):
        slot._owner_id = "owner1"
        slot._state = AudioState.FADING_OUT
        slot._volume = 0.8
        slot._max_volume = 0.8
        slot._fade_out_ms = 100
        fc = FakeChannel()
        slot._channel = fc
        slot.tick(150.0)
        assert slot.state == AudioState.IDLE
        assert slot.volume == 0.0
        assert fc._stopped

    def test_follow_image_false_ignores_stop(self, slot):
        slot.play("owner1", "sound.mp3", follow_image=False)
        slot._state = AudioState.PLAYING
        slot._channel = FakeChannel()
        slot.stop("owner1")
        assert slot.state == AudioState.PLAYING


class TestAudioSlotIdempotent:
    """Idempotent play: same owner re-trigger ignored."""

    def test_same_owner_fading_in_ignored(self, slot):
        slot.play("owner1", "sound.mp3")
        assert slot.state == AudioState.FADING_IN
        slot.play("owner1", "sound.mp3")
        assert slot.state == AudioState.FADING_IN

    def test_same_owner_playing_ignored(self, slot):
        slot._owner_id = "owner1"
        slot._state = AudioState.PLAYING
        slot._channel = FakeChannel()
        slot.play("owner1", "sound.mp3")
        assert slot.state == AudioState.PLAYING

    def test_same_owner_fading_out_resumes(self, slot):
        slot._owner_id = "owner1"
        slot._state = AudioState.FADING_OUT
        slot._channel = FakeChannel()
        slot._volume = 0.3
        slot.play("owner1", "sound.mp3")
        assert slot.state == AudioState.FADING_IN

    def test_different_owner_replaces(self, slot):
        slot._owner_id = "owner1"
        slot._state = AudioState.PLAYING
        slot._channel = FakeChannel()
        slot.play("owner2", "other.mp3")
        assert slot.owner_id == "owner2"
        assert slot.state == AudioState.FADING_IN


class TestAudioSlotOwnerSafety:
    """Owner-based stop safety."""

    def test_stop_owner_mismatch_ignored(self, slot):
        slot.play("owner1", "sound.mp3")
        slot._state = AudioState.PLAYING
        slot._channel = FakeChannel()
        slot.stop("owner2")
        assert slot.state == AudioState.PLAYING

    def test_stop_idle_noop(self, slot):
        slot.stop("any_owner")
        assert slot.state == AudioState.IDLE


class TestAudioSlotConfigDefaults:
    """Old config with minimal fields still works."""

    def test_default_follow_image_true(self, slot):
        slot.play("owner1", "sound.mp3")
        slot._state = AudioState.PLAYING
        slot._channel = FakeChannel()
        slot.stop("owner1")
        assert slot.state == AudioState.FADING_OUT

    def test_default_loop_true(self, slot):
        slot.play("owner1", "sound.mp3")
        assert slot._loop == True

    def test_explicit_loop_false(self, slot):
        slot.play("owner1", "sound.mp3", loop=False)
        assert slot._loop == False

    def test_shutdown_cleans_up(self, slot):
        slot._owner_id = "owner1"
        slot._state = AudioState.PLAYING
        slot._channel = FakeChannel()
        slot.shutdown()
        assert slot.state == AudioState.IDLE
        assert slot.owner_id is None

