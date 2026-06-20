import pytest, time, os
from src.engine.state_machine import StateMachine, EngineState
from src.engine.signals import VisionSignal
from src.engine.mapping_engine import MappingEngine

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

@pytest.fixture
def sm():
    me = MappingEngine()
    me.load(os.path.join(FIX, "mappings.default.json"), os.path.join(FIX, "mappings.user.json"))
    return StateMachine(me)


class TestStateMachine:
    def test_initial_state_idle(self, sm):
        assert sm.state == EngineState.IDLE

    def test_idle_to_detecting(self, sm):
        sm.update(vision_signal=VisionSignal(
            features={"is_donk": True},
            frame_id=0, timestamp=time.time()))
        assert sm.state == EngineState.DETECTING

    def test_detecting_to_idle_on_loss(self, sm):
        sm.update(vision_signal=VisionSignal(
            features={"is_donk": True},
            frame_id=0, timestamp=time.time()))
        assert sm.state == EngineState.DETECTING
        sm.update()
        assert sm.state == EngineState.IDLE

    def test_detecting_to_triggered_after_debounce(self, sm):
        result = None
        for i in range(8):
            result = sm.update(vision_signal=VisionSignal(
                features={"is_donk": True},
                frame_id=i, timestamp=time.time()))
        assert result is not None
        assert result.mapping_id == "donk"
        assert sm.state == EngineState.TRIGGERED

    def test_triggered_to_cooldown(self, sm):
        for i in range(8):
            sm.update(vision_signal=VisionSignal(
                features={"is_donk": True},
                frame_id=i, timestamp=time.time()))
        sm.update()
        assert sm.state == EngineState.COOLDOWN

    def test_cooldown_blocks_retrigger(self, sm):
        for i in range(8):
            sm.update(vision_signal=VisionSignal(
                features={"is_donk": True},
                frame_id=i, timestamp=time.time()))
        sm.update()  # -> COOLDOWN
        r = sm.update(vision_signal=VisionSignal(
            features={"is_donk": True},
            frame_id=100, timestamp=time.time()))
        assert r is None
        assert sm.state == EngineState.COOLDOWN

    def test_debounce_progress(self, sm):
        assert sm.debounce_progress == 0.0
        for i in range(4):
            sm.update(vision_signal=VisionSignal(
                features={"is_donk": True},
                frame_id=i, timestamp=time.time()))
        assert sm.debounce_progress == 4 / 8
