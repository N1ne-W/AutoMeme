"""State machine: IDLE -> DETECTING -> TRIGGERED -> COOLDOWN."""
import time
import logging
from enum import Enum, auto
from .debounce import Debounce, DebounceState
from .cooldown import Cooldown
from .signals import VisionSignal, AudioSignal, TriggerEvent
from .mapping_engine import MappingEngine

logger = logging.getLogger(__name__)

try:
    from src.diagnostics import log_trace
except ImportError:
    def log_trace(*args, **kwargs): pass


class EngineState(Enum):
    IDLE = auto()
    DETECTING = auto()
    TRIGGERED = auto()
    COOLDOWN = auto()


class StateMachine:
    def __init__(self, mapping_engine: MappingEngine):
        self._mapping = mapping_engine
        self._state = EngineState.IDLE
        self._active_mapping_id: str | None = None
        self._cooldown = Cooldown(3000)
        self._debounce = Debounce(8)

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def debounce_progress(self) -> float:
        return self._debounce.progress

    def update(
        self,
        vision_signal: VisionSignal | None = None,
        audio_signal: AudioSignal | None = None,
        trace_id: str = "",
    ) -> TriggerEvent | None:
        now = time.time()
        candidate: object | None = None

        if vision_signal is not None:
            candidate = self._mapping.match_vision(vision_signal.features)
        if candidate is None and audio_signal is not None:
            candidate = self._mapping.match_audio(audio_signal.keyword)

        # ---- IDLE ----
        if self._state == EngineState.IDLE:
            if candidate is not None:
                self._active_mapping_id = candidate.id
                self._debounce.reset()
                self._debounce.set_threshold(candidate.debounce_frames)
                result = self._debounce.update(True)
                if result == DebounceState.CONFIRMED:
                    # Immediate trigger (e.g. voice with debounce_frames=1)
                    self._state = EngineState.TRIGGERED
                    self._cooldown.start(candidate.cooldown_ms)
                    if trace_id:
                        log_trace(trace_id, "state.machine", "trigger_decision",
                                  decision="accepted", mapping_id=candidate.id,
                                  reason="immediate")
                    return TriggerEvent(
                        mapping_id=candidate.id,
                        action_type=candidate.action_mode,
                        image_path=candidate.image_path,
                        audio_path=candidate.audio_path,
                        priority=candidate.priority,
                        timestamp=now,
                        display_mode=getattr(candidate, 'display_mode', 'hold'),
                        duration_ms=getattr(candidate, 'duration_ms', 2000),
                    )
                self._state = EngineState.DETECTING
                if trace_id:
                    log_trace(trace_id, "state.machine", "state_changed",
                              from_state="IDLE", to_state="DETECTING", feature=candidate.id)

        # ---- DETECTING ----
        elif self._state == EngineState.DETECTING:
            if candidate is not None and candidate.id == self._active_mapping_id:
                result = self._debounce.update(True)
                if result == DebounceState.CONFIRMED:
                    self._state = EngineState.TRIGGERED
                    self._cooldown.start(candidate.cooldown_ms)
                    if trace_id:
                        log_trace(trace_id, "state.machine", "trigger_decision",
                                  decision="accepted", mapping_id=candidate.id,
                                  reason="debounce_satisfied")
                    return TriggerEvent(
                        mapping_id=candidate.id,
                        action_type=candidate.action_mode,
                        image_path=candidate.image_path,
                        audio_path=candidate.audio_path,
                        priority=candidate.priority,
                        timestamp=now,
                        display_mode=getattr(candidate, 'display_mode', 'hold'),
                        duration_ms=getattr(candidate, 'duration_ms', 2000),
                    )
            elif candidate is not None:
                self._active_mapping_id = candidate.id
                self._debounce.reset()
                self._debounce.update(True)
            else:
                self._debounce.update(False)
                if self._debounce.counter == 0:
                    self._state = EngineState.IDLE
                    self._active_mapping_id = None

        # ---- TRIGGERED ----
        elif self._state == EngineState.TRIGGERED:
            self._state = EngineState.COOLDOWN
            if trace_id:
                log_trace(trace_id, "state.machine", "cooldown_started",
                          cooldown_ms=str(self._cooldown._cooldown_s * 1000))

        # ---- COOLDOWN ----
        elif self._state == EngineState.COOLDOWN:
            if candidate is not None and trace_id:
                log_trace(trace_id, "state.machine", "trigger_rejected",
                          reason="cooldown_active",
                          cooldown_remaining_ms=f"{self._cooldown.remaining_ms:.0f}")
            if not self._cooldown.is_active:
                self._state = EngineState.IDLE
                self._active_mapping_id = None
                self._debounce.reset()
                if candidate is not None:
                    self._active_mapping_id = candidate.id
                    self._debounce.update(True)
                    self._state = EngineState.DETECTING
                elif audio_signal is not None:
                    candidate2 = self._mapping.match_audio(audio_signal.keyword)
                    if candidate2 is not None:
                        self._active_mapping_id = candidate2.id
                        self._debounce.update(True)
                        self._state = EngineState.DETECTING

        return None
