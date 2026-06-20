"""判定引擎状态机：IDLE → DETECTING → TRIGGERED → COOLDOWN。"""
import time
import logging
from enum import Enum, auto
from .debounce import Debounce, DebounceState
from .cooldown import Cooldown
from .signals import VisionSignal, AudioSignal, TriggerEvent
from .mapping_engine import MappingEngine

logger = logging.getLogger(__name__)


class EngineState(Enum):
    IDLE = auto()
    DETECTING = auto()
    TRIGGERED = auto()
    COOLDOWN = auto()


class StateMachine:
    """核心判定引擎。

    消费视觉/音频信号，管理状态转移，输出 TriggerEvent。
    """

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
    ) -> TriggerEvent | None:
        """每帧调用，返回本轮是否产生触发事件。"""
        now = time.time()

        # --- 信号合并：获取当前最佳候选映射 ---
        candidate: object | None = None

        if vision_signal is not None:
            candidate = self._mapping.match_vision(vision_signal.features)
        if candidate is None and audio_signal is not None:
            candidate = self._mapping.match_audio(audio_signal.keyword)

        # --- 状态机转移 ---
        if self._state == EngineState.IDLE:
            if candidate is not None:
                self._active_mapping_id = candidate.id
                self._debounce.reset()
                self._debounce.update(True)
                self._state = EngineState.DETECTING
                logger.debug("IDLE → DETECTING (%s)", candidate.id)

        elif self._state == EngineState.DETECTING:
            if candidate is not None and candidate.id == self._active_mapping_id:
                # 同一映射持续满足
                result = self._debounce.update(True)
                if result == DebounceState.CONFIRMED:
                    self._state = EngineState.TRIGGERED
                    self._cooldown.trigger()
                    logger.info("DETECTING → TRIGGERED (%s)", candidate.id)

                    return TriggerEvent(
                        mapping_id=candidate.id,
                        action_type=candidate.action_mode,
                        image_path=candidate.image_path,
                        audio_path=candidate.audio_path,
                        priority=candidate.priority,
                        timestamp=now,
                    )
            elif candidate is not None:
                # 不同映射，重置去抖动跟踪新映射
                self._active_mapping_id = candidate.id
                self._debounce.reset()
                self._debounce.update(True)
            else:
                # 条件消失
                self._debounce.update(False)
                if self._debounce.counter == 0:
                    self._state = EngineState.IDLE
                    self._active_mapping_id = None
                    logger.debug("DETECTING → IDLE (lost)")

        elif self._state == EngineState.TRIGGERED:
            # 立即进入冷却
            self._state = EngineState.COOLDOWN
            logger.debug("TRIGGERED → COOLDOWN")

        elif self._state == EngineState.COOLDOWN:
            if not self._cooldown.is_active:
                # 冷却结束，检查是否有持续信号
                self._state = EngineState.IDLE
                self._active_mapping_id = None
                self._debounce.reset()
                logger.debug("COOLDOWN → IDLE")

                # 回查：冷却期间是否有持续信号可直接进入 DETECTING
                if candidate is not None:
                    self._active_mapping_id = candidate.id
                    self._debounce.update(True)
                    self._state = EngineState.DETECTING

        return None
