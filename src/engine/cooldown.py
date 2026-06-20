"""冷却计时器：触发后一段时间内禁止重复触发。"""
import time
import logging

logger = logging.getLogger(__name__)


class Cooldown:
    """简单冷却器。"""

    def __init__(self, cooldown_ms: int = 3000):
        self._cooldown_s = cooldown_ms / 1000.0
        self._last_trigger_time: float = 0.0

    @property
    def is_active(self) -> bool:
        """当前是否在冷却中。"""
        return time.time() - self._last_trigger_time < self._cooldown_s

    @property
    def remaining_ms(self) -> float:
        elapsed = time.time() - self._last_trigger_time
        return max(0.0, (self._cooldown_s - elapsed) * 1000)

    def trigger(self) -> None:
        """开始冷却计时。"""
        self._last_trigger_time = time.time()

    def reset(self) -> None:
        """强制结束冷却。"""
        self._last_trigger_time = 0.0
