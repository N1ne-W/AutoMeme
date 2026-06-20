"""去抖动：要求连续 N 帧满足条件才确认。"""
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class DebounceState(Enum):
    DETECTING = auto()
    CONFIRMED = auto()


class Debounce:
    """帧级别去抖动器。

    - 连续 threshold 帧 update(True) → CONFIRMED
    - 任意帧 update(False) → 计数器重置为 0 → DETECTING
    """

    def __init__(self, threshold: int = 8):
        self.threshold = max(1, threshold)
        self.counter = 0

    @property
    def progress(self) -> float:
        return min(1.0, self.counter / self.threshold)

    def update(self, condition: bool) -> DebounceState:
        if condition:
            self.counter += 1
            if self.counter >= self.threshold:
                return DebounceState.CONFIRMED
            return DebounceState.DETECTING
        else:
            self.counter = 0
            return DebounceState.DETECTING

    def reset(self) -> None:
        self.counter = 0
