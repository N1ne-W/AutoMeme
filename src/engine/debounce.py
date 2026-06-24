"""???????? N ?????????????????"""
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class DebounceState(Enum):
    DETECTING = auto()
    CONFIRMED = auto()


class Debounce:
    """????????????????

    - ?? threshold ? update(True) -> CONFIRMED
    - update(False) -> ???? 1??????????????????
    - reset() -> ????
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
            self.counter = max(0, self.counter - 1)
            return DebounceState.DETECTING

    def set_threshold(self, threshold: int) -> None:
        self.threshold = max(1, threshold)

    def reset(self) -> None:
        self.counter = 0

