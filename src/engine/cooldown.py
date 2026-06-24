"""Cooldown timer: blocks re-trigger for a configurable duration."""
import time
import logging

logger = logging.getLogger(__name__)


class Cooldown:
    """Cooldown with dynamic duration per trigger."""

    def __init__(self, default_cooldown_ms: int = 3000):
        self._default_s = default_cooldown_ms / 1000.0
        self._duration_s = self._default_s
        self._triggered_at: float = 0.0

    @property
    def is_active(self) -> bool:
        return time.time() - self._triggered_at < self._duration_s

    @property
    def remaining_ms(self) -> float:
        elapsed = time.time() - self._triggered_at
        return max(0.0, (self._duration_s - elapsed) * 1000)

    def start(self, duration_ms: int) -> None:
        """Start cooldown with a specific duration. 0 or negative uses default."""
        self._duration_s = (duration_ms / 1000.0) if duration_ms > 0 else self._default_s
        self._triggered_at = time.time()

    def reset(self) -> None:
        """Force-end cooldown."""
        self._triggered_at = 0.0
