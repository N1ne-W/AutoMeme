"""Gesture stabilizer: time-based enter/exit debounce for visual features.

Separates raw_detected (per-frame) from stabilized_active (time-smoothed).
Features must be raw True for enter_debounce_ms before becoming "active".
Once active, features must be raw False for exit_debounce_ms before "ended".
Brief losses within exit_debounce_ms enter "lost_grace" and recover without re-trigger.
"""
import time
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

try:
    from src.diagnostics import log_trace
except ImportError:
    def log_trace(*args, **kwargs): pass


class FeatureState(Enum):
    INACTIVE = auto()       # raw False, not tracking
    ENTERING = auto()       # raw True, debouncing in
    ACTIVE = auto()         # confirmed active
    LOST_GRACE = auto()     # raw False, in exit grace window


class GestureEvent:
    """Emitted when a feature transitions ACTIVE or finishes LOST_GRACE->INACTIVE."""
    __slots__ = ("feature_id", "event_type", "timestamp")

    def __init__(self, feature_id: str, event_type: str, timestamp: float):
        self.feature_id = feature_id
        self.event_type = event_type   # "started" | "ended"
        self.timestamp = timestamp


class GestureStabilizer:
    """Time-based gesture stabilizer.

    Each feature transitions through states:
      INACTIVE --raw True--> ENTERING --enter_ms elapsed--> ACTIVE
      ACTIVE   --raw False--> LOST_GRACE --exit_ms elapsed--> INACTIVE (emit ended)
      LOST_GRACE --raw True--> ACTIVE (recovery, no event)

    enter_debounce_ms: raw must stay True this long before becoming ACTIVE.
    exit_debounce_ms:  raw must stay False this long before becoming INACTIVE/ended.
    min_hold_ms:       minimum time in ACTIVE before allowing exit (optional).
    """

    def __init__(self, enter_debounce_ms: int = 150, exit_debounce_ms: int = 600,
                 min_hold_ms: int = 0):
        self._enter_ms = enter_debounce_ms / 1000.0
        self._exit_ms = exit_debounce_ms / 1000.0
        self._min_hold_ms = min_hold_ms / 1000.0
        # Per-feature tracking: {feature_id: {state, state_since, active_since}}
        self._tracking: dict[str, dict] = {}
        # Per-feature overrides: {feature_id: {enter_ms, exit_ms, min_hold_ms}}
        self._overrides: dict[str, dict] = {}

    @property
    def enter_debounce_ms(self) -> int:
        return int(self._enter_ms * 1000)

    @property
    def exit_debounce_ms(self) -> int:
        return int(self._exit_ms * 1000)

    def update(self, raw_features: dict[str, bool], now: float) -> list[GestureEvent]:
        """Process raw per-frame features and return any transition events.

        Args:
            raw_features: {feature_id: bool} from FeatureExtractor
            now: current time.time()

        Returns:
            List of GestureEvent (started/ended) for features that transitioned.
        """
        events: list[GestureEvent] = []

        # Clean up stale entries for features no longer reported.
        # If a feature was ACTIVE or LOST_GRACE, emit "ended" before removing.
        reported = set(raw_features.keys())
        for fid in list(self._tracking.keys()):
            if fid not in reported:
                t = self._tracking[fid]
                if t["state"] in (FeatureState.ACTIVE, FeatureState.LOST_GRACE):
                    logger.info("[stabilizer] %s: removed from tracking -> ended", fid)
                    log_trace("gesture", "gesture.stabilizer", "feature_ended",
                              feature_id=fid, reason="not_reported")
                    events.append(GestureEvent(fid, "ended", now))
                del self._tracking[fid]

        for fid, raw in raw_features.items():
            event = self._update_feature(fid, raw, now)
            if event is not None:
                events.append(event)

        return events

    def _update_feature(self, fid: str, raw: bool, now: float) -> GestureEvent | None:
        """Update a single feature and return event if state transitioned."""
        if fid not in self._tracking:
            self._tracking[fid] = {
                "state": FeatureState.INACTIVE,
                "state_since": now,
                "active_since": 0.0,
            }

        t = self._tracking[fid]
        state: FeatureState = t["state"]
        elapsed = now - t["state_since"]

        # --- INACTIVE ---
        if state == FeatureState.INACTIVE:
            if raw:
                t["state"] = FeatureState.ENTERING
                t["state_since"] = now
                logger.debug("[stabilizer] %s: INACTIVE -> ENTERING", fid)
            return None

        # --- ENTERING ---
        if state == FeatureState.ENTERING:
            if not raw:
                t["state"] = FeatureState.INACTIVE
                t["state_since"] = now
                logger.debug("[stabilizer] %s: ENTERING -> INACTIVE (lost)", fid)
                return None
            enter_ms = self._overrides.get(fid, {}).get("enter_s", self._enter_ms)
            if elapsed >= enter_ms:
                t["state"] = FeatureState.ACTIVE
                t["state_since"] = now
                t["active_since"] = now
                logger.info("[stabilizer] %s: ENTERING -> ACTIVE (enter_debounce=%.0fms)",
                            fid, self._enter_ms * 1000)
                log_trace("gesture", "gesture.stabilizer", "feature_started",
                          feature_id=fid, enter_ms=str(int(self._enter_ms * 1000)))
                return GestureEvent(fid, "started", now)
            return None

        # --- ACTIVE ---
        if state == FeatureState.ACTIVE:
            if not raw:
                t["state"] = FeatureState.LOST_GRACE
                t["state_since"] = now
                logger.debug("[stabilizer] %s: ACTIVE -> LOST_GRACE", fid)
                log_trace("gesture", "gesture.stabilizer", "feature_lost_grace",
                          feature_id=fid, exit_ms=str(int(self._exit_ms * 1000)))
                return None
            # Still active, nothing to do
            return None

        # --- LOST_GRACE ---
        if state == FeatureState.LOST_GRACE:
            if raw:
                # Recovered within grace period
                t["state"] = FeatureState.ACTIVE
                t["state_since"] = t.get("active_since", now)  # preserve original active_since
                logger.debug("[stabilizer] %s: LOST_GRACE -> ACTIVE (recovered)", fid)
                log_trace("gesture", "gesture.stabilizer", "feature_recovered",
                          feature_id=fid)
                return None
            exit_ms = self._overrides.get(fid, {}).get("exit_s", self._exit_ms)
            if elapsed >= exit_ms:
                # Check min_hold
                active_elapsed = now - t.get("active_since", 0)
                min_hold = self._overrides.get(fid, {}).get("min_hold_s", self._min_hold_ms)
                if active_elapsed < min_hold:
                    logger.debug("[stabilizer] %s: exit blocked by min_hold (%.0f < %.0fms)",
                                fid, active_elapsed * 1000, self._min_hold_ms * 1000)
                    return None  # don"t exit yet
                t["state"] = FeatureState.INACTIVE
                t["state_since"] = now
                t["active_since"] = 0.0
                logger.info("[stabilizer] %s: LOST_GRACE -> INACTIVE (ended, exit_debounce=%.0fms)",
                            fid, self._exit_ms * 1000)
                log_trace("gesture", "gesture.stabilizer", "feature_ended",
                          feature_id=fid, exit_ms=str(int(self._exit_ms * 1000)))
                return GestureEvent(fid, "ended", now)
            return None

        return None

    def set_feature_config(self, feature_id: str, enter_ms: int | None = None,
                           exit_ms: int | None = None, min_hold_ms: int | None = None) -> None:
        """Override global debounce defaults for a specific feature.
        Positive values override; None or -1 means use global default.
        If multiple mappings share a feature, the highest-priority mapping wins
        (caller should call set_feature_config only for the winning mapping).
        """
        override = {}
        if enter_ms is not None and enter_ms > 0:
            override["enter_s"] = enter_ms / 1000.0
        if exit_ms is not None and exit_ms > 0:
            override["exit_s"] = exit_ms / 1000.0
        if min_hold_ms is not None and min_hold_ms > 0:
            override["min_hold_s"] = min_hold_ms / 1000.0
        if override:
            self._overrides[feature_id] = override
            logger.debug("[stabilizer] %s override: %s", feature_id, override)

    def get_state(self, feature_id: str) -> FeatureState:
        """Get current state of a feature."""
        t = self._tracking.get(feature_id)
        return t["state"] if t else FeatureState.INACTIVE

    def is_active_or_grace(self, feature_id: str) -> bool:
        """True if feature is ACTIVE or in LOST_GRACE (should keep displaying)."""
        state = self.get_state(feature_id)
        return state in (FeatureState.ACTIVE, FeatureState.LOST_GRACE)

    def reset(self) -> None:
        """Reset all tracking."""
        self._tracking.clear()
        logger.debug("[stabilizer] All features reset")


__all__ = ["GestureStabilizer", "GestureEvent", "FeatureState"]

