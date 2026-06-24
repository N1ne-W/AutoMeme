"""Mapping engine: feature vector + voice keyword -> mapping entry match."""
import json
import logging
import os

logger = logging.getLogger(__name__)

try:
    from src.diagnostics import log_trace
except ImportError:
    def log_trace(*args, **kwargs): pass


class MappingEntry:
    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.name: str = data.get("name", "")
        self.enabled: bool = data.get("enabled", True)
        conditions = data.get("conditions", {})
        self.condition_type: str = conditions.get("type", "all")
        self.features: list[str] = conditions.get("features", [])
        self.voice_keywords: list[str] = conditions.get("voice_keywords", [])
        actions = data.get("actions", {})
        self.image_path: str | None = actions.get("image")
        self.audio_path: str | None = actions.get("audio")
        self.action_mode: str = actions.get("mode", "image")
        self.priority: int = data.get("priority", 0)
        self.cooldown_ms: int = data.get("cooldown_ms", 3000)
        self.debounce_frames: int = data.get("debounce_frames", 8)
        self.display_mode: str = data.get("display_mode", "hold")  # "hold" | "duration"
        self.duration_ms: int = data.get("duration_ms", 2000)
        # Gesture stabilizer config (per-mapping override, -1 = use global default)
        self.enter_debounce_ms: int = data.get("enter_debounce_ms", -1)
        self.exit_debounce_ms: int = data.get("exit_debounce_ms", -1)
        self.min_hold_ms: int = data.get("min_hold_ms", -1)


class MappingEngine:
    def __init__(self):
        self._mappings: list[MappingEntry] = []
        self._audio_keyword_map: dict[str, list[MappingEntry]] = {}

    @property
    def mappings(self) -> list[MappingEntry]:
        return self._mappings

    def load(self, default_path: str, user_path: str | None = None) -> None:
        merged: dict[str, dict] = {}
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                default_data = json.load(f)
            for m in default_data.get("mappings", []):
                merged[m["id"]] = m
            logger.info("Loaded %d default mappings", len(default_data.get("mappings", [])))
        if user_path and os.path.exists(user_path):
            with open(user_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            for m in user_data.get("mappings", []):
                merged[m["id"]] = m
            logger.info("Loaded %d user mappings", len(user_data.get("mappings", [])))
        self._mappings = [MappingEntry(m) for m in merged.values()]
        self._mappings.sort(key=lambda m: -m.priority)
        self._audio_keyword_map.clear()
        for m in self._mappings:
            if not m.enabled:
                continue
            for kw in m.voice_keywords:
                kw_lower = kw.lower()
                if kw_lower not in self._audio_keyword_map:
                    self._audio_keyword_map[kw_lower] = []
                self._audio_keyword_map[kw_lower].append(m)
        logger.info("Total mappings: %d", len(self._mappings))
        # Startup asset check
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for m in self._mappings:
            if m.image_path and m.enabled:
                full = os.path.join(project_root, "assets", m.image_path)
                if not os.path.exists(full):
                    logger.warning("[config] Asset missing: %s -> %s", m.id, m.image_path)
                    log_trace("config", "mapping.engine", "asset_missing",
                              mapping_id=m.id, path=m.image_path)

    def match_vision(self, features: dict[str, bool]) -> MappingEntry | None:
        active_features = [k for k, v in features.items() if v]
        if not active_features:
            return None
        for m in self._mappings:
            if not m.enabled or not m.features:
                continue
            if self._evaluate_condition(m, features):
                log_trace("map", "mapping.engine", "mapping_matched",
                          features=",".join(active_features), mapping_id=m.id,
                          priority=str(m.priority), image=m.image_path or "null")
                return m
        log_trace("map", "mapping.engine", "mapping_missed",
                  features=",".join(active_features))
        return None

    def match_audio(self, keyword: str) -> MappingEntry | None:
        matches = self._audio_keyword_map.get(keyword.lower(), [])
        matches.sort(key=lambda m: -m.priority)
        for m in matches:
            if m.enabled:
                return m
        return None

    def _evaluate_condition(self, mapping: MappingEntry, features: dict[str, bool]) -> bool:
        if not mapping.features:
            return False
        results = [features.get(f, False) for f in mapping.features]
        if mapping.condition_type == "all":
            return all(results)
        else:
            return any(results)
