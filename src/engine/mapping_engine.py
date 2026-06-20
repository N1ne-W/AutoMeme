"""映射引擎：特征向量 + 语音关键词 → 映射条目匹配。"""
import json
import logging
import os

logger = logging.getLogger(__name__)


class MappingEntry:
    """单条映射条目。"""
    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.name: str = data.get("name", "")
        self.enabled: bool = data.get("enabled", True)
        conditions = data.get("conditions", {})
        self.condition_type: str = conditions.get("type", "all")  # "all" | "any"
        self.features: list[str] = conditions.get("features", [])
        self.voice_keywords: list[str] = conditions.get("voice_keywords", [])
        actions = data.get("actions", {})
        self.image_path: str | None = actions.get("image")
        self.audio_path: str | None = actions.get("audio")
        self.action_mode: str = actions.get("mode", "image")  # "image" | "audio" | "both"
        self.priority: int = data.get("priority", 0)
        self.cooldown_ms: int = data.get("cooldown_ms", 3000)
        self.debounce_frames: int = data.get("debounce_frames", 8)


class MappingEngine:
    """管理映射表，接受特征向量和语音信号，返回匹配的 TriggerEvent。"""

    def __init__(self):
        self._mappings: list[MappingEntry] = []
        self._audio_keyword_map: dict[str, list[MappingEntry]] = {}

    def load(self, default_path: str, user_path: str | None = None) -> None:
        """加载默认映射 + 用户映射（用户覆盖默认）。"""
        merged: dict[str, dict] = {}

        # 加载默认
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                default_data = json.load(f)
            for m in default_data.get("mappings", []):
                merged[m["id"]] = m
            logger.info("Loaded %d default mappings", len(default_data.get("mappings", [])))

        # 加载用户（覆盖同 ID）
        if user_path and os.path.exists(user_path):
            with open(user_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            for m in user_data.get("mappings", []):
                merged[m["id"]] = m
            logger.info("Loaded %d user mappings", len(user_data.get("mappings", [])))

        self._mappings = [MappingEntry(m) for m in merged.values()]
        self._mappings.sort(key=lambda m: -m.priority)  # 降序

        # 构建语音关键词索引
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

    def match_vision(self, features: dict[str, bool]) -> MappingEntry | None:
        """根据特征向量匹配映射条目（优先级最高者）。"""
        for m in self._mappings:
            if not m.enabled or not m.features:
                continue
            if self._evaluate_condition(m, features):
                return m
        return None

    def match_audio(self, keyword: str) -> MappingEntry | None:
        """根据语音关键词匹配映射条目。"""
        matches = self._audio_keyword_map.get(keyword.lower(), [])
        for m in matches:
            if m.enabled:
                return m
        return None

    def _evaluate_condition(self, mapping: MappingEntry, features: dict[str, bool]) -> bool:
        """评估特征条件。"""
        if not mapping.features:
            return False
        results = [features.get(f, False) for f in mapping.features]
        if mapping.condition_type == "all":
            return all(results)
        else:  # "any"
            return any(results)
