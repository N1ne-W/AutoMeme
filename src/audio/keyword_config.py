"""从映射配置中提取语音关键词列表。"""
import json
import os


def extract_keywords(mappings_path: str) -> list[str]:
    """解析 mappings.json，收集所有 voice_keywords。"""
    if not os.path.exists(mappings_path):
        return []
    with open(mappings_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    keywords = set()
    for m in config.get("mappings", []):
        if not m.get("enabled", True):
            continue
        for kw in m.get("conditions", {}).get("voice_keywords", []):
            keywords.add(kw.strip().lower())
    return sorted(keywords)
