"""轻量级诊断上下文：trace_id 生成 + 节流日志。"""
import logging
import time
import uuid

logger = logging.getLogger("diagnostics")

# 全局 debug 配置引用（启动时由 main.py 注入）
_debug_cfg: dict = {}


def init_debug_config(cfg: dict) -> None:
    """注入 debug 配置。"""
    global _debug_cfg
    _debug_cfg = cfg


def new_trace_id() -> str:
    """生成 8 位短 trace_id。"""
    return uuid.uuid4().hex[:8]


def log_trace(trace_id: str, component: str, event: str, level: str = "info", **data):
    """结构化诊断日志。所有 data 会被序列化为 key=value 格式。"""
    extra = " ".join(f"{k}={v}" for k, v in data.items())
    msg = f"[trace={trace_id}] [{component}] {event} {extra}".strip()
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.debug(msg)  # trace 级默认走 DEBUG


class Throttle:
    """节流器：同 key 至少间隔 interval_ms 才输出。"""

    def __init__(self, interval_ms: int = 1000, max_entries: int = 1000):
        self._interval = interval_ms / 1000.0
        self._last: dict[str, float] = {}
        self._max_entries = max_entries

    def should_log(self, key: str) -> bool:
        now = time.time()
        if key not in self._last or now - self._last[key] >= self._interval:
            if len(self._last) >= self._max_entries:
                oldest = min(self._last, key=lambda k: self._last[k])
                del self._last[oldest]
            self._last[key] = now
            return True
        return False
