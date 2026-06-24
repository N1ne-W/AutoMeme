"""引擎信号定义。"""
from dataclasses import dataclass, field


@dataclass
class VisionSignal:
    """视觉管道输入信号。"""
    features: dict[str, bool]  # {"is_squat": True, ...}
    frame_id: int
    timestamp: float


@dataclass
class AudioSignal:
    """音频管道输入信号。"""
    keyword: str
    confidence: float
    timestamp: float


@dataclass
class TriggerEvent:
    """引擎输出的触发事件。"""
    mapping_id: str
    action_type: str          # "image" | "audio" | "both"
    image_path: str | None    # 素材相对路径
    audio_path: str | None
    priority: int
    timestamp: float
    display_mode: str = "hold"
    duration_ms: int = 2000
