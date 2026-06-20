"""特征检测器基类。"""
import abc
from dataclasses import dataclass


@dataclass
class FeatureContext:
    """传递给特征检测器的上下文（单帧 MediaPipe 结果）。"""
    pose_landmarks: object | None     # mp PoseLandmark 列表
    left_hand_landmarks: object | None
    right_hand_landmarks: object | None
    face_landmarks: object | None


class BaseFeature(abc.ABC):
    """所有特征检测器的基类。

    子类必须实现 detect(ctx)→bool 和 category 属性。
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @property
    @abc.abstractmethod
    def category(self) -> str:
        """特征类别：'hand' | 'pose' | 'face' | 'combo'"""
        ...

    @abc.abstractmethod
    def detect(self, ctx: FeatureContext) -> bool:
        """返回该特征在当前帧是否触发。"""
        ...
