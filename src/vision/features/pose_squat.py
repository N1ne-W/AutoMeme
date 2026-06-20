"""下蹲检测：通过膝盖弯曲角度判定。"""
import numpy as np
from .base import BaseFeature, FeatureContext


class SquatDetector(BaseFeature):
    @property
    def category(self) -> str:
        return "pose"

    def detect(self, ctx: FeatureContext) -> bool:
        if ctx.pose_landmarks is None:
            return False

        lm = ctx.pose_landmarks.landmark
        angle_threshold = self.params.get("angle_threshold", 140)

        # 左膝角度
        hip = lm[23]    # 左髋
        knee = lm[25]   # 左膝
        ankle = lm[27]  # 左踝

        angle = self._compute_angle(hip, knee, ankle)
        return angle < angle_threshold

    @staticmethod
    def _compute_angle(a, b, c) -> float:
        """计算 a-b-c 三点构成的夹角（度数），b 为顶点。"""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
