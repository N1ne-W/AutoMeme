"""双手比心检测。"""
import numpy as np
from .base import BaseFeature, FeatureContext


class HeartDetector(BaseFeature):
    @property
    def category(self) -> str:
        return "hand"

    def detect(self, ctx: FeatureContext) -> bool:
        """双手拇指和食指围成心形：需要双手都存在，且指尖靠近。"""
        if ctx.left_hand_landmarks is None or ctx.right_hand_landmarks is None:
            return False

        left_lm = ctx.left_hand_landmarks.landmark
        right_lm = ctx.right_hand_landmarks.landmark

        # 左手拇指尖(4)与右手拇指尖(4)距离
        l_thumb = left_lm[4]
        r_thumb = right_lm[4]

        # 左手食指尖(8)与右手食指尖(8)距离
        l_index = left_lm[8]
        r_index = right_lm[8]

        thumb_dist = np.sqrt((l_thumb.x - r_thumb.x)**2 + (l_thumb.y - r_thumb.y)**2)
        index_dist = np.sqrt((l_index.x - r_index.x)**2 + (l_index.y - r_index.y)**2)

        threshold = self.params.get("distance_threshold", 0.08)
        return thumb_dist < threshold and index_dist < threshold
