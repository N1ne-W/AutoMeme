"""剪刀手/比耶检测。"""
import numpy as np
from .base import BaseFeature, FeatureContext


class VictoryDetector(BaseFeature):
    @property
    def category(self) -> str:
        return "hand"

    def detect(self, ctx: FeatureContext) -> bool:
        left_ok = False
        right_ok = False

        if ctx.left_hand_landmarks is not None:
            left_ok = self._check_hand(ctx.left_hand_landmarks)
        if ctx.right_hand_landmarks is not None:
            right_ok = self._check_hand(ctx.right_hand_landmarks)

        # 任一只手做出剪刀手即为真
        return left_ok or right_ok

    def _check_hand(self, hand_landmarks) -> bool:
        """检测单只手是否为剪刀手：
        食指(8) 和中指(12) 伸直，无名指(16) 和小指(20) 弯曲。
        判定方式：指尖到手腕距离 > MCP到手腕距离 → 伸直。
        """
        lm = hand_landmarks.landmark
        wrist = lm[0]

        tip_ids = [8, 12, 16, 20]   # 食、中、无、小指尖
        mcp_ids = [5, 9, 13, 17]    # 对应 MCP 关节
        expected = [True, True, False, False]  # 预期：食伸直、中伸直、无弯曲、小弯曲

        for tip_id, mcp_id, expect_straight in zip(tip_ids, mcp_ids, expected):
            tip = lm[tip_id]
            mcp = lm[mcp_id]
            d_tip = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            d_mcp = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
            is_straight = d_tip > d_mcp
            if is_straight != expect_straight:
                return False
        return True
