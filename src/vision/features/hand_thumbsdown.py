import numpy as np
from .base import BaseFeature, FeatureContext

class ThumbsDownDetector(BaseFeature):
    """点踩：拇指朝下，其余四指弯曲。"""
    @property
    def category(self) -> str:
        return "hand"

    def detect(self, ctx: FeatureContext) -> bool:
        for hand in [ctx.left_hand_landmarks, ctx.right_hand_landmarks]:
            if hand is None:
                continue
            lm = hand.landmark
            # Thumb: tip(4) below IP(3) below MCP(2) -> pointing down
            thumb_down = lm[4].y > lm[3].y > lm[2].y
            if not thumb_down:
                continue
            # Other 4 fingers curled
            wrist = lm[0]
            fingers_curled = True
            for tip_id, mcp_id in [(8,5),(12,9),(16,13),(20,17)]:
                d_tip = np.sqrt((lm[tip_id].x - wrist.x)**2 + (lm[tip_id].y - wrist.y)**2)
                d_mcp = np.sqrt((lm[mcp_id].x - wrist.x)**2 + (lm[mcp_id].y - wrist.y)**2)
                if d_tip > d_mcp:
                    fingers_curled = False
                    break
            if fingers_curled:
                return True
        return False
