import numpy as np
from .base import BaseFeature, FeatureContext

class ThumbsUpDetector(BaseFeature):
    """Thumbs up: fist clenched, thumb extended ~90 degrees away from fist."""
    @property
    def category(self) -> str:
        return "hand"

    def detect(self, ctx: FeatureContext) -> bool:
        for hand in [ctx.left_hand_landmarks, ctx.right_hand_landmarks]:
            if hand is None:
                continue
            lm = hand.landmark
            wrist = lm[0]

            # 1. All four fingers curled (fist)
            for tip_id, mcp_id in [(8,5),(12,9),(16,13),(20,17)]:
                d_t = np.sqrt((lm[tip_id].x - wrist.x)**2 + (lm[tip_id].y - wrist.y)**2)
                d_m = np.sqrt((lm[mcp_id].x - wrist.x)**2 + (lm[mcp_id].y - wrist.y)**2)
                if d_t > d_m:
                    break
            else:
                # All 4 curled -> fist confirmed. Now check thumb.
                pass
            if not self._all_curled(lm, wrist):
                continue

            # 2. Thumb EXTENDED: tip far from wrist compared to MCP
            d_tip = np.sqrt((lm[4].x - wrist.x)**2 + (lm[4].y - wrist.y)**2)
            d_mcp2 = np.sqrt((lm[2].x - wrist.x)**2 + (lm[2].y - wrist.y)**2)
            if d_tip <= d_mcp2 * 1.2:
                continue  # thumb not extended

            # 3. Thumb is AWAY from the fist: tip far from index finger tip(8)
            d_thumb_to_index = np.sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
            if d_thumb_to_index < 0.1:
                continue  # thumb wrapped on top of fist

            # 4. Thumb points roughly upward: tip.y < IP.y < MCP.y
            if not (lm[4].y < lm[3].y < lm[2].y):
                continue

            # 5. Thumb tip is clearly above wrist
            if not (lm[4].y < wrist.y - 0.03):
                continue

            return True
        return False

    def _all_curled(self, lm, wrist):
        for tip_id, mcp_id in [(8,5),(12,9),(16,13),(20,17)]:
            d_t = np.sqrt((lm[tip_id].x - wrist.x)**2 + (lm[tip_id].y - wrist.y)**2)
            d_m = np.sqrt((lm[mcp_id].x - wrist.x)**2 + (lm[mcp_id].y - wrist.y)**2)
            if d_t > d_m:
                return False
        return True
