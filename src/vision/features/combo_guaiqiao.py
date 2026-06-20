import numpy as np
from .base import BaseFeature, FeatureContext

class GuaiQiaoDetector(BaseFeature):
    """GuaiQiao: both hands half-fisted in front of body."""
    @property
    def category(self) -> str:
        return "combo"

    def detect(self, ctx: FeatureContext) -> bool:
        if ctx.pose_landmarks is None:
            return False
        left_h = ctx.left_hand_landmarks
        right_h = ctx.right_hand_landmarks
        if left_h is None or right_h is None:
            return False

        # Check half-fist for each hand
        if not self._is_half_fist(left_h):
            return False
        if not self._is_half_fist(right_h):
            return False

        # Check hands in front of body
        pose_lm = ctx.pose_landmarks.landmark
        l_shoulder = pose_lm[11]
        r_shoulder = pose_lm[12]
        l_hip = pose_lm[23]
        r_hip = pose_lm[24]

        # Hand position: use middle finger MCP (9) as hand center
        l_center = left_h.landmark[9]
        r_center = right_h.landmark[9]

        # Hand x between shoulders, y between shoulders and hips
        shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_mid_y = (l_hip.y + r_hip.y) / 2

        for hand_center in [l_center, r_center]:
            if not (l_shoulder.x < hand_center.x < r_shoulder.x):
                return False
            if not (shoulder_mid_y < hand_center.y < hip_mid_y + 0.15):
                return False

        return True

    def _is_half_fist(self, hand) -> bool:
        """Check if hand is half-fisted: fingers partially curled."""
        lm = hand.landmark
        wrist = lm[0]
        tip_ids = [8, 12, 16, 20]
        mcp_ids = [5, 9, 13, 17]
        curled_count = 0
        for tid, mid in zip(tip_ids, mcp_ids):
            d_tip = np.sqrt((lm[tid].x - wrist.x)**2 + (lm[tid].y - wrist.y)**2)
            d_mcp = np.sqrt((lm[mid].x - wrist.x)**2 + (lm[mid].y - wrist.y)**2)
            if d_mcp == 0:
                continue
            ratio = d_tip / d_mcp
            # Half-fist: tip not fully extended (ratio < 1.0) but not tightly clenched (ratio > 0.4)
            if 0.4 < ratio < 1.0:
                curled_count += 1
        return curled_count >= 3
