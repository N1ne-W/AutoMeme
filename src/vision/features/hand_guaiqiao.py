import numpy as np
from .base import BaseFeature, FeatureContext

class GuaiqiaoDetector(BaseFeature):
    """Guaiqiao: both hands half-fist in front of body."""
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

        pose = ctx.pose_landmarks.landmark

        # Hand center (MCP of middle finger = landmark 9)
        l_c = left_h.landmark[9]
        r_c = right_h.landmark[9]

        # Hands must be below shoulders (y >= shoulder_y) and above hips (y <= hip_y)
        # Use relaxed horizontal: within 1.5x shoulder width
        l_shoulder = pose[11]
        r_shoulder = pose[12]
        l_hip = pose[23]
        r_hip = pose[24]

        shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_mid_y = (l_hip.y + r_hip.y) / 2

        # Vertical: between shoulder and hip (generous margins)
        margin_v = 0.1
        for pt in [l_c, r_c]:
            if not (shoulder_mid_y - margin_v <= pt.y <= hip_mid_y + margin_v):
                return False

        # Horizontal: within expanded torso width (1.5x shoulder span)
        shoulder_span = abs(r_shoulder.x - l_shoulder.x)
        torso_mid_x = (l_shoulder.x + r_shoulder.x) / 2
        half_width = shoulder_span * 0.9  # generous
        for pt in [l_c, r_c]:
            if not (torso_mid_x - half_width <= pt.x <= torso_mid_x + half_width):
                return False

        # Both hands: all four fingers curled (half-fist)
        def is_fist(hand_lm):
            wrist = hand_lm[0]
            for tip_id, mcp_id in [(8,5),(12,9),(16,13),(20,17)]:
                d_t = np.sqrt((hand_lm[tip_id].x - wrist.x)**2 + (hand_lm[tip_id].y - wrist.y)**2)
                d_m = np.sqrt((hand_lm[mcp_id].x - wrist.x)**2 + (hand_lm[mcp_id].y - wrist.y)**2)
                if d_t > d_m:
                    return False
            return True

        return is_fist(left_h.landmark) and is_fist(right_h.landmark)
