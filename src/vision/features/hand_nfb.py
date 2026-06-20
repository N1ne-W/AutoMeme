import numpy as np
from .base import BaseFeature, FeatureContext
from ._palm_util import is_palm_open

class NFBDetector(BaseFeature):
    @property
    def category(self) -> str:
        return "hand"

    def detect(self, ctx: FeatureContext) -> bool:
        if ctx.face_landmarks is None or ctx.pose_landmarks is None:
            return False
        m_up = ctx.face_landmarks.landmark[13]
        m_down = ctx.face_landmarks.landmark[14]
        mouth_open = abs(m_down.y - m_up.y)
        if mouth_open >= 0.03:  # mouth must be closed
            return False

        l_ear = ctx.pose_landmarks.landmark[7]
        r_ear = ctx.pose_landmarks.landmark[8]

        for hand in [ctx.left_hand_landmarks, ctx.right_hand_landmarks]:
            if hand is None:
                continue
            if not is_palm_open(hand):
                continue
            mcp = hand.landmark[5]
            d_le = np.sqrt((mcp.x - l_ear.x)**2 + (mcp.y - l_ear.y)**2)
            d_re = np.sqrt((mcp.x - r_ear.x)**2 + (mcp.y - r_ear.y)**2)
            if d_le < 0.08 or d_re < 0.08:
                return True
        return False
