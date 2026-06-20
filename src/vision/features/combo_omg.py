import numpy as np
from .base import BaseFeature, FeatureContext
from ._palm_util import is_palm_open

class OMGDetector(BaseFeature):
    @property
    def category(self) -> str:
        return "combo"

    def detect(self, ctx: FeatureContext) -> bool:
        if ctx.face_landmarks is None or ctx.pose_landmarks is None:
            return False
        left_h = ctx.left_hand_landmarks
        right_h = ctx.right_hand_landmarks
        if left_h is None or right_h is None:
            return False

        m_up = ctx.face_landmarks.landmark[13]
        m_down = ctx.face_landmarks.landmark[14]
        mouth_open = abs(m_down.y - m_up.y)
        if mouth_open <= 0.03:
            return False

        if not is_palm_open(left_h) or not is_palm_open(right_h):
            return False

        l_ear = ctx.pose_landmarks.landmark[7]
        r_ear = ctx.pose_landmarks.landmark[8]
        l_mcp = left_h.landmark[5]
        r_mcp = right_h.landmark[5]

        d_l = np.sqrt((l_mcp.x - l_ear.x)**2 + (l_mcp.y - l_ear.y)**2)
        d_r = np.sqrt((r_mcp.x - r_ear.x)**2 + (r_mcp.y - r_ear.y)**2)

        return d_l < 0.12 and d_r < 0.12
