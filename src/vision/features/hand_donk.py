import numpy as np
from .base import BaseFeature, FeatureContext

class DonkDetector(BaseFeature):
    @property
    def category(self) -> str:
        return "hand"

    def detect(self, ctx: FeatureContext) -> bool:
        if ctx.face_landmarks is None:
            return False
        nose = ctx.face_landmarks.landmark[1]
        m_up = ctx.face_landmarks.landmark[13]

        for hand in [ctx.left_hand_landmarks, ctx.right_hand_landmarks]:
            if hand is None:
                continue
            tip = hand.landmark[8]
            dist_x = abs(tip.x - nose.x)
            dist_y = abs(tip.y - m_up.y)
            if dist_y < 0.05 and dist_x < 0.025:
                return True
        return False
