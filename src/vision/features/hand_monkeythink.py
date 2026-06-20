import numpy as np
from .base import BaseFeature, FeatureContext

class MonkeyThinkDetector(BaseFeature):
    @property
    def category(self) -> str:
        return "hand"

    def detect(self, ctx: FeatureContext) -> bool:
        if ctx.face_landmarks is None:
            return False
        nose = ctx.face_landmarks.landmark[1]
        l_corner = ctx.face_landmarks.landmark[61]
        r_corner = ctx.face_landmarks.landmark[291]

        for hand in [ctx.left_hand_landmarks, ctx.right_hand_landmarks]:
            if hand is None:
                continue
            tip = hand.landmark[8]
            dist_x = abs(tip.x - nose.x)
            if dist_x < 0.025:
                continue  # too close to nose -> Donk territory
            d_l = np.sqrt((tip.x - l_corner.x)**2 + (tip.y - l_corner.y)**2)
            d_r = np.sqrt((tip.x - r_corner.x)**2 + (tip.y - r_corner.y)**2)
            if d_l < 0.04 or d_r < 0.04:
                return True
        return False
