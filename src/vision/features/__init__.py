from .base import BaseFeature
from .pose_squat import SquatDetector
from .hand_victory import VictoryDetector
from .hand_heart import HeartDetector
from .hand_donk import DonkDetector
from .hand_monkeythink import MonkeyThinkDetector
from .hand_nfb import NFBDetector
from .combo_omg import OMGDetector
from .hand_thumbsup import ThumbsUpDetector

FEATURE_REGISTRY: dict[str, type[BaseFeature]] = {
    "pose_squat": SquatDetector,
    "hand_victory": VictoryDetector,
    "hand_heart": HeartDetector,
    "hand_donk": DonkDetector,
    "hand_monkeythink": MonkeyThinkDetector,
    "hand_nfb": NFBDetector,
    "combo_omg": OMGDetector,
    "hand_thumbsup": ThumbsUpDetector,
}
