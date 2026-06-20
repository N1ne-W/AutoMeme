"""特征提取器：将 MediaPipe 结果转化为布尔特征向量。"""
import json
import logging
import os
from dataclasses import dataclass, field
from .features.base import FeatureContext
from .features import FEATURE_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """单帧特征向量。"""
    frame_id: int
    timestamp: float
    features: dict[str, bool] = field(default_factory=dict)


class FeatureExtractor:
    """根据 features.json 注册的特征列表，逐特征检测并输出 FeatureVector。"""

    def __init__(self, features_config_path: str):
        self._feature_configs: dict[str, dict] = {}
        self._feature_instances: dict[str, object] = {}
        self._load_config(features_config_path)

    def _load_config(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Features config not found: %s", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self._feature_configs = config.get("features", {})

        for name, cfg in self._feature_configs.items():
            module_key = cfg.get("module", "")
            if module_key not in FEATURE_REGISTRY:
                logger.warning("Feature '%s' has unknown module '%s'", name, module_key)
                continue
            detector_cls = FEATURE_REGISTRY[module_key]
            params = cfg.get("params", {})
            self._feature_instances[name] = detector_cls(params)
            logger.debug("Feature loaded: %s (%s)", name, module_key)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_configs.keys())

    def extract(self, results, frame_id: int, timestamp: float) -> FeatureVector:
        """从 MediaPipe results 提取特征向量。"""
        ctx = FeatureContext(
            pose_landmarks=results.pose_landmarks if results else None,
            left_hand_landmarks=results.left_hand_landmarks if results else None,
            right_hand_landmarks=results.right_hand_landmarks if results else None,
            face_landmarks=results.face_landmarks if results else None,
        )

        fv = FeatureVector(frame_id=frame_id, timestamp=timestamp)
        for name, detector in self._feature_instances.items():
            try:
                fv.features[name] = detector.detect(ctx)
            except Exception as e:
                logger.error("Feature '%s' detection failed: %s", name, e)
                fv.features[name] = False
        return fv
