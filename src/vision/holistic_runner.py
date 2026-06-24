"""MediaPipe Holistic 推理封装 (兼容 0.10+ API)。"""
import mediapipe as mp
import cv2
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.components.containers import landmark as lm_containers
import numpy as np
import logging

logger = logging.getLogger(__name__)


class _LegacyLandmarks:
    """兼容包装：让新 API 的 List[NormalizedLandmark] 支持 .landmark 属性。"""
    def __init__(self, landmarks):
        self.landmark = landmarks if landmarks else []

    def __bool__(self):
        return len(self.landmark) > 0

    def __len__(self):
        return len(self.landmark)


class LegacyResults:
    """兼容包装：让新 API 的 HolisticLandmarkerResult 像旧 mp.solutions 的结果。"""
    def __init__(self, result):
        self.pose_landmarks = _LegacyLandmarks(result.pose_landmarks) if (result.pose_landmarks and len(result.pose_landmarks) > 0) else None
        self.face_landmarks = _LegacyLandmarks(result.face_landmarks) if (result.face_landmarks and len(result.face_landmarks) > 0) else None
        self.left_hand_landmarks = _LegacyLandmarks(result.left_hand_landmarks) if (result.left_hand_landmarks and len(result.left_hand_landmarks) > 0) else None
        self.right_hand_landmarks = _LegacyLandmarks(result.right_hand_landmarks) if (result.right_hand_landmarks and len(result.right_hand_landmarks) > 0) else None


class HolisticRunner:
    """MediaPipe HolisticLandmarker (0.10+ API) 封装。"""

    def __init__(self, model_path: str, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._model_path = model_path
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._landmarker: vision.HolisticLandmarker | None = None

    def initialize(self) -> bool:
        try:
            options = vision.HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_face_detection_confidence=self._min_detection_confidence,
                min_pose_detection_confidence=self._min_detection_confidence,
                min_hand_landmarks_confidence=self._min_tracking_confidence,
            )
            self._landmarker = vision.HolisticLandmarker.create_from_options(options)
            logger.info("Holistic initialized: model=%s", self._model_path)
            return True
        except Exception as e:
            logger.error("Holistic init failed: %s", e)
            return False

    def process(self, bgr_frame) -> LegacyResults | None:
        if self._landmarker is None:
            return None
        frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        return LegacyResults(result)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("Holistic closed")
