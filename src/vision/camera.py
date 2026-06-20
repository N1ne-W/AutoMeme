"""摄像头采集封装。"""
import cv2
import logging

logger = logging.getLogger(__name__)


class Camera:
    """OpenCV VideoCapture 封装，支持分辨率/帧率配置。"""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self._index = index
        self._width = width
        self._height = height
        self._target_fps = fps
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def open(self) -> bool:
        """打开摄像头，返回是否成功。"""
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            logger.error("Camera index %d not available", self._index)
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        logger.info("Camera opened: %dx%d @ %d FPS", self._width, self._height, self._target_fps)
        return True

    def read(self) -> tuple[bool, object | None]:
        """读取一帧，返回 (成功, BGR帧)。帧已水平翻转。"""
        if not self.is_open:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Camera read failed")
            return False, None
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")
