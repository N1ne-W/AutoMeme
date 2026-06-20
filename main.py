"""
AutoMeme - 主程序入口
基于 AI 视觉融合识别的自动吊图跳脸器
License: GNU GPL v3
"""
import os
import sys
import time
import logging
import yaml

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# 确保日志目录存在
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

# --- 日志 ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "automeme.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger("main")

# --- 加载应用配置 ---
def load_app_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, "config", "app.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    log.warning("app.yaml not found, using defaults")
    return {}

app_config = load_app_config()

# --- 导入模块 ---
from src.vision import Camera, HolisticRunner, FeatureExtractor
from src.engine import VisionSignal, StateMachine, MappingEngine


def main():
    log.info("=== AutoMeme 启动 ===")

    camera_cfg = app_config.get("camera", {})
    mp_cfg = app_config.get("mediapipe", {})
    debug_cfg = app_config.get("debug", {})

    # 1. 初始化视觉管道
    camera = Camera(
        index=camera_cfg.get("index", 0),
        width=camera_cfg.get("width", 640),
        height=camera_cfg.get("height", 480),
        fps=camera_cfg.get("fps", 30),
    )
    if not camera.open():
        log.error("摄像头不可用，退出")
        return

    model_path = os.path.join(PROJECT_ROOT, mp_cfg.get("model_path", "assets/models/holistic_landmarker.task"))
    holistic = HolisticRunner(
        model_path=model_path,
        min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
    )
    if not holistic.initialize():
        log.error("MediaPipe 初始化失败，退出")
        camera.release()
        return

    feature_extractor = FeatureExtractor(
        os.path.join(PROJECT_ROOT, "config", "features.json")
    )
    log.info("已注册特征: %s", feature_extractor.feature_names)

    # 2. 初始化引擎
    mapping_engine = MappingEngine()
    mapping_engine.load(
        default_path=os.path.join(PROJECT_ROOT, "config", "mappings.default.json"),
        user_path=os.path.join(PROJECT_ROOT, "config", "mappings.user.json"),
    )
    state_machine = StateMachine(mapping_engine)

    # 3. 初始化 PyGame 渲染
    import pygame
    import cv2
    import numpy as np

    pygame.init()
    renderer_cfg = app_config.get("renderer", {})
    screen_w = renderer_cfg.get("width", 1280)
    screen_h = renderer_cfg.get("height", 720)
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("AutoMeme - Debug Mode")
    clock = pygame.time.Clock()

    # 加载素材缓存
    def load_image(rel_path: str | None):
        if rel_path is None:
            return None
        full = os.path.join(PROJECT_ROOT, "assets", rel_path)
        if not os.path.exists(full):
            return None
        try:
            img = pygame.image.load(full).convert_alpha()
            return pygame.transform.scale(img, (400, 400))
        except Exception as e:
            log.warning("Failed to load image %s: %s", rel_path, e)
            return None

    # 预加载默认素材
    default_images = {}
    for m in mapping_engine._mappings:
        if m.image_path:
            default_images[m.id] = load_image(m.image_path)

    # --- 主循环 ---
    frame_id = 0
    active_meme_id: str | None = None
    fade_alpha = 0
    fade_speed = 15
    running = True

    log.info("进入主循环")
    while running:
        dt = clock.tick(30) / 1000.0

        # 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 摄像头
        ret, frame = camera.read()
        if not ret:
            break

        # MediaPipe 推理
        results = holistic.process(frame)
        now = time.time()

        # 特征提取
        fv = feature_extractor.extract(results, frame_id, now)

        # 引擎更新
        event = state_machine.update(
            vision_signal=VisionSignal(
                features=fv.features,
                frame_id=frame_id,
                timestamp=now,
            )
        )

        # 触发处理
        if event is not None:
            log.info("触发! mapping=%s type=%s image=%s",
                     event.mapping_id, event.action_type, event.image_path)
            active_meme_id = event.mapping_id

        # 淡入淡出
        if active_meme_id is not None:
            if state_machine.state.value >= 3:  # TRIGGERED or COOLDOWN
                fade_alpha = min(255, fade_alpha + fade_speed)
            else:
                fade_alpha = max(0, fade_alpha - fade_speed)
            if fade_alpha <= 0:
                active_meme_id = None

        # --- 渲染 ---
        # 背景：摄像头画面
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bg_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(bg_surface, (screen_w, screen_h)), (0, 0))

        # Debug：绘制骨骼线
        if debug_cfg.get("draw_landmarks", False) and results is not None:
            _draw_landmarks(screen, results, screen_w, screen_h, frame.shape)

        # Debug：状态文字
        if debug_cfg.get("show_status_text", False):
            font = pygame.font.SysFont("consolas", 18)
            status_lines = [
                f"State: {state_machine.state.name}",
                f"Debounce: {state_machine.debounce_progress:.0%}",
                f"Features: {', '.join(f'{k}={v}' for k,v in fv.features.items())}",
                f"Frame: {frame_id}",
            ]
            y = 10
            for line in status_lines:
                surf = font.render(line, True, (0, 255, 0))
                screen.blit(surf, (10, y))
                y += 22

        # 渲染吊图
        if active_meme_id and fade_alpha > 0:
            meme_img = default_images.get(active_meme_id)
            if meme_img is None and event is not None:
                meme_img = load_image(event.image_path)
                if meme_img is not None:
                    default_images[active_meme_id] = meme_img
            if meme_img is not None:
                meme_surface = meme_img.copy()
                meme_surface.set_alpha(fade_alpha)
                screen.blit(
                    meme_surface,
                    meme_surface.get_rect(center=(screen_w // 2, screen_h // 2)),
                )

        pygame.display.flip()
        frame_id += 1

    # --- 清理 ---
    camera.release()
    holistic.close()
    pygame.quit()
    log.info("AutoMeme 退出")


def _draw_landmarks(screen, results, sw: int, sh: int, frame_shape):
    """Debug：绘制 MediaPipe 骨骼线。"""
    import pygame

    h, w = frame_shape[:2]
    scale_x = sw / w
    scale_y = sh / h
    color = (0, 255, 0)

    def pt(lm):
        return (int(lm.x * w * scale_x), int(lm.y * h * scale_y))

    # Pose
    if results.pose_landmarks and results.pose_landmarks.landmark:
        lm_list = results.pose_landmarks.landmark
        for connection in [(11, 12), (11, 23), (12, 24), (23, 24),
                           (23, 25), (24, 26), (25, 27), (26, 28)]:
            if connection[0] < len(lm_list) and connection[1] < len(lm_list):
                a = pt(lm_list[connection[0]])
                b = pt(lm_list[connection[1]])
                pygame.draw.line(screen, color, a, b, 2)

    # Hands
    for hand_lm in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_lm is None:
            continue
        lm_list = hand_lm.landmark
        for connection in [(0, 1), (1, 2), (2, 3), (3, 4),           # thumb
                           (0, 5), (5, 6), (6, 7), (7, 8),           # index
                           (0, 9), (9, 10), (10, 11), (11, 12),      # middle
                           (0, 13), (13, 14), (14, 15), (15, 16),    # ring
                           (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
                           (5, 9), (9, 13), (13, 17)]:
            if connection[0] < len(lm_list) and connection[1] < len(lm_list):
                a = pt(lm_list[connection[0]])
                b = pt(lm_list[connection[1]])
                pygame.draw.line(screen, (255, 255, 0), a, b, 1)

    # Face mesh (simplified)
    if results.face_landmarks and results.face_landmarks.landmark:
        for lm in results.face_landmarks.landmark:
            x, y = pt(lm)
            if 0 <= x < sw and 0 <= y < sh:
                pygame.draw.circle(screen, (0, 128, 255), (x, y), 1)


if __name__ == "__main__":
    main()
