import cv2
import mediapipe as mp
import pygame
import numpy as np
import os

# --- 1. 配置区 ---
ACTION_CONFIG = [
    {
        "id": 1,
        "name": "Donk",
        "image_path": r"C:\Myproject\automeme\assets\images\34ecf09f-b8d1-4848-ad5e-840a94cad80e.png",
        "priority": 2
    },
    {
        "id": 2,
        "name": "MonkeyThink",
        "image_path": r"C:\Myproject\automeme\assets\images\the-original-image-of-the-monkey-thinking-meme-v0-ea1hkdjnx9af1.png",
        "priority": 1
    },
    {
        "id": 3,
        "name": "nfb",
        "image_path": r"C:\Myproject\automeme\assets\images\29f740ef-fcb7-46e5-b2d2-d7396e547cb8.png",
        "priority": 3
    },
    {
        "id": 4,
        "name": "OMG",
        "image_path": r"C:\Myproject\automeme\assets\images\7d565fa76b33f1233e1c1d4c3e52a22a.png",
        "priority": 4
    }
]

# 将判定函数移到类外面，方便全局调用，或者定义为静态方法
def is_palm_open(hand_landmarks):
    if not hand_landmarks: return False
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]
    opened_fingers = 0
    wrist = hand_landmarks.landmark[0]
    for tip_id, mcp_id in zip(tip_ids, mcp_ids):
        tip = hand_landmarks.landmark[tip_id]
        mcp = hand_landmarks.landmark[mcp_id]
        d_tip = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        d_mcp = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
        if d_tip > d_mcp: opened_fingers += 1
    return opened_fingers >= 4

class AutoMemeEngine:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        pygame.init()
        self.screen_w, self.screen_h = 1280, 720
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("AutoMeme - Advanced Mode")
        self.active_meme_id = None
        self.global_alpha = 0
        self.fade_speed = 15
        self.clock = pygame.time.Clock()
        self._load_resources()

    def _load_resources(self):
        for m in ACTION_CONFIG:
            if os.path.exists(m["image_path"]):
                try:
                    img = pygame.image.load(m["image_path"]).convert_alpha()
                    m["img_data"] = pygame.transform.scale(img, (700, 700))
                except: m["img_data"] = None
            else:
                print(f"❌ 警告: 图片不存在: {m['image_path']}")
                m["img_data"] = None

    def get_triggered_action(self, results):
        if not results.pose_landmarks or not results.face_landmarks:
            return None

        # 1. 基础参照点
        nose = results.face_landmarks.landmark[1]
        m_up = results.face_landmarks.landmark[13]
        m_down = results.face_landmarks.landmark[14]
        l_corner = results.face_landmarks.landmark[61]
        r_corner = results.face_landmarks.landmark[291]
        l_ear = results.pose_landmarks.landmark[7]
        r_ear = results.pose_landmarks.landmark[8]

        # 计算开口度
        mouth_open_dist = abs(m_down.y - m_up.y)

        # 2. 收集左右手（如果存在）
        left_h = results.left_hand_landmarks
        right_h = results.right_hand_landmarks

        #模块 4:OMG
        # 逻辑：两只手都要在，且两只手都要张开，且都要靠近耳朵，且张嘴
        if left_h and right_h and mouth_open_dist > 0.03:
            l_mcp = left_h.landmark[5]
            r_mcp = right_h.landmark[5]

            dist_l = np.sqrt((l_mcp.x - l_ear.x) ** 2 + (l_mcp.y - l_ear.y) ** 2)
            dist_r = np.sqrt((r_mcp.x - r_ear.x) ** 2 + (r_mcp.y - r_ear.y) ** 2)

            # 如果双手都在耳朵附近
            if dist_l < 0.12 and dist_r < 0.12:
                if is_palm_open(left_h) and is_palm_open(right_h):
                    return next(m for m in ACTION_CONFIG if m["id"] == 4)

        # --- 优先级 2：单手动作判定 (循环处理) ---
        # 把手放进列表，逐一检查单手逻辑
        hands_to_check = []
        if left_h: hands_to_check.append(left_h)
        if right_h: hands_to_check.append(right_h)

        for hand in hands_to_check:
            tip = hand.landmark[8]
            mcp = hand.landmark[5]

            # 基础距离计算
            dist_x_to_center = abs(tip.x - nose.x)
            dist_y_to_mouth = abs(tip.y - m_up.y)

            # 模块 1：Donk
            if dist_y_to_mouth < 0.05 and dist_x_to_center < 0.025:
                return next(m for m in ACTION_CONFIG if m["id"] == 1)

            # 模块 2：MonkeyThink
            d_l_c = np.sqrt((tip.x - l_corner.x) ** 2 + (tip.y - l_corner.y) ** 2)
            d_r_c = np.sqrt((tip.x - r_corner.x) ** 2 + (tip.y - r_corner.y) ** 2)
            if (d_l_c < 0.04 or d_r_c < 0.04) and dist_x_to_center >= 0.025:
                return next(m for m in ACTION_CONFIG if m["id"] == 2)

            # 模块 3：nfb (只要有一只手满足张开摸耳)
            if mouth_open_dist < 0.03:
                if is_palm_open(hand):
                    d_l_ear = np.sqrt((mcp.x - l_ear.x) ** 2 + (mcp.y - l_ear.y) ** 2)
                    d_r_ear = np.sqrt((mcp.x - r_ear.x) ** 2 + (mcp.y - r_ear.y) ** 2)
                    if d_l_ear < 0.08 or d_r_ear < 0.08:
                        return next(m for m in ACTION_CONFIG if m["id"] == 3)

        return None

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)
            target_meme = self.get_triggered_action(results)

            if self.active_meme_id is None:
                if target_meme:
                    self.active_meme_id = target_meme["id"]
                    self.global_alpha += self.fade_speed
            else:
                if target_meme and target_meme["id"] == self.active_meme_id:
                    self.global_alpha += self.fade_speed
                else:
                    self.global_alpha -= self.fade_speed
                if self.global_alpha <= 0: self.active_meme_id = None

            self.global_alpha = max(0, min(255, self.global_alpha))
            bg_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.screen.blit(pygame.transform.scale(bg_surface, (self.screen_w, self.screen_h)), (0, 0))

            if self.active_meme_id and self.global_alpha > 0:
                m_info = next((m for m in ACTION_CONFIG if m["id"] == self.active_meme_id), None)
                if m_info and m_info["img_data"]:
                    meme_surface = m_info["img_data"].copy()
                    meme_surface.set_alpha(self.global_alpha)
                    self.screen.blit(meme_surface, meme_surface.get_rect(center=(self.screen_w//2, self.screen_h//2)))

            pygame.display.flip()
            self.clock.tick(30)
        cap.release()
        pygame.quit()

if __name__ == "__main__":
    AutoMemeEngine().run()