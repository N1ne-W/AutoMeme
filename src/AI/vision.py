import cv2
import mediapipe as mp
import pygame
import numpy as np
import os
import etract
import predict

# --- 1. 配置区 ---
ACTION_CONFIG = [
    {
        "id": 1,
        "name": "Donk",
        "image_path": r"C:\Users\ecm22\Desktop\AutoMeme\assets\images\34ecf09f-b8d1-4848-ad5e-840a94cad80e.png",
        "priority": 2
    },
    {
        "id": 2,
        "name": "MonkeyThink",
        "image_path": r"C:\Users\ecm22\Desktop\AutoMeme\assets\images\the-original-image-of-the-monkey-thinking-meme-v0-ea1hkdjnx9af1.png",
        "priority": 1
    }
]


class AutoMemeEngine:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        # 必须开启 face_landmarks 以支持图片2的嘴角判定
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        pygame.init()
        self.screen_w, self.screen_h = 1280, 720
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("AutoMeme - Dual Mode")

        self.active_meme_id = None
        self.global_alpha = 0
        self.fade_speed = 15
        self.clock = pygame.time.Clock()

        self._load_resources()

    def _load_resources(self):
        for m in ACTION_CONFIG:
            if not os.path.exists(m["image_path"]):
                print(f"❌ 警告: 图片不存在: {m['image_path']}")
                m["img_data"] = None
                continue
            try:
                img = pygame.image.load(m["image_path"]).convert_alpha()
                m["img_data"] = pygame.transform.scale(img, (700, 700))
                print(f"✅ 成功加载: {m['name']}")
            except Exception as e:
                print(f"❌ 加载出错: {e}")
                m["img_data"] = None

    def get_triggered_action(self, results):
        """精准判定：中轴线触发图1，两侧触发图2"""
        if not results.pose_landmarks or not results.face_landmarks:
            return None
        features = etract.extract_features(results)
        pred, prob = predict.predict_action(features)
        confidence = prob[pred]
        if confidence < 0.7:
            return None
        for m in ACTION_CONFIG:
            if m["id"] == pred+1:
                return m
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

            # 互斥淡入淡出逻辑
            if self.active_meme_id is None:
                if target_meme:
                    self.active_meme_id = target_meme["id"]
                    self.global_alpha += self.fade_speed
            else:
                if target_meme and target_meme["id"] == self.active_meme_id:
                    self.global_alpha += self.fade_speed
                else:
                    self.global_alpha -= self.fade_speed

                if self.global_alpha <= 0:
                    self.active_meme_id = None

            self.global_alpha = max(0, min(255, self.global_alpha))

            # 渲染流程
            bg_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.screen.blit(pygame.transform.scale(bg_surface, (self.screen_w, self.screen_h)), (0, 0))

            # --- 关键修复：根据 active_meme_id 动态选择图片 ---
            if self.active_meme_id and self.global_alpha > 0:
                # 在配置表中寻找当前激活的 ID 对应的配置
                m_info = next((m for m in ACTION_CONFIG if m["id"] == self.active_meme_id), None)
                if m_info and m_info["img_data"]:
                    meme_surface = m_info["img_data"].copy()
                    meme_surface.set_alpha(self.global_alpha)
                    rect = meme_surface.get_rect(center=(self.screen_w // 2, self.screen_h // 2))
                    self.screen.blit(meme_surface, rect)

            pygame.display.flip()
            self.clock.tick(30)

        cap.release()
        pygame.quit()


if __name__ == "__main__":
    AutoMemeEngine().run()