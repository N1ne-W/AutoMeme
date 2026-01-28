import cv2
import mediapipe as mp
import math
import time

# 初始化模型
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def main():
    # --- 1. 预加载图片 ---
    img_path = r"C:\Myproject\automeme\assets\images\the-original-image-of-the-monkey-thinking-meme-v0-ea1hkdjnx9af1.png"
    monkey_img = cv2.imread(img_path)
    if monkey_img is None:
        print("图片路径错误！")
        return

    cap = cv2.VideoCapture(0)

    # 过渡效果相关变量
    alpha = 0.0  # 当前透明度 (0.0 到 1.0)
    fade_speed = 0.05  # 消失/显现的速度（每帧增加多少，越大越快）

    with mp_pose.Pose(min_detection_confidence=0.7) as pose, \
            mp_hands.Hands(min_detection_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(image_rgb)
            hand_results = hands.process(image_rgb)

            mouth_pos = None
            is_triggering = False

            # 2. 判定动作
            if pose_results.pose_landmarks:
                mouth_pos = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                mouth_pos.y += 0.05

            if hand_results.multi_hand_landmarks and mouth_pos:
                for hand_lms in hand_results.multi_hand_landmarks:
                    dist = get_distance(hand_lms.landmark[8], mouth_pos)
                    if dist < 0.08:
                        is_triggering = True

            # 3. 透明度逻辑核心
            if is_triggering:
                alpha += fade_speed  # 逐渐显现
            else:
                alpha -= fade_speed  # 逐渐消失

            # 限制 alpha 在 0 到 1 之间
            alpha = max(0, min(1, alpha))

            # 4. 渲染过渡效果
            if alpha > 0:
                # 调整图片大小以适应屏幕中心
                meme_resized = cv2.resize(monkey_img, (w // 2, h // 2))
                mw, mh = meme_resized.shape[1], meme_resized.shape[0]
                start_y, start_x = h // 4, w // 4

                # 提取背景区域（ROI）
                roi = frame[start_y:start_y + mh, start_x:start_x + mw]

                # 【混合公式】：结果 = 背景 * (1 - alpha) + 前景 * alpha
                # addWeighted 可以实现两张图的加权融合
                blended = cv2.addWeighted(roi, 1 - alpha, meme_resized, alpha, 0)

                # 把融合后的图块贴回原图
                frame[start_y:start_y + mh, start_x:start_x + mw] = blended

            cv2.imshow('AutoMeme - Smooth Transition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()