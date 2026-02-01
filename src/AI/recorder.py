import cv2
import mediapipe as mp
import os
import time
import numpy as np
from pynput import mouse

# 在最上面加
recording_indicator_color = (0, 255, 0)   # 绿色 = 录制中
idle_indicator_color = (0, 0, 255)        # 红色 = 未录制

# ========== 1. 输入动作名字 ==========
label = input("请输入动作名称（例如 Donk）：")
base_dir = "../dataset"
save_dir = os.path.join(base_dir, label)
os.makedirs(save_dir, exist_ok=True)
print(f"数据将保存到: {save_dir}")

# ========== 2. MediaPipe 初始化 ==========
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ========== 3. 提取特征函数 ==========
def extract_features(results):
    if not results.face_landmarks:
        return None

    nose = results.face_landmarks.landmark[1]
    l_corner = results.face_landmarks.landmark[61]
    r_corner = results.face_landmarks.landmark[291]

    l_index = None
    r_index = None
    if results.left_hand_landmarks:
        l_index = results.left_hand_landmarks.landmark[8]
    if results.right_hand_landmarks:
        r_index = results.right_hand_landmarks.landmark[8]

    features = [
        nose.x, nose.y,
        l_corner.x, l_corner.y,
        r_corner.x, r_corner.y,
        (l_index.x if l_index else 0), (l_index.y if l_index else 0),
        (r_index.x if r_index else 0), (r_index.y if r_index else 0),
    ]
    return features

# ========== 4. 鼠标状态 ==========
mouse_pressed_time = None
recording = False
record_start_time = None
sample_count = 0

def on_click(x, y, button, pressed):
    global mouse_pressed_time
    if button == mouse.Button.left:
        if pressed:
            mouse_pressed_time = time.time()
        else:
            mouse_pressed_time = None

listener = mouse.Listener(on_click=on_click)
listener.start()

# ========== 5. 摄像头 ==========
cap = cv2.VideoCapture(0)

print("按住鼠标左键 2 秒开始录制，录制 1 秒")

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    current_time = time.time()

    # 判断是否按住 2 秒
    if mouse_pressed_time and not recording:
        if current_time - mouse_pressed_time >= 2:
            recording = True
            record_start_time = current_time
            sample_count = 0
            print("🎥 开始录制 1 秒...")

    # 正在录制
    if recording:
        features = extract_features(results)
        if features:
            filename = os.path.join(save_dir, f"{int(time.time()*1000)}.npy")
            np.save(filename, np.array(features))
            sample_count += 1

        # 录制 1 秒结束
        if current_time - record_start_time >= 1:
            recording = False
            mouse_pressed_time = None
            print(f"✅ 录制完成，共保存 {sample_count} 条样本")
            print("再次按住鼠标左键 2 秒可继续录制")

    # ========== 视觉反馈部分 ==========
    h, w, _ = frame.shape

    # 录制状态指示灯（左上角）
    if recording:
        cv2.circle(frame, (30, 30), 10, recording_indicator_color, -1)
        cv2.putText(frame, "RECORDING", (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.circle(frame, (30, 30), 10, idle_indicator_color, -1)
        cv2.putText(frame, "IDLE", (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 显示检测到的关键点数量（调试用）
    info = []
    if results.face_landmarks:
        info.append("Face")
    if results.left_hand_landmarks:
        info.append("Left Hand")
    if results.right_hand_landmarks:
        info.append("Right Hand")

    cv2.putText(frame, f"Detected: {', '.join(info) if info else 'None'}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Recorder", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
