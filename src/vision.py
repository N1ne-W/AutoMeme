import cv2
import mediapipe as mp


class GestureManager:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # 转换颜色空间（OpenCV默认BGR，MediaPipe需要RGB）
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # 在画面上画出手部骨骼
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                print("检测到手部！")

        return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    manager = GestureManager()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = manager.process_frame(frame)
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
