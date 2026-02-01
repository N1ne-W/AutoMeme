import cv2
import mediapipe as mp
import os
import time
import numpy as np
from pynput import mouse


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