import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,    # détecter 2 mains
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

def detect_hand(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_coords_list = []  # Liste pour stocker les coordonnées de chaque main

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coords = []
            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                coords.append((cx, cy))
            hand_coords_list.append(coords)

    return frame, hand_coords_list
