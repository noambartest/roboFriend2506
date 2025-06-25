"""
collect_landmarks.py – save hand‑landmark vectors for ONE gesture
─────────────────────────────────────────────────────────────────
• הפעל פעם אחת לכל מחווה (Rock / Paper / Scissors).
• לוחצים **S** כדי לשמור דגימה (63 ערכים) כ‑.npy.
• Esc לסיום.
"""

import cv2
import os
import numpy as np
import mediapipe as mp

# ─────────────── settings ───────────────
SAVE_DIR = "data/rock"   # ← change for paper / scissors
os.makedirs(SAVE_DIR, exist_ok=True)

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found")

index = len(os.listdir(SAVE_DIR))


def lmk_to_vec(hand_landmarks):
    """Flatten NormalizedLandmarkList → (63,) numpy float32"""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()

print("✋ Collecting – press S (Esc = quit)")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        else:
            hand = None

        # overlay sample count
        cv2.putText(frame, f"Saved: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Collect – press S", frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('s'), ord('S')):
            if hand:
                vec = lmk_to_vec(hand)
                np.save(os.path.join(SAVE_DIR, f"{index:04}.npy"), vec)
                index += 1
                print(f"💾 saved {index}")
            else:
                print("⛔ No hand detected – try again")

        if k == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

print("👋 Finished. Total samples:", index)
