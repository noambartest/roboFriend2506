import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam?")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipeexpects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp.solutions.hands.HAND_CONNECTIONS
                )

        cv2.imshow("MediaPipe Hands – demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
