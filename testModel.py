"""
Rock–Paper–Scissors:
====================
• זיהוי מחווה ביד (MediaPipe + מודל Joblib)
• משחק מול "רובוט" – היד של Allegro ב-PyBullet מציגה את בחירת הבוט
• GUI קטן לבחירת מספר סיבובים
• OpenCV מציג מצלמה, ניקוד ואייקונים

מקש Esc בחלון המצלמה → יציאה.
"""

# ------------------------------------------------------------
# 0) Imports & static assets
# ------------------------------------------------------------
from pathlib import Path
import time, random, math, threading, queue, sys, re

import cv2, numpy as np, joblib, mediapipe as mp
import ttkbootstrap as tb
from ttkbootstrap.constants import *

import pybullet as p, pybullet_data

# ---------- AI files ----------
MODEL_PATH = Path("rps_landmarks.joblib")
LABELS_TXT = Path("label_map.txt")
ICON_DIR   = Path("icons")                   # rock.png / paper.png / scissors.png

clf = joblib.load(MODEL_PATH)
CLASS_NAMES = [ln.strip().title() for ln in LABELS_TXT.read_text().splitlines()]
MOVES = ["Rock", "Paper", "Scissors"]

# MediaPipe Hands – **model_complexity חייב להיות int**
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,              # 0/1/2  (היה הבאג)
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)
draw_utils = mp.solutions.drawing_utils

# PNG icons (RGBA)
icons = {m: cv2.imread(str(ICON_DIR / f"{m.lower()}.png"), cv2.IMREAD_UNCHANGED)
         for m in MOVES}

def overlay_icon(bgr: np.ndarray, icon: np.ndarray, x: int, y: int) -> None:
    h, w = icon.shape[:2]
    if y + h > bgr.shape[0] or x + w > bgr.shape[1]:
        return
    roi = bgr[y:y+h, x:x+w]
    rgb, alpha = icon[:, :, :3], icon[:, :, 3] / 255.0
    inv = 1 - alpha
    for c in range(3):
        roi[:, :, c] = alpha * rgb[:, :, c] + inv * roi[:, :, c]

def decide_winner(user: str, bot: str) -> str:
    if user == bot:
        return "Draw"
    wins = {("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")}
    return "User" if (user, bot) in wins else "Robot"

# ------------------------------------------------------------
# 1)  PyBullet worker (runs in background thread)
# ------------------------------------------------------------
def setup_allegro_hand():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    from robot_descriptions import allegro_hand_description
    hand_uid = p.loadURDF(
        allegro_hand_description.URDF_PATH,
        basePosition=[0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
    )

    # collect revolute joints
    name2id = {p.getJointInfo(hand_uid, j)[1].decode(): j
               for j in range(p.getNumJoints(hand_uid))}
    rev = [jid for jid in name2id.values()
           if p.getJointInfo(hand_uid, jid)[2] == p.JOINT_REVOLUTE]

    # sort numerically then group 4-by-4
    rev.sort(key=lambda jid: tuple(int(n) for n in re.findall(r"\d+", p.getJointInfo(hand_uid, jid)[1].decode())))
    fingers = [rev[i:i+4] for i in range(0, len(rev), 4)]
    # sort by X (thumb left, pinky right)
    fingers.sort(key=lambda g: p.getLinkState(hand_uid, g[0])[4][0])

    thumb, index, middle, pinky = fingers[:4]

    # helper motions
    def open_finger(g, mcp=-0.3, pip=0, dip=0, spread=0):
        for jid, tgt in zip(g, (mcp, pip, dip, spread)):
            p.setJointMotorControl2(hand_uid, jid, p.POSITION_CONTROL, tgt, force=4)

    def curl_finger(g, curl=1.5):
        for jid in g[:3]:
            p.setJointMotorControl2(hand_uid, jid, p.POSITION_CONTROL, curl, force=4)

    def rock():
        for g in fingers: curl_finger(g)

    def paper():
        for g in fingers: open_finger(g)

    def scissors():
        # index straight, middle bent+tilted, others curled
        open_finger(index,  mcp=-0.35, pip=0.0, dip=0.0, spread= 0.10)
        open_finger(middle, mcp=-0.15, pip=0.25, dip=0.15, spread=-0.25)
        curl_finger(thumb)
        curl_finger(pinky)

    poses = {"Rock": rock, "Paper": paper, "Scissors": scissors}
    return hand_uid, poses, rev

def pybullet_thread(cmd_q: queue.Queue[str], stop_ev: threading.Event):
    hand_uid, poses, rev = setup_allegro_hand()
    current = "Paper"
    poses[current]()
    t = 0.0
    while not stop_ev.is_set():
        # bring latest command
        try:
            while True:
                current = cmd_q.get_nowait()
                poses[current]()
        except queue.Empty:
            pass
        # idle wrist motion
        p.setJointMotorControl2(hand_uid, rev[0], p.POSITION_CONTROL,
                                targetPosition=0.35*math.sin(t), force=8)
        p.stepSimulation()
        time.sleep(1/240)
        t += 0.01
    p.disconnect()

# ------------------------------------------------------------
# 2)  OpenCV game (main thread)
# ------------------------------------------------------------
def play_game(best_of: int, menu_root: tb.Window,
              cmd_q: queue.Queue[str], stop_ev: threading.Event):
    score = {"User": 0, "Robot": 0, "Draw": 0}; rounds = 0
    WAIT, COUNT, SHOW, DONE = range(4)
    state, until = WAIT, 0.0
    COUNT_SEC, SHOW_SEC = 3, 2
    last_bot = "Paper"; cmd_q.put(last_bot)
    banner = ""

    cap = cv2.VideoCapture(0); ok, _ = cap.read()
    if not ok:
        tb.messagebox.showerror("Camera", "Webcam not found"); return

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            now = time.time()

            # header
            head = f"User {score['User']}  Bot {score['Robot']}  Draw {score['Draw']}    {rounds}/{best_of}"
            cv2.putText(frame, head, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

            # ----------------------------------------------------------------
            if state == WAIT:
                cv2.putText(frame, "Press SPACE", (10, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)

            elif state == COUNT:
                cv2.putText(frame, str(int(until-now)+1), (10, frame.shape[0]-60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,255,0), 3)
                if now >= until:
                    # --- detect user move ---
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = mp_hands.process(rgb)
                    if res.multi_hand_landmarks:
                        hnd = res.multi_hand_landmarks[0]
                        draw_utils.draw_landmarks(frame, hnd, mp.solutions.hands.HAND_CONNECTIONS)
                        vec = np.float32([[lm.x,lm.y,lm.z] for lm in hnd.landmark]).flatten()[None,:]
                        idx = int(clf.predict(vec)[0])
                        conf = clf.predict_proba(vec)[0][idx]
                        user_move = CLASS_NAMES[idx] if conf >= 0.8 else random.choice(MOVES)
                    else:
                        user_move = random.choice(MOVES)

                    # bot random & robot hand
                    last_bot = random.choice(MOVES); cmd_q.put(last_bot)
                    winner = decide_winner(user_move, last_bot)
                    score[winner] += 1; rounds += 1
                    banner = f"Winner: {winner.upper()}"
                    state, until = SHOW, now + SHOW_SEC

            elif state == SHOW:
                cv2.putText(frame, banner, (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,255), 2)
                cv2.putText(frame, f"BOT: {last_bot}", (10, frame.shape[0]-30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (200,220,255), 2)
                icon = icons[last_bot]
                if icon is not None:
                    overlay_icon(frame, icon, frame.shape[1]-icon.shape[1]-10, 10)
                if now >= until:
                    state = WAIT if rounds < best_of else DONE

            elif state == DONE:
                cv2.putText(frame, "GAME OVER  –  Esc", (10, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            # ----------------------------------------------------------------
            cv2.imshow("ROBOFRIEND", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == 32 and state == WAIT:
                state, until = COUNT, now + COUNT_SEC

    finally:
        stop_ev.set(); cap.release(); cv2.destroyAllWindows(); menu_root.deiconify()

# ------------------------------------------------------------
# 3)  GUI launcher
# ------------------------------------------------------------
def build_launcher(cmd_q: queue.Queue[str], stop_ev: threading.Event):
    root = tb.Window(themename="flatly")
    root.title("ROBOFRIEND – choose rounds")
    root.geometry("+600+300"); root.resizable(False, False)

    tb.Label(root, text="Best-of rounds", font=("Segoe UI", 14, "bold")).pack(pady=(15,5))
    rounds_var = tb.StringVar(value="3")
    tb.Combobox(root, textvariable=rounds_var,
                values=("3","5","10"), width=6, state="readonly").pack(pady=6)

    def start(*_):
        root.withdraw()
        play_game(int(rounds_var.get()), root, cmd_q, stop_ev)

    tb.Button(root, text="Start Game", bootstyle=SUCCESS, command=start).pack(pady=(12,24))
    root.bind("<Return>", start)
    return root

# ------------------------------------------------------------
# 4)  main
# ------------------------------------------------------------
if __name__ == "__main__":
    cmd_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    threading.Thread(target=pybullet_thread, args=(cmd_queue, stop_event),
                     daemon=True).start()
    build_launcher(cmd_queue, stop_event).mainloop()
