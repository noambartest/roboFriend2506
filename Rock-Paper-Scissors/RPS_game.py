"""
ROBO‑FRIEND – Rock‑Paper‑Scissors with a robotic hand 🖐️
========================================================
* MediaPipe + scikit‑learn (joblib) recognises your hand gesture.
* PyBullet + Allegro URDF – robot hand shows the bot’s move.
* OpenCV live camera with score‑bar, icons & countdown.
* ttkbootstrap GUI – slider to choose rounds, live value label & help dialog.

Hot‑keys
--------
SPACE → start a round | R → replay | ESC → quit back to menu
"""
from __future__ import annotations

# ------------------------------------------------------------
# Imports & static assets
# ------------------------------------------------------------
from pathlib import Path
import time, random, math, threading, queue, json, re

import cv2, numpy as np, joblib, mediapipe as mp, ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import messagebox as msg

import pybullet as p, pybullet_data

# Project paths -------------------------------------------------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "rps_landmarks.joblib"   # trained scikit‑learn classifier on hand landmarks
LABELS_TXT = ROOT / "label_map.txt"          # text file: class label per line
ICON_DIR   = ROOT / "icons"                  # directory of PNG icons for Rock/Paper/Scissors
CFG_FILE   = ROOT / "config.json"            # persists last "rounds" selection from launcher

# ------------------------------------------------------------
# Machine‑learning model & resources
# ------------------------------------------------------------
# Load trained classifier (expects flattened 21×3 landmark array).
clf = joblib.load(MODEL_PATH)
# Normalize labels: strip, title‑case for consistency with MOVES list.
CLASS_NAMES = [ln.strip().title() for ln in LABELS_TXT.read_text().splitlines()]
MOVES = ["Rock", "Paper", "Scissors"]

# MediaPipe Hands solution (single, dynamic hand, modest confidence thresholds).
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)
mp_draw  = mp.solutions.drawing_utils

# Preload RGBA icons for overlay at runtime.
icons = {m: cv2.imread(str(ICON_DIR / f"{m.lower()}.png"), cv2.IMREAD_UNCHANGED) for m in MOVES}

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def overlay_icon(bgr: np.ndarray, icon: np.ndarray, x: int, y: int):
    """Overlay an RGBA PNG *icon* onto *bgr* frame at top‑left (x,y) in‑place.
    Performs bounds checking; silently returns if out of frame or icon missing."""
    if icon is None:
        return
    h, w = icon.shape[:2]
    if x < 0 or y < 0 or x + w > bgr.shape[1] or y + h > bgr.shape[0]:
        return
    roi = bgr[y : y + h, x : x + w]
    rgb, a = icon[:, :, :3], icon[:, :, 3:] / 255.0  # alpha channel to [0,1]
    roi[:] = a * rgb + (1 - a) * roi                  # alpha‑blend


def decide_winner(user: str, bot: str) -> str:
    """Return outcome label given *user* and *bot* moves."""
    if user == bot:
        return "Draw"
    wins = {("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")}
    return "User" if (user, bot) in wins else "Robot"

# ------------------------------------------------------------
# PyBullet worker (background thread per game)
# ------------------------------------------------------------

def _setup_hand() -> tuple[int, dict[str, callable], list[int]]:
    """Initialize PyBullet GUI, load Allegro hand, define pose helpers.
    Returns: (body_uid, poses_dict, revolute_joint_ids)."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    from robot_descriptions import allegro_hand_description
    uid = p.loadURDF(
        allegro_hand_description.URDF_PATH,
        basePosition=[0.0, 0.0, 0.1],      # spawn slightly above the plane
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
    )

    # Configure a pleasant camera view & enable shadows for better presentation.
    p.resetDebugVisualizerCamera(
        cameraDistance=1.06,
        cameraYaw=76.80,
        cameraPitch=-22.40,
        cameraTargetPosition=[-0.62, 0.14, -0.13],
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    # Collect revolute joints (the Allegro finger joints) and group into 4‑joint fingers.
    name2id = {p.getJointInfo(uid, j)[1].decode(): j for j in range(p.getNumJoints(uid))}
    rev = [j for j in name2id.values() if p.getJointInfo(uid, j)[2] == p.JOINT_REVOLUTE]
    # Sort by numeric tokens in joint name to ensure stable ordering.
    rev.sort(key=lambda j: tuple(int(n) for n in re.findall(r"\d+", p.getJointInfo(uid, j)[1].decode())))
    # Slice into contiguous groups of 4 (MCP, PIP, DIP, spread) per finger.
    fingers = [rev[i : i + 4] for i in range(0, len(rev), 4)]
    # Sort left‑to‑right in world X so we consistently map to thumb/index/middle/pinky.
    fingers.sort(key=lambda g: p.getLinkState(uid, g[0])[4][0])
    thumb, index, middle, pinky = fingers[:4]

    # --- Low‑level joint helpers -------------------------------------------------
    def open_finger(g, mcp=-0.3, pip=0.0, dip=0.0, spread=0.0):
        """Drive finger *g* to an extended/open posture."""
        for jid, tgt in zip(g, (mcp, pip, dip, spread)):
            p.setJointMotorControl2(uid, jid, p.POSITION_CONTROL, tgt, force=4)

    def curl_finger(g, curl=1.5):
        """Drive first 3 joints of finger *g* into a curled/flexed posture."""
        for jid in g[:3]:
            p.setJointMotorControl2(uid, jid, p.POSITION_CONTROL, curl, force=4)

    # --- High‑level gesture poses ------------------------------------------------
    def pose_rock():
        """Closed fist."""
        for g in fingers:
            curl_finger(g)

    def pose_paper():
        """Open flat hand."""
        for g in fingers:
            open_finger(g)

    def pose_scissors():
        """Index+Middle extended; others curled."""
        open_finger(index, mcp=-0.35, spread=0.10)
        open_finger(middle, mcp=-0.15, pip=0.25, dip=0.15, spread=-0.25)
        curl_finger(thumb)
        curl_finger(pinky)

    return uid, {"Rock": pose_rock, "Paper": pose_paper, "Scissors": pose_scissors}, rev


def pybullet_worker(cmd_q: queue.Queue[str], stop_ev: threading.Event):
    """Physics thread.
    Listens on *cmd_q* for move names; applies corresponding pose.
    Also idles with a gentle wrist wave until *stop_ev* is set."""
    uid, poses, rev = _setup_hand()
    current = "Paper"  # default rest pose
    poses[current]()
    t = 0.0
    while not stop_ev.is_set():
        try:
            # Drain all pending commands (keep the latest one).
            while True:
                current = cmd_q.get_nowait()
                poses[current]()
        except queue.Empty:
            pass
        # Idle wrist oscillation for a bit of life.
        p.setJointMotorControl2(uid, rev[0], p.POSITION_CONTROL, 0.35 * math.sin(t), force=8)
        p.stepSimulation()
        time.sleep(1 / 240)
        t += 0.01
    p.disconnect()

# ------------------------------------------------------------
# OpenCV game loop (main thread)
# ------------------------------------------------------------

def _draw_score(frame, score: dict[str, int], round_id: int, best_of: int):
    """Draw transparent score bar across the top of *frame*."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, f"User {score['User']}",   (10, 25), f, 0.7, ( 60,215, 60), 2)
    cv2.putText(frame, f"Bot {score['Robot']}",   (200,25), f, 0.7, ( 60,215,215), 2)
    cv2.putText(frame, f"Draw {score['Draw']}",   (370,25), f, 0.7, (215,215, 60), 2)
    cv2.putText(frame, f"{round_id}/{best_of}",   (frame.shape[1]-85, 25), f, 0.7, (255,255,255), 2)


def _draw_circle(frame, center, radius, frac):
    """Countdown wedge: *frac* ∈ [0,1] drives the angle."""
    cv2.ellipse(frame, center, (radius, radius), -90, 0, int(360 * frac), (0, 255, 0), -1)
    cv2.circle(frame, center, radius - 6, (0, 0, 0), -1)


def play_game(best_of: int, root: tb.Window) -> None:
    """Main OpenCV loop – runs in the main thread.
    Spawns a PyBullet worker thread, captures webcam frames, detects user gesture,
    decides winner, updates score, and renders UI/state machine."""

    # ---------- PyBullet worker ----------
    cmd_q: queue.Queue[str] = queue.Queue()
    stop_ev = threading.Event()
    threading.Thread(
        target=pybullet_worker,
        args=(cmd_q, stop_ev),
        daemon=True,
    ).start()

    # ---------- game‑state ----------
    score = {"User": 0, "Robot": 0, "Draw": 0}
    rounds = 0
    WAIT, COUNT, SHOW, DONE = range(4)   # simple state machine
    state, until = WAIT, 0.0
    COUNT_T, SHOW_T = 3, 2               # seconds per phase
    last_bot = "Paper"
    cmd_q.put(last_bot)                  # initialize robot pose
    banner = ""

    # ---------- camera ----------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        msg.showerror("Camera", "Webcam not found")
        return

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            if not ok:
                break

            # score‑bar ------------------------------------------------------
            _draw_score(frame, score, rounds, best_of)

            # ------------------------------- state handlers -----------------
            if state == WAIT:
                cv2.putText(
                    frame,
                    "Press SPACE",
                    (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            elif state == COUNT:
                _draw_circle(frame, (60, frame.shape[0] - 80), 40, max((until - now) / COUNT_T, 0))
                if now >= until:
                    # -------- detect user move ---------------------------------
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = mp_hands.process(rgb)

                    if res.multi_hand_landmarks:
                        lm = res.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                        vec = np.float32([[pt.x, pt.y, pt.z] for pt in lm.landmark]).flatten()[None, :]
                        idx  = int(clf.predict(vec)[0])
                        conf =        clf.predict_proba(vec)[0][idx]
                        # If classifier not confident enough -> fallback random move (avoids stale/ambiguous frames).
                        user_mv = (CLASS_NAMES[idx] if conf >= 0.8 else random.choice(MOVES))
                    else:
                        # No hand detected -> random fallback.
                        user_mv = random.choice(MOVES)

                    # Robot chooses move + animate hand.
                    last_bot = random.choice(MOVES)
                    cmd_q.put(last_bot)

                    # Update score.
                    winner = decide_winner(user_mv, last_bot)
                    score[winner] += 1
                    rounds += 1
                    banner = f"Winner: {winner.upper()}"
                    state, until = SHOW, now + SHOW_T

                    # Icon overlays (user bottom‑left, bot top‑right).
                    overlay_icon(frame, icons[user_mv], 10, frame.shape[0] - icons[user_mv].shape[0] - 10)
                    overlay_icon(frame, icons[last_bot], frame.shape[1] - icons[last_bot].shape[1] - 10, 50)

            elif state == SHOW:
                cv2.putText(
                    frame,
                    banner,
                    (15, 70),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
                if now >= until:
                    state = WAIT if rounds < best_of else DONE

            elif state == DONE:
                cv2.putText(
                    frame,
                    "GAME OVER – R to replay / ESC",
                    (30, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                )

            # -----------------------------------------------------------------
            cv2.imshow("ROBOFRIEND", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                       # ESC
                break
            if key == ord(" ") and state == WAIT:
                state, until = COUNT, now + COUNT_T
            if key == ord("r") and state == DONE:
                score = {k: 0 for k in score}
                rounds = 0
                state = WAIT

    finally:
        # Signal physics thread to exit & clean up resources.
        stop_ev.set()
        cap.release()
        cv2.destroyAllWindows()
        root.deiconify()  # restore launcher window


# ------------------------------------------------------------
#  GUI launcher
# ------------------------------------------------------------

def build_launcher() -> tb.Window:
    """Create and return the launcher window that lets user choose #rounds."""
    root = tb.Window(themename="flatly")
    root.title("ROBOFRIEND – Rock-Paper-Scissors")
    root.geometry("+600+300")
    root.resizable(False, False)

    # Try to show a small icon logo (tkinter PhotoImage must be kept referenced).
    try:
        import tkinter as tk
        logo = tk.PhotoImage(file=str(ICON_DIR / "rock.png")).subsample(3, 3)
        tb.Label(root, image=logo).pack(pady=6)
        root.logo = logo  # keep ref
    except Exception:
        pass

    tb.Label(root, text="Choose number of rounds", font=("Segoe UI", 13, "bold")).pack(pady=(5, 2))

    # Load last saved rounds count (if any) so user sees previous preference.
    last = 3
    if CFG_FILE.exists():
        try:
            last = int(json.loads(CFG_FILE.read_text())["rounds"])
        except Exception:
            pass
    rounds_var = tb.IntVar(value=last)

    slider = tb.Scale(
        root,
        from_=1,
        to=15,
        orient="horizontal",
        length=220,
        variable=rounds_var,
        bootstyle="success",
    )
    slider.pack(pady=2)

    tb.Label(root, textvariable=rounds_var, font=("Segoe UI", 10, "bold")).pack(pady=(0, 6))

    def help_box() -> None:
        """Modal help dialog summarizing controls."""
        msg.showinfo(
            "How to play",
            "SPACE – start round\n"
            "ESC – quit to menu\n"
            "R – replay after game over\n\n"
            "Show your hand clearly to the camera.",
        )

    tb.Button(root, text="?", width=3, command=help_box).place(x=5, y=5)

    def start(*_):
        """Persist selection, hide launcher, and enter game loop."""
        CFG_FILE.write_text(json.dumps({"rounds": rounds_var.get()}))  # persist user preference
        root.withdraw()                       # hide launcher while game running
        play_game(rounds_var.get(), root)     # run main loop

    tb.Button(root, text="Start", bootstyle=SUCCESS, width=18, command=start).pack(pady=(2, 8))
    tb.Button(root, text="Exit", bootstyle=SECONDARY, command=root.destroy).pack(pady=(0, 10))

    root.bind("<Return>", start)

    return root


if __name__ == "__main__":
    build_launcher().mainloop()
