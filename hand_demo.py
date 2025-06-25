"""
Allegro-Hand R / P / S
----------------------
r = Rock | p = Paper | s = Scissors (Index straight, Middle tilted) | Space = Random | q = Quit
"""

import pybullet as p, pybullet_data, time, math, random, sys, re
from robot_descriptions import allegro_hand_description

# -------------------------------------------------- #
# 1)  world & model
# -------------------------------------------------- #
p.connect(p.GUI)                                # p.DIRECT  →  head-less
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

hand_uid = p.loadURDF(
    allegro_hand_description.URDF_PATH,
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION,
)

# -------------------------------------------------- #
# 2)  collect revolute joints, group 4×4
# -------------------------------------------------- #
name2id = {p.getJointInfo(hand_uid, j)[1].decode(): j
           for j in range(p.getNumJoints(hand_uid))}
rev = [jid for jid in name2id.values()
       if p.getJointInfo(hand_uid, jid)[2] == p.JOINT_REVOLUTE]

rev.sort(key=lambda jid: tuple(int(x) for x in re.findall(r"\d+", p.getJointInfo(hand_uid, jid)[1].decode())))
groups = [rev[i:i+4] for i in range(0, len(rev), 4)]
if len(groups) < 4:
    sys.exit("⚠️  URDF לא מאורגן כצפוי – חסרים מפרקי REVOLUTE.")

# מיון קבוצות לפי X-position כדי לזהות Pinky
def x_of(group):  return p.getLinkState(hand_uid, group[0])[4][0]
groups.sort(key=x_of)            # שמאל→ימין
thumb_group, index_group, middle_group, pinky_group = groups[:4]

# -------------------------------------------------- #
# 3)  helpers
# -------------------------------------------------- #
def open_finger(group, *, mcp=-0.3, pip=0.0, dip=0.0, spread=0.0):
    jid_mcp, jid_pip, jid_dip, jid_lat = group
    p.setJointMotorControl2(hand_uid, jid_mcp, p.POSITION_CONTROL, targetPosition=mcp, force=4)
    p.setJointMotorControl2(hand_uid, jid_pip, p.POSITION_CONTROL, targetPosition=pip, force=4)
    p.setJointMotorControl2(hand_uid, jid_dip, p.POSITION_CONTROL, targetPosition=dip, force=4)
    p.setJointMotorControl2(hand_uid, jid_lat, p.POSITION_CONTROL, targetPosition=spread, force=2)

def curl_finger(group, *, curl=1.5):
    for jid in group[:3]:
        p.setJointMotorControl2(hand_uid, jid, p.POSITION_CONTROL, targetPosition=curl, force=4)

def apply_all(gs, func):  [func(g) for g in gs]

# -------------------------------------------------- #
# 4)  gestures
# -------------------------------------------------- #
def pose_rock():
    apply_all(groups, curl_finger)

def pose_paper():
    apply_all(groups, open_finger)

def pose_scissors():
    # Index: ישרה; Middle: נטויה + קימור קל; Thumb + Pinky מקופלות
    open_finger(index_group,  mcp=-0.35, pip=0.00, dip=0.00, spread= 0.10)
    open_finger(middle_group, mcp=-0.10, pip=0.25, dip=0.15, spread=-0.25)
    curl_finger(thumb_group)
    curl_finger(pinky_group)

POSES = dict(rock=pose_rock, paper=pose_paper, scissors=pose_scissors)

# -------------------------------------------------- #
# 5)  utilities
# -------------------------------------------------- #
def countdown():
    for n in (3, 2, 1):
        print(n, end="…", flush=True); time.sleep(0.45)
    print()

def idle_motion(t):
    p.setJointMotorControl2(hand_uid, rev[0], p.POSITION_CONTROL,
                            targetPosition=0.4*math.sin(t), force=10)

# -------------------------------------------------- #
# 6)  main loop
# -------------------------------------------------- #
print("r / p / s / Space / q   –   בחר מחווה (פוקוס על חלון PyBullet)")
t = 0.0
while True:
    idle_motion(t)

    keys = p.getKeyboardEvents()
    if ord('q') in keys: sys.exit(0)

    ch = None
    if ord('r') in keys:   ch = "rock"
    elif ord('p') in keys: ch = "paper"
    elif ord('s') in keys: ch = "scissors"
    elif ord(' ') in keys: ch = random.choice(list(POSES))

    if ch:
        print(">>>", ch); countdown(); POSES[ch]()

    p.stepSimulation(); time.sleep(1/240); t += 0.01
