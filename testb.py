import pybullet as p
import pybullet_data
import time

# start GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load plane and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# run simulation
for i in range(1000):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

# disconnect
p.disconnect()