from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import calibrate_camera

# Make the robot env
env = RobotEnv()
env.gripper_action_space = "velocity"
controller = VRPolicy()
camera_id = "243222071972"
camera_id2 = "243522075067"

print("Ready?")
# calibrate_camera(env, camera_id, controller)
input("Press Enter to continue third person camera calibration...")
calibrate_camera(env, camera_id2, controller)
