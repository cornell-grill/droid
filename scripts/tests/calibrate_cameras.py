from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import calibrate_camera

# Make the robot env
env = RobotEnv()
env.gripper_action_space = "velocity"
controller = VRPolicy()
hand_camera_id = "243222071972"
ext_camera_id = "243522075067"
ext_left_camera_id = "243322071546"

input("Ready? Press Enter to continue calibrating camera...")
calibrate_camera(env, ext_camera_id, controller)
# input("Press Enter to continue third person camera calibration...")
# calibrate_camera(env, ext_left_camera_id, controller)