import os
from cv2 import aruco

# Robot Params #
nuc_ip = "192.168.1.6"
robot_ip = "192.168.1.11"
laptop_ip = "192.168.1.22" # "128.84.103.13"
sudo_password = "ning7412"
robot_type = "fr3"  # 'panda' or 'fr3'
robot_serial_number = "295341-2320008"

# Camera ID's #
hand_camera_id = "243222071972"
varied_camera_1_id = "243522075067"
varied_camera_2_id = ""

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = "C1cW3BDf6KTrdKSaVShui1MXCvLsLs"

# Code Version [DONT CHANGE] #
droid_version = "1.3"

