from copy import deepcopy

import cv2
import numpy as np

from droid.misc.parameters import hand_camera_id
from droid.misc.time import time_ms

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    print("WARNING: You have not setup the realsense cameras, and currently cannot use them")

def gather_cameras():
    all_cameras = []
    print("get Cameras: ")
    try:
        cameras = rs.context().query_devices()
    except NameError:
        return []

    for cam in cameras:
        cam = Realsense(cam)
        all_cameras.append(cam)

    return all_cameras


resize_func_map = {"cv2": cv2.resize, None: None}


class Realsense:
    def __init__(self, camera):
        # Save Parameters #
        self.serial_number = str(camera.get_info(rs.camera_info.serial_number))
        self.is_hand_camera = self.serial_number == hand_camera_id
        self.high_res_calibration = False
        self.current_mode = None
        self._current_params = None
        self._extriniscs = {}

        # Open Camera #
        print("Opening Realsense: ", self.serial_number)

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # Non-Permenant Values #
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        # Permenant Values #
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    ### Camera Modes ###
    def set_calibration_mode(self):
        # Set Parameters #
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.rgb_resolution = (1920, 1080)
        self.depth_resolution = (1280, 720)
        self.resizer_resolution = (0, 0)

        self._configure_camera()
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        # Set Parameters #
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        if self.resize_func is None:
            self.rgb_resolution = (640, 480)
            self.resizer_resolution = (0, 0)
        else:
            self.rgb_resolution = (640, 480)
            self.resizer_resolution = self.traj_resolution

        self._configure_camera()
        self.current_mode = "trajectory"

    def _configure_camera(self, init_params=None):
        # Close Existing Camera #
        self.disable_camera()

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 30)

        cfg = pipeline.start(config)
        self.pipeline = pipeline

        self.align = rs.align(rs.stream.color)

        profile = cfg.get_stream(rs.stream.color) 
        intr_params = profile.as_video_stream_profile().get_intrinsics()

        # Save Intrinsics #
        # careful with the latency
        self.latency = int(2.5 * (1e3 / 30))
        intr = self._process_intrinsics(intr_params)
        self._intrinsics = {
            self.serial_number : intr
        }

    ### Calibration Utilities ###
    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array([[params.fx, 0, params.ppx], [0, params.fy, params.ppy], [0, 0, 1]])
        intrinsics["distCoeffs"] = np.array(list(params.coeffs))
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    ### Recording Utilities ###
    def start_recording(self, filename):
        assert filename.endswith(".bag")
        if hasattr(self, "pipeline"):
            self.pipeline.stop()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_record_to_file(filename)
        pipeline.start(config)
        self.pipeline = pipeline

    def stop_recording(self):
        self.pipeline.stop()

    ### Basic Camera Utilities ###
    def _process_frame(self, frame):
        image = np.asanyarray(frame.get_data(), dtype=np.uint8)
        image = deepcopy(image)
        if self.resizer_resolution == (0, 0):
            return image
        return self.resize_func(image, self.resizer_resolution)

    def read_camera(self):
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}, {}

        # Read Camera #
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        # Benchmark Latency #
        received_time = frames.get_timestamp()
        timestamp_dict[self.serial_number + "_frame_received"] = received_time
        timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        # Return Data #
        data_dict = {}

        if self.image:
            color_frame = aligned_frames.get_color_frame()
            data_dict["image"] = {self.serial_number: self._process_frame(color_frame)}

        return data_dict, timestamp_dict

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if hasattr(self, "pipeline"):
            self._current_params = None
            self.pipeline.stop()
            self.pipeline = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"

# testing
if __name__ == "__main__":
    cameras = gather_cameras()
    for camera in cameras:
        print(camera.serial_number)
        print(camera.is_hand_camera)
        print(camera.high_res_calibration)
        print(camera.current_mode)
        print(camera._current_params)
        print(camera._extriniscs)
        camera.set_calibration_mode()
        print(camera.get_intrinsics())