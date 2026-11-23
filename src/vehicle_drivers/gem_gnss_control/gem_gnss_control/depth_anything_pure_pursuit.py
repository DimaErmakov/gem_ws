#!/usr/bin/env python3

# ================================================================
# Description: Pure pursuit for GEM. Waypoints come from the front
#   camera: lane keeping on each RGB frame, with depth from
#   Depth-Anything-V2 run on the same frame. GPS is used only to
#   record the start point and to finish once the vehicle is back
#   within stop_distance of it.
# Author: Zillur Rahman, Jiaming Zhang, Hang Cui, Dimitry
# Date: 2026-09-03
# ================================================================

import os
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import pymap3d as pm
import pygame

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32
from pacmod2_msgs.msg import (
    PositionWithSpeed,
    VehicleSpeedRpt,
    GlobalCmd,
    SystemCmdFloat,
    SystemCmdInt,
    GlobalRpt,
    SystemRptFloat,
    SystemRptInt,
)

from sensor_msgs.msg import CameraInfo, Image, NavSatFix, PointCloud2
from septentrio_gnss_driver.msg import INSNavGeod
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PointStamped
import torch
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import sys
import logging

# LaneDetection_DIR = (
#     "/home/dermakov/Documents/fa26_ece484_mp1_team_cobbledtogether_group_52/"
#     "LaneAssist/src/LaneDetection"
# )
LaneDetection_DIR = "/home/gem/gem_ws_gem_or_death/" "LaneAssist/src/LaneDetection"
if LaneDetection_DIR not in sys.path:
    sys.path.append(LaneDetection_DIR)
from detect import LaneDetection

# LaneKeeping_DIR = (
#     "/home/dermakov/Documents/fa26_ece484_mp1_team_cobbledtogether_group_52/"
#     "LaneAssist/src/LaneKeeping"
# )
LaneKeeping_DIR = "/home/gem/gem_ws_gem_or_death/" "LaneAssist/src/LaneKeeping"
if LaneKeeping_DIR not in sys.path:
    sys.path.append(LaneKeeping_DIR)
from lanekeeping import LaneKeeping

# DEPTH_ANYTHING_DIR = "/home/dermakov/Documents/fa26_ece484_mp1_team_cobbledtogether_group_52/Depth-Anything-V2"

DEPTH_ANYTHING_DIR = "/home/gem/gem_ws_gem_or_death/Depth-Anything-V2"
if DEPTH_ANYTHING_DIR not in sys.path:
    sys.path.append(DEPTH_ANYTHING_DIR)

try:
    from depth_anything_v2.dpt import DepthAnythingV2

    HAS_LOCAL_DEPTH_ANYTHING = True
except ImportError:
    HAS_LOCAL_DEPTH_ANYTHING = False

weights_path = (
    "/home/gem/gem_ws_gem_or_death/"
    "Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
)

# GPS finish is armed only after the vehicle leaves this radius around the
# start point, so the program does not finish at the start. Not a ROS parameter.
START_ZONE_RADIUS = 5.0

# Initialize pygame for joystick
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick connected")
joystick = pygame.joystick.Joystick(0)
joystick.init()


class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.wg = wg
        self.iterm = 0
        self.last_e = 0
        self.last_t = None

    def reset(self):
        self.iterm = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e):
        if self.last_t is None:
            dt = 0.0
            de = 0.0
        else:
            dt = t - self.last_t
            de = (e - self.last_e) / dt if dt > 0.0 else 0.0

        self.iterm += e * dt
        if self.wg is not None:
            self.iterm = max(min(self.iterm, self.wg), -self.wg)

        self.last_e = e
        self.last_t = t

        return self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter:
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted[0]


def quaternion_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PurePursuit(Node):

    def __init__(self):
        super().__init__("pure_pursuit_node")
        # Declare parameters with default values

        self.declare_parameter("vehicle_name", "e4")
        vehicle_name = self.get_parameter("vehicle_name").value
        self.declare_parameter("wheelbase", 2.57)
        self.declare_parameter("rate_hz", 20)
        self.declare_parameter("look_ahead", 5.0)
        self.declare_parameter("offset", 1.26)
        self.declare_parameter("origin_lat", 40.0927422)
        self.declare_parameter("origin_lon", -88.2359639)
        self.declare_parameter("desired_speed", 2.6)
        self.declare_parameter("max_accel", 0.5)
        self.declare_parameter("stop_distance", 2.0)

        self.declare_parameter("kick_start_accel", 0.3)
        self.declare_parameter("kick_start_duration", 1.0)

        self.declare_parameter("kick_start_speed_threshold", 0.10)

        self.declare_parameter("pid/kp", 0.6)
        self.declare_parameter("pid/ki", 0.0)
        self.declare_parameter("pid/kd", 0.1)
        self.declare_parameter("pid/wg", 10)

        self.declare_parameter("filter/cutoff", 1.2)
        self.declare_parameter("filter/fs", 30)
        self.declare_parameter("filter/order", 4)

        if vehicle_name == "":
            self.get_logger().warn(
                "No vehicle_name parameter found. No config file loaded, defaulting to 'e4' parameters."
            )
        else:
            self.get_logger().info(f"Using vehicle config: {vehicle_name}")
        self.vehicle_name = self.get_parameter("vehicle_name").value
        self.rate_hz = self.get_parameter("rate_hz").value
        self.look_ahead = self.get_parameter("look_ahead").value
        self.wheelbase = self.get_parameter("wheelbase").value
        self.offset = self.get_parameter("offset").value
        self.olat = self.get_parameter("origin_lat").value
        self.olon = self.get_parameter("origin_lon").value
        self.stop_dist_threshold = self.get_parameter("stop_distance").value
        self.kick_start_accel = self.get_parameter("kick_start_accel").value
        self.kick_start_duration = self.get_parameter("kick_start_duration").value
        self.kick_start_speed_threshold = self.get_parameter(
            "kick_start_speed_threshold"
        ).value

        self.desired_speed = min(
            5.0, self.get_parameter("desired_speed").value
        )  # desired speed capped at 5 m/s
        self.max_accel = min(
            2.0, self.get_parameter("max_accel").value
        )  # max acceleration capped at 2 m/s^2
        self.pid_speed = PID(
            kp=self.get_parameter("pid/kp").value,
            ki=self.get_parameter("pid/ki").value,
            kd=self.get_parameter("pid/kd").value,
            wg=self.get_parameter("pid/wg").value,
        )
        self.speed_filter = OnlineFilter(
            cutoff=self.get_parameter("filter/cutoff").value,
            fs=self.get_parameter("filter/fs").value,
            order=self.get_parameter("filter/order").value,
        )

        self.goal = 0
        self.declare_parameter("lane_width", 3)
        self.lane_width = self.get_parameter("lane_width").value
        self.declare_parameter("fx", 456.55551147)
        self.fx = self.get_parameter("fx").value
        self.declare_parameter("multi_class", True)
        self.multi_class = self.get_parameter("multi_class").value
        self.declare_parameter("fy", 456.40646362)
        self.fy = self.get_parameter("fy").value
        self.declare_parameter("cx", 382.24740601)
        self.cx = self.get_parameter("cx").value
        self.declare_parameter("cy", 242.46728516)
        self.cy = self.get_parameter("cy").value
        self.declare_parameter("img_width", 768)
        self.declare_parameter("img_height", 480)

        self._img_width = self.get_parameter("img_width").value
        self._img_height = self.get_parameter("img_height").value
        self.declare_parameter("height_of_cones_theshold", 1)  # real height is 0.75
        self.height_of_cones_theshold = self.get_parameter(
            "height_of_cones_theshold"
        ).value
        self.declare_parameter("camera_topic", "/oak/rgb/image_raw")
        self.camera_topic = self.get_parameter("camera_topic").value
        self.declare_parameter("lidar_topic", "/ouster/points")
        self.lidar_topic = self.get_parameter("lidar_topic").value
        # self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info")
        # self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.declare_parameter("depth_output_topic", "/depth_image_output_ADD")
        self.depth_output_topic = self.get_parameter("depth_output_topic").value

        # Declared for compatibility only. Depth now comes from Depth-Anything-V2
        # on the camera frame, so this topic is not subscribed.
        self.declare_parameter("depth_camera_topic", "/oak/stereo/image_raw")
        self.depth_camera_topic = self.get_parameter("depth_camera_topic").value

        self.declare_parameter("lane_fit_topic", "/lane_fit_topic_ADD")
        self.lane_fit_topic = self.get_parameter("lane_fit_topic").value

        # Lane detection and keeping are created once here, not per frame
        self._lane_keeper = LaneKeeping(
            self._img_width,
            self._img_height,
            logging.getLogger("LaneKeeping"),
        )
        self._lane_detector = LaneDetection(
            self._img_width,
            self._img_height,
            self._lane_keeper,
        )

        self._cv_bridge = CvBridge()
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Depth-Anything-V2 (vits), loaded once from local weights ---
        self._depth_model = None

        if HAS_LOCAL_DEPTH_ANYTHING:
            try:
                model_configs = {
                    "vits": {
                        "encoder": "vits",
                        "features": 64,
                        "out_channels": [48, 96, 192, 384],
                    }
                }

                self._depth_model = DepthAnythingV2(**model_configs["vits"])

                if os.path.exists(weights_path):
                    self._depth_model.load_state_dict(
                        torch.load(weights_path, map_location="cpu")
                    )
                    self._depth_model = self._depth_model.to(self._dev).eval()
                    self.get_logger().info(
                        "Loaded local Depth Anything V2 model successfully offline"
                    )
                else:
                    self.get_logger().error(
                        f"Depth checkpoint not found at: {weights_path}"
                    )
                    self._depth_model = None

            except Exception as e:
                self.get_logger().error(
                    f"Could not load local Depth Anything V2 model: {e}"
                )
                self._depth_model = None

        else:
            self.get_logger().error(
                f"Could not import depth_anything_v2 from {DEPTH_ANYTHING_DIR}"
            )

        # Latest camera-derived waypoints in vehicle frame (x forward, y left), near to far
        self.path_points_x = np.array([])
        self.path_points_y = np.array([])
        self.waypoint_pixels = []

        self.depth_pub = self.create_publisher(Image, self.depth_output_topic, 10)

        self.lane_fit_pub = self.create_publisher(Image, self.lane_fit_topic, 10)

        # Subscriptions
        self.create_subscription(NavSatFix, "/navsatfix", self.gnss_callback, 10)
        self.create_subscription(INSNavGeod, "/insnavgeod", self.ins_callback, 10)
        self.create_subscription(Bool, "/pacmod/enabled", self.enable_callback, 10)
        self.create_subscription(
            VehicleSpeedRpt, "/pacmod/vehicle_speed_rpt", self.speed_callback, 10
        )
        self.create_subscription(
            SystemRptFloat, "/pacmod/brake_rpt", self.brake_rpt_callback, 10
        )
        self.current_brake = 0.0
        self.create_subscription(
            SystemRptInt, "/pacmod/shift_rpt", self.shift_rpt_callback, 10
        )
        self.current_gear = 2

        self.create_subscription(
            Float32, "/joystick/brake_value", self.brake_callback, 10
        )
        self.pacmod_override_active = False
        self.create_subscription(
            GlobalRpt, "/pacmod/global_rpt", self.global_rpt_callback, 10
        )
        self.create_subscription(Image, self.camera_topic, self._on_camera_image, 10)
        self.create_subscription(
            PointCloud2, self.lidar_topic, self._on_lidar_points, 10
        )
        # self.create_subscription(
        #     CameraInfo, self.camera_info_topic, self._on_camera_info, 10
        # )

        # Publishers
        self.global_pub = self.create_publisher(GlobalCmd, "/pacmod/global_cmd", 10)
        self.gear_pub = self.create_publisher(SystemCmdInt, "/pacmod/shift_cmd", 10)
        self.brake_pub = self.create_publisher(SystemCmdFloat, "/pacmod/brake_cmd", 10)
        self.accel_pub = self.create_publisher(SystemCmdFloat, "/pacmod/accel_cmd", 10)
        self.turn_pub = self.create_publisher(SystemCmdInt, "/pacmod/turn_cmd", 10)
        self.steer_pub = self.create_publisher(
            PositionWithSpeed, "/pacmod/steering_cmd", 10
        )

        self.local_odom_pub = self.create_publisher(Odometry, "/gem/local_odom", 10)
        self.target_point_pub = self.create_publisher(
            PointStamped, "/pure_pursuit/target_point", 10
        )

        # Commands
        self.global_cmd = GlobalCmd(
            enable=False, clear_override=True, ignore_override=False
        )
        self.gear_cmd = SystemCmdInt(command=2)  # NEUTRAL
        self.brake_cmd = SystemCmdFloat(command=0.0)
        self.accel_cmd = SystemCmdFloat(command=0.0)
        self.turn_cmd = SystemCmdInt(command=1)  # no signal
        self.steer_cmd = PositionWithSpeed(
            angular_position=0.0, angular_velocity_limit=4.0
        )

        # Initialize
        self.object_detected = False
        self.object_detected_coordinate = None
        self.cone_detected = False
        self.cone_detected_coordinate = None

        # GPS state: used only for the start point and the finish check
        self.lat = 0.0
        self.lon = 0.0
        self.start_lat = 0.0
        self.start_lon = 0.0
        self.start_recorded = False
        self.left_start_zone = False
        self.finished = False

        self.heading = 0.0
        self.speed = 0.0
        self.gem_enable = False
        self.pacmod_enable = False
        self.brake_value = 0.0
        self.is_stopping = False
        self.stop_start_time = 0.0
        self.ramp_duration = 1.5
        self.max_stop_brake = 0.7
        self.kick_start_active = False
        self.kick_start_end_time = 0.0

        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

    def gnss_callback(self, msg):
        # Ignore invalid fixes. The first valid fix becomes the start point.
        if msg.status.status < 0 or not (
            math.isfinite(msg.latitude) and math.isfinite(msg.longitude)
        ):
            return

        self.lat = msg.latitude
        self.lon = msg.longitude

        if not self.start_recorded:
            self.start_lat = self.lat
            self.start_lon = self.lon
            self.start_recorded = True
            self.get_logger().info(
                f"Start point recorded: lat {self.start_lat:.7f}, lon {self.start_lon:.7f}"
            )

    def ins_callback(self, msg):
        self.heading = msg.heading

    def dist_to_start(self):
        """Distance in meters from the current GPS fix to the recorded start point."""
        if not self.start_recorded:
            return None
        e, n, _ = pm.geodetic2enu(
            self.lat, self.lon, 0.0, self.start_lat, self.start_lon, 0.0
        )
        return math.hypot(e, n)

    def speed_callback(self, msg):
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def global_rpt_callback(self, msg):
        self.pacmod_override_active = msg.override_active

        if self.pacmod_override_active:
            # Operator physically grabbed the steering wheel/brake/accelerator.
            self.global_cmd.enable = False
            self.get_logger().warn(
                "Manual override detected on PACMod - disengaging autonomous control",
                throttle_duration_sec=1.0,
            )
            self.gear_cmd.command = 2  # NEUTRAL
            self.gear_pub.publish(self.gear_cmd)

            self.global_pub.publish(self.global_cmd)

    def brake_callback(self, msg):
        self.brake_value = msg.data

    def brake_rpt_callback(self, msg):
        self.current_brake = msg.output

    def shift_rpt_callback(self, msg):
        self.current_gear = msg.output

    # def _on_camera_info(self, msg):
    #     intrinsic_matrix = np.array(msg)
    def pixel_to_vehicle_frame(self, u, v, depth):
        X_cam = (u - self.cx) * depth / self.fx
        # Y_cam = (v - self.cy) * depth / self.fy

        x_veh = depth
        y_veh = -X_cam

        return x_veh, y_veh

    def get_depth_at(
        self,
        depth_image,
        u,
        v,
        default_depth=2.0,
        min_depth=0.05,
        max_depth=None,
        window_size=1,
    ):

        if depth_image is None or getattr(depth_image, "ndim", 0) != 2:
            self.get_logger().warn(
                "Depth image is None or not 2D; returning default depth.",
                throttle_duration_sec=2.0,
            )
            return default_depth

        try:
            uu = int(round(u))
            vv = int(round(v))
            height, width = depth_image.shape

            if not (0 <= vv < height and 0 <= uu < width):
                return default_depth

            if window_size <= 1:
                d = float(depth_image[vv, uu])
                if not np.isfinite(d) or d < min_depth:
                    return default_depth
                if max_depth is not None and d > max_depth:
                    return default_depth
                return d

            half = window_size // 2
            v0, v1 = max(0, vv - half), min(height, vv + half + 1)
            u0, u1 = max(0, uu - half), min(width, uu + half + 1)

            patch = np.asarray(depth_image[v0:v1, u0:u1], dtype=np.float64)
            valid = patch[np.isfinite(patch) & (patch >= min_depth)]
            if max_depth is not None:
                valid = valid[valid <= max_depth]

            if valid.size == 0:
                return default_depth

            return float(np.median(valid))

        except (TypeError, ValueError, IndexError) as exc:
            self.get_logger().debug(f"Error extracting depth at ({u}, {v}): {exc}")
            return default_depth

    def desired_lane_to_waypoints(self, desired_lane, depth_map):
        """Convert the LaneKeeping desired lane (one pixel column per row) into
        vehicle-frame waypoints ordered near to far. Rows without valid depth
        are dropped instead of being placed at a default distance.

        Returns the waypoints and the matching pixel locations for visualization.
        """
        if desired_lane is None:
            self.get_logger().warn("No lane detected", throttle_duration_sec=2.0)
            return [], []

        # Scale depth lookups if the depth map resolution differs from the RGB image
        if depth_map is not None and getattr(depth_map, "ndim", 0) == 2:
            sx = depth_map.shape[1] / self._img_width
            sy = depth_map.shape[0] / self._img_height
        else:
            sx = sy = 1.0

        rows = self._lane_keeper.plot_y  # bottom (near) to top (far)
        waypoints = []
        pixels = []

        for u, v in zip(desired_lane, rows):
            if not (0 <= u < self._img_width):
                continue
            # NaN default: pixels without valid depth are skipped below
            depth = self.get_depth_at(depth_map, u * sx, v * sy, default_depth=np.nan)
            if not np.isfinite(depth):
                continue
            x, y = self.pixel_to_vehicle_frame(u, v, depth)
            waypoints.append([float(x), float(y)])
            pixels.append((int(round(u)), int(round(v))))

        return waypoints, pixels

    def _clear_waypoints(self):
        self.path_points_x = np.array([])
        self.path_points_y = np.array([])
        self.waypoint_pixels = []

    def _estimate_depth(self, image, header):
        """Run Depth-Anything-V2 on a BGR frame. Returns an HxW float32 depth map,
        or None if the model is not loaded. The map is also published on
        depth_output_topic for debugging.
        """
        if self._depth_model is None:
            return None

        depth = np.asarray(self._depth_model.infer_image(image), dtype=np.float32)
        if depth.ndim != 2 or not np.any(np.isfinite(depth)):
            self.get_logger().warn(
                "Depth-Anything-V2 returned an invalid depth map",
                throttle_duration_sec=2.0,
            )
            return None

        out_msg = self._cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        out_msg.header = header
        self.depth_pub.publish(out_msg)

        return depth

    def _on_camera_image(self, msg):
        """Front camera frame -> lane detection and Depth-Anything-V2 -> vehicle-frame waypoints."""
        self.get_logger().info(
            f"Received RGB image | Width: {msg.width}, Height: {msg.height}",
            throttle_duration_sec=2.0,
        )

        image = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        lane_det_results = self._lane_detector.lanes_detection(image)
        if lane_det_results is None:
            self._clear_waypoints()
            return

        depth_map = self._estimate_depth(image, msg.header)
        if depth_map is None:
            self._clear_waypoints()
            return

        desired_lane, _ = self._lane_keeper.lane_keeping(lane_det_results)

        waypoints, self.waypoint_pixels = self.desired_lane_to_waypoints(
            desired_lane, depth_map
        )
        if not waypoints:
            self._clear_waypoints()
            return

        wp = np.asarray(waypoints, dtype=float)
        self.path_points_x = wp[:, 0]
        self.path_points_y = wp[:, 1]

    def _on_lidar_points(self, msg):
        """Callback for 3D LiDAR point cloud data."""
        self.get_logger().info(
            f"Received PointCloud2 | Frame ID: {msg.header.frame_id}, Points: {len(msg.data)} bytes",
            throttle_duration_sec=2.0,
        )
        self._lidar_msg = msg
        self.object_detected, self.cone_detected = self.detect_objects_and_cones(5.0)

    def heading_to_yaw(self, heading):
        return np.radians(90 - heading) if heading < 270 else np.radians(450 - heading)

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # helper function for lidar obstacle detection
    def dist_3D(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )

    def front2steer(self, f_angle):
        f_angle = max(min(f_angle, 35), -35)
        angle = abs(f_angle)
        steer_angle = -0.1084 * angle**2 + 21.775 * angle
        return round(steer_angle if f_angle >= 0 else -steer_angle, 2)

    def check_joystick_enable(self):
        pygame.event.pump()
        try:
            lb = joystick.get_button(6)
            rb = joystick.get_button(7)
        except pygame.error:
            self.get_logger().warn("Joystick read failed")
            return 2
        if lb and rb:
            # enable
            self.get_logger().warn("Joystick Pressed")
            return 1
        elif lb and not rb:
            # disable
            self.get_logger().warn("Joystick Released")
            return 0
        # others
        return 2

    # function to detect if the lidar sees something too close to the car
    def detect_objects_and_cones(self, threshold=5.0):
        points = point_cloud2.read_points_list(
            self._lidar_msg, field_names=("x", "y", "z")
        )
        obj_detected = False
        points_in_zone = 0
        cone_valid = True  # Assume valid until a tall point is found
        first_point = None

        for point in points:
            # front of car, between lanes, within distance threshold
            if (
                point[0] > 0
                and -1.5 < point[1] < 1.5
                and self.dist_3D(point, (0, 0, 0)) < threshold
            ):
                points_in_zone += 1

                # General object detection (capture the first valid point)
                if not obj_detected:
                    print("First object detected at point:", point)
                    self.get_logger().info(f"First object detected at point: {point}")
                    self.object_detected_coordinate = point
                    obj_detected = True
                    first_point = point
                print("Close object detected at point:", point)
                self.get_logger().info(f"Close object detected at point: {point}")

                # Veto cone detection if ANY point in the zone is above the height threshold
                if point[2] >= self.height_of_cones_theshold:
                    cone_valid = False

        # Cone is only detected if an object is present, points exist, and NO point exceeded the height limit
        cone_detected = obj_detected and (points_in_zone > 0) and cone_valid

        if cone_detected:
            self.cone_detected_coordinate = first_point
            print("Cone verified and detected at point:", first_point)
            self.get_logger().info(
                f"Cone verified and detected at point: {first_point}"
            )
        return obj_detected, cone_detected

    def publish_local_odom(self, x, y, yaw):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.speed
        self.local_odom_pub.publish(odom)

    def publish_target_point(self, x, y):
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        # pt.header.frame_id = "map"
        pt.header.frame_id = "base_link"
        pt.point.x = x
        pt.point.y = y
        self.target_point_pub.publish(pt)

    def control_loop(self):
        joy_enable = self.check_joystick_enable()

        # curr_x, curr_y, curr_yaw = self.get_gem_state()

        # self.publish_local_odom(curr_x, curr_y, curr_yaw)

        if joy_enable == 1 and not self.pacmod_enable:
            # joystick enable when vehicle disabled
            self.global_cmd.enable = True
            self.global_cmd.clear_override = True
            self.global_pub.publish(self.global_cmd)

            self.brake_cmd.command = 0.0
            self.brake_pub.publish(self.brake_cmd)

            self.accel_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)

            self.turn_cmd.command = 3
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn(
                "Joystick enabled, Pacmod enabled, forward gear engaged"
            )

        elif joy_enable == 0 and self.pacmod_enable:
            # joystick disable when vehicle enabled
            self.global_cmd.enable = False

            self.turn_cmd.command = 1
            self.turn_pub.publish(self.turn_cmd)

            self.gear_cmd.command = 2
            self.gear_pub.publish(self.gear_cmd)

            self.global_pub.publish(self.global_cmd)
            self.get_logger().warn("Joystick Disabled, Pacmod disabled")

        elif joy_enable != 0 and self.pacmod_enable:
            # self.path_points_x = np.array(self.path_points_lon_x)
            # self.path_points_y = np.array(self.path_points_lat_y)
            # self.get_logger().info(
            #     f"Next waypoint: ({self.path_points_x[0]}, {self.path_points_y[0]})"
            # )
            # # Check distance to the FINAL waypoint in the track
            # end_x = self.path_points_x[-1]
            # end_y = self.path_points_y[-1]
            # dist_to_end = self.dist((end_x, end_y), (curr_x, curr_y))

            # # Trigger stopping sequence if close to final waypoint or already in stopping sequence
            # if dist_to_end <= self.stop_dist_threshold or self.is_stopping:
            if self.finished:
                return

            # GPS finish check: arm after leaving the start zone, then stop near start
            dist_to_start = self.dist_to_start()
            if dist_to_start is not None and not self.left_start_zone:
                if dist_to_start > START_ZONE_RADIUS:
                    self.left_start_zone = True
                    self.get_logger().info(
                        f"Left start zone ({dist_to_start:.2f}m). Finish armed."
                    )
            at_start = (
                self.left_start_zone
                and dist_to_start is not None
                and dist_to_start <= self.stop_dist_threshold
            )

            # Trigger stopping sequence at the start point or if already stopping
            if at_start or self.is_stopping:
                now_sec = self.get_clock().now().nanoseconds * 1e-9

                # First iteration entering stop state
                if not self.is_stopping:
                    self.is_stopping = True
                    self.stop_start_time = now_sec
                    self.pid_speed.reset()
                    self.get_logger().warn(
                        f"Start point reached ({dist_to_start:.2f}m). Initiating smooth braking sequence..."
                    )

                # Elapsed time since stopping triggered
                elapsed = now_sec - self.stop_start_time

                # 1. Zero accelerator command
                self.accel_cmd.command = 0.0
                self.accel_pub.publish(self.accel_cmd)

                # 2. Smoothly ramp brake from 0.0 to max over ramp_duration
                brake_target = min(
                    self.max_stop_brake,
                    self.max_stop_brake * (elapsed / self.ramp_duration),
                )
                self.brake_cmd.command = brake_target
                self.brake_pub.publish(self.brake_cmd)

                # 3. Center steering wheel during stop
                self.steer_cmd.angular_position = 0.0
                self.steer_pub.publish(self.steer_cmd)

                # 4. Keep FORWARD while moving, shift to NEUTRAL once stopped
                if self.speed < 0.05:  # Vehicle is fully stopped
                    self.gear_cmd.command = 2  # NEUTRAL
                    self.get_logger().warn(
                        "Vehicle completely stopped. Gear shifted to NEUTRAL. Finishing program."
                    )
                    self.is_stopping = False
                    self.finished = True
                else:
                    self.gear_cmd.command = 3  # Keep FORWARD until stopped
                    self.get_logger().info(
                        f"Decelerating... Speed: {self.speed:.2f} m/s, Brake: {brake_target:.2f}",
                        throttle_duration_sec=0.5,
                    )

                self.gear_pub.publish(self.gear_cmd)

                # Maintain PACMod enable so brake pressure remains active
                self.global_cmd.enable = True
                self.global_pub.publish(self.global_cmd)

                return

            # Shift to FORWARD if not already in drive (driver presses brake above 50%)
            if self.current_gear != 3:
                self.get_logger().info(
                    f"Attempting to shift to FORWARD. Current brake: {self.current_brake:.2f}. Press brake above 50% to proceed."
                )
                if self.current_brake > 0.50:
                    self.gear_cmd.command = 3  # FORWARD
                    self.gear_pub.publish(self.gear_cmd)
            # for i in range(self.wp_size):
            #     self.dist_arr[i] = self.dist(
            #         (self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y)
            #     )

            # self.goal = np.argmin(self.dist_arr)
            # ld = self.look_ahead + max(0.0, self.speed - 2.5) * 2
            # for i in range(self.goal, self.wp_size):
            #     if self.dist_arr[i] > ld:
            #         self.goal = i
            #         break
            # # TODO
            # target_x = self.path_points_x[self.goal]
            # target_y = self.path_points_y[self.goal]
            # target_yaw = self.path_points_heading[self.goal]
            # No lane this cycle: no throttle, keep PACMod enabled
            if self.path_points_x.size == 0:
                self.get_logger().warn(
                    "No lane waypoints this cycle", throttle_duration_sec=1.0
                )
                self.accel_cmd.command = 0.0
                self.accel_pub.publish(self.accel_cmd)
                self.brake_cmd.command = max(0.0, min(self.brake_value, 0.8))
                self.brake_pub.publish(self.brake_cmd)
                self.global_cmd.enable = True
                self.global_pub.publish(self.global_cmd)
                return

            # Pure pursuit in vehicle frame: the car is at the origin with yaw 0,
            # so waypoints (x forward, y left, near to far) are used directly.
            path_x = self.path_points_x
            path_y = self.path_points_y
            wp_size = len(path_x)
            dists = np.hypot(path_x, path_y)

            self.goal = int(np.argmin(dists))
            ld = self.look_ahead + max(0.0, self.speed - 2.5) * 2
            found = False
            for i in range(self.goal, wp_size):
                if dists[i] > ld:
                    self.goal = i
                    found = True
                    break

            if not found and wp_size > 0:
                self.goal = wp_size - 1

            target_x = float(path_x[self.goal])
            target_y = float(path_y[self.goal])
            self.publish_target_point(target_x, target_y)
            # alpha = math.atan2(target_y - curr_y, target_x - curr_x) - curr_yaw
            alpha = math.atan2(target_y, target_x)
            curvature = 0.0 if self.speed < 0.2 else 2.0 * math.sin(alpha) / ld
            steering_angle = math.atan(self.wheelbase * curvature)
            steering_wheel_angle = self.front2steer(math.degrees(steering_angle))

            self.steer_cmd.angular_position = math.radians(steering_wheel_angle)
            self.steer_pub.publish(self.steer_cmd)

            # Speed control
            now = self.get_clock().now().nanoseconds * 1e-9
            speed_error = self.desired_speed - self.speed
            if abs(speed_error) < 0.05:
                speed_error = 0.0
            pid_throttle_cmd = self.pid_speed.get_control(now, speed_error)
            pid_throttle_cmd = max(0.0, min(pid_throttle_cmd, self.max_accel))

            if (
                self.vehicle_name == "e2"
                and self.current_gear == 3
                and self.speed < self.kick_start_speed_threshold
            ):
                if not self.kick_start_active:
                    self.kick_start_active = True
                    self.kick_start_end_time = now + self.kick_start_duration
                    self.get_logger().info(
                        f"Kick-start: holding accel={self.kick_start_accel:.2f} for up to "
                        f"{self.kick_start_duration:.1f}s to break static friction"
                    )
                if now < self.kick_start_end_time:
                    self.get_logger().info("Waiting for kick-start to complete")
                    throttle_cmd = self.kick_start_accel
                else:
                    self.kick_start_active = False
                    throttle_cmd = pid_throttle_cmd
            else:
                self.kick_start_active = False
                throttle_cmd = pid_throttle_cmd

            self.accel_cmd.command = throttle_cmd
            self.brake_cmd.command = max(0.0, min(self.brake_value, 0.8))
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            self.global_cmd.enable = True
            self.global_pub.publish(self.global_cmd)

            self.get_logger().info(
                f"Target: ({target_x:.2f}, {target_y:.2f}), Speed: {self.speed:.2f}, "
                f"Throttle: {throttle_cmd:.2f}, Steering: {steering_wheel_angle:.2f}"
            )

    def destroy_node(self):
        self.timer.cancel()
        self.destroy_timer(self.timer)

        self.get_logger().info("Publishing final disable command...")
        self.global_cmd.enable = False

        # Publish multiple times to give the DDS discovery/transport layer a chance
        for _ in range(3):
            self.global_pub.publish(self.global_cmd)
            # Give the network stack a tiny window to flush the queue out
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info("Destroying PurePursuit node")
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)
    pure_pursuit = PurePursuit()

    try:
        while rclpy.ok() and not pure_pursuit.finished:
            rclpy.spin_once(pure_pursuit, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        # Call destroy_node before rclpy.shutdown() while context is still alive
        pure_pursuit.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
