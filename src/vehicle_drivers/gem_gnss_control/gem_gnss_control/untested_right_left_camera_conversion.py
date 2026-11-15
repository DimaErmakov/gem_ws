import numpy as np
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R_scipy

# --- Inside your node's __init__ ---
self.tf_buffer = Buffer()
self.tf_listener = TransformListener(self.tf_buffer, self)


# Helper function to look up individual TF transforms
def get_transform_matrices(self, target_frame, source_frame):
    """
    Looks up the rigid body transform from source_frame to target_frame
    and returns a 3x3 Rotation matrix and 3x1 Translation vector.
    """
    try:
        trans = self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time()
        )

        # Translation
        t = trans.transform.translation
        T = np.array([t.x, t.y, t.z], dtype=np.float32).reshape(3, 1)

        # Rotation Matrix from Quaternion
        q = trans.transform.rotation
        R = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)

        return R, T
    except Exception as e:
        self.get_logger().warn(
            f"Could not get transform from {source_frame} to {target_frame}: {e}"
        )
        return None, None


# --- Generalized Pixel-to-Vehicle Converter ---
def camera_pixel_to_vehicle_frame(self, u, v, depth, R_cam2veh, T_cam2veh):
    """
    Transforms a pixel from any camera directly into the vehicle coordinate frame
    using a precomputed camera-to-vehicle transformation.

    :param u, v: Pixel coordinates in the camera image
    :param depth: Estimated depth of the pixel
    :param R_cam2veh: 3x3 Rotation matrix from camera to vehicle frame
    :param T_cam2veh: 3x1 Translation vector from camera to vehicle frame
    :return: (x_veh, y_veh) in the vehicle coordinate frame
    """
    if depth <= 0.0 or np.isnan(depth):
        return None, None

    # 1. Back-project pixel to 3D point in the camera's local frame
    X_cam = (u - self.cx) * depth / self.fx
    Y_cam = (v - self.cy) * depth / self.fy
    Z_cam = depth
    P_cam = np.array([X_cam, Y_cam, Z_cam], dtype=np.float32)

    # 2. Transform directly to the vehicle frame: P_veh = R * P_cam + T
    P_veh = np.dot(R_cam2veh, P_cam) + T_cam2veh.reshape(3)

    # 3. Extract standard ROS vehicle coordinates (X = forward, Y = left)
    x_veh = P_veh[0]
    y_veh = P_veh[1]

    return x_veh, y_veh


# Choose your target camera frame dynamically ('right_camera_link' or 'left_camera_link')
camera_frame = "right_camera_link"  # or 'left_camera_link'

# 1. Look up individual links from TF
R_cam2front, T_cam2front = self.get_transform_matrices(
    "front_camera_link", camera_frame
)
R_front2veh, T_front2veh = self.get_transform_matrices("base_link", "front_camera_link")

# 2. Precompute the unified transform once (Base_link <- Camera)
# P_veh = R_front2veh * (R_cam2front * P_cam + T_cam2front) + T_front2veh
#       = (R_front2veh @ R_cam2front) * P_cam + (R_front2veh @ T_cam2front + T_front2veh)
if R_cam2front is not None and R_front2veh is not None:
    R_cam2veh = np.dot(R_front2veh, R_cam2front)
    T_cam2veh = np.dot(R_front2veh, T_cam2front) + T_front2veh

    # 3. Process your pixels smoothly in your callback loop:
    u, v = 320, 240  # example pixel coordinates
    depth = self.get_depth_at(depth_image, u, v)

    x_veh, y_veh = self.camera_pixel_to_vehicle_frame(u, v, depth, R_cam2veh, T_cam2veh)


def control_loop(self):
    # ... [joystick & enable checks] ...

    # 1. Run your cone & object checks in the LiDAR callback / update step
    self.object_detected, self.cone_detected = self.detect_objects_and_cones(5.0)

    # 2. If a cone is flagged, modify the path waypoints before pure pursuit targets them
    if self.cone_detected:
        self.turn_around_cone()

    # 3. Continue with standard Pure Pursuit goal tracking ...
    for i in range(self.wp_size):
        self.dist_arr[i] = self.dist(
            (self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y)
        )
    # ...
