def turn_around_cone(self):
    if self.cone_detected_coordinate is None:
        return

    # 1. Get current vehicle pose in map frame
    curr_x, curr_y, curr_yaw = self.get_gem_state()

    # 2. Extract cone coordinates in vehicle frame (X = front, Y = left)
    cx_rel = self.cone_detected_coordinate[0]
    cy_rel = self.cone_detected_coordinate[1]

    # 3. Transform cone position from vehicle frame to map frame
    cone_map_x = curr_x + cx_rel * math.cos(curr_yaw) - cy_rel * math.sin(curr_yaw)
    cone_map_y = curr_y + cx_rel * math.sin(curr_yaw) + cy_rel * math.cos(curr_yaw)

    # 4. Find the waypoint closest to the cone in the map frame
    cone_dist_arr = [
        math.hypot(
            self.path_points_x[i] - cone_map_x, self.path_points_y[i] - cone_map_y
        )
        for i in range(len(self.path_points_x))
    ]
    closest_wp_idx = np.argmin(cone_dist_arr)

    # 5. Define detour parameters around that waypoint index
    num_waypoints_to_shift = 20  # Number of waypoints affected by the bypass curve
    bypass_offset = 1.5  # Distance in meters to shift away from the cone

    # Determine shift direction: if cone is to the left in vehicle frame (cy_rel > 0), shift right (-1); else shift left (+1)
    shift_sign = -1.0 if cy_rel > 0 else 1.0

    start_idx = max(0, closest_wp_idx - 5)  # Start slightly before the cone
    end_idx = min(start_idx + num_waypoints_to_shift, len(self.path_points_x))

    # 6. Apply a smooth lateral shift (using a sine bell curve) to upcoming waypoints
    for i in range(start_idx, end_idx):
        progress = (i - start_idx) / max(1, (end_idx - start_idx))
        weight = math.sin(progress * math.pi)  # Peaks in the middle of the detour

        # Compute normal vector perpendicular to the waypoint heading
        yaw_rad = self.heading_to_yaw(self.path_points_heading[i])
        nx = -math.sin(yaw_rad)  # Lateral X component (left normal)
        ny = math.cos(yaw_rad)  # Lateral Y component (left normal)

        # Shift the waypoint coordinates laterally in the map frame
        self.path_points_x[i] += shift_sign * bypass_offset * weight * nx
        self.path_points_y[i] += shift_sign * bypass_offset * weight * ny

    self.get_logger().warn(
        f"Cone avoided! Shifted waypoints around cone at map coords ({cone_map_x:.2f}, {cone_map_y:.2f})"
    )

    # Clear the coordinate so the car doesn't keep modifying the path on every tick
    self.cone_detected_coordinate = None
