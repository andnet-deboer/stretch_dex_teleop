#!/usr/bin/env python3

import cv2
import numpy as np
import cv2.aruco as aruco
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R_scipy

def minimum_distance_between_corners(corners):
    c0 = corners[0]
    dist0 = np.min(np.linalg.norm(corners[1:4] - c0, axis=1))
    c1 = corners[1]
    dist1 = np.min(np.linalg.norm(corners[2:4] - c1, axis=1))
    c2 = corners[2]
    dist2 = np.min(np.linalg.norm(corners[3:4] - c2, axis=1))
    return np.min(np.array([dist0, dist1, dist2]))

def axes_to_quaternion(x, y, z):
    # Reconstruct the 3x3 rotation matrix from basis vectors
    mat = np.column_stack((x, y, z))
    return R_scipy.from_matrix(mat).as_quat()

class ArucoMarker:
    def __init__(self, aruco_id, marker_info, node, show_debug_images=False):
        self.node = node
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.show_debug_images = show_debug_images
        self.aruco_id = aruco_id
        
        self.frame_id = 'camera_color_optical_frame'
        self.info = marker_info.get(str(self.aruco_id), marker_info['default'])
        self.length_of_marker_mm = self.info['length_mm']
        
        self.frame_number = None
        self.ready = False
        self.x_axis = None
        self.y_axis = None
        self.z_axis = None

    def broadcast_tf(self, name, pos, x, y, z):
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = self.frame_id
        t.child_frame_id = str(name)

        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos

        q = axes_to_quaternion(x, y, z)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def update(self, corners, frame_number, rgb_camera_info):
        self.corners = corners
        self.frame_number = frame_number
        camera_matrix = rgb_camera_info['camera_matrix']
        distortion_coefficients = rgb_camera_info['distortion_coefficients']

        points_3D = np.array([
            (-self.length_of_marker_mm / 2,  self.length_of_marker_mm / 2, 0),
            ( self.length_of_marker_mm / 2,  self.length_of_marker_mm / 2, 0),
            ( self.length_of_marker_mm / 2, -self.length_of_marker_mm / 2, 0),
            (-self.length_of_marker_mm / 2, -self.length_of_marker_mm / 2, 0),
        ])

        _, rvec, tvec = cv2.solvePnP(objectPoints=points_3D,
                                     imagePoints=self.corners,
                                     cameraMatrix=camera_matrix,
                                     distCoeffs=distortion_coefficients)                                              
        
        self.marker_position = tvec.flatten() / 1000.0
        R_mat = cv2.Rodrigues(rvec)[0]

        # ── MINIMAL SWIZZLE (Red=Fwd, Green=Left, Blue=Up) ──
        # Maps OpenCV camera frame to Stretch base frame convention
        self.x_axis = -R_mat[:3, 1] # New X (Red)   = Old -Y
        self.y_axis = -R_mat[:3, 0] # New Y (Green) = Old -X
        self.z_axis = -R_mat[:3, 2] # New Z (Blue)  = Old -Z

        self.ready = True
        self.broadcast_tf(f"tag_{self.aruco_id}", self.marker_position, self.x_axis, self.y_axis, self.z_axis)

    def get_min_dist_between_corners(self):
        return minimum_distance_between_corners(self.corners)

    def get_position_and_axes(self):
        return np.array(self.marker_position), np.array(self.x_axis), np.array(self.y_axis), np.array(self.z_axis)

class ArucoMarkerCollection:
    def __init__(self, marker_info, node, show_debug_images=False):
        self.node = node
        self.marker_info = marker_info
        # Switched to AprilTag dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.aruco_detection_parameters = aruco.DetectorParameters()
        self.aruco_detection_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        
        self.collection = {}
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_detection_parameters)
        self.frame_number = 0

    def update(self, rgb_image, rgb_camera_info):
        self.frame_number += 1
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        self.aruco_corners, self.aruco_ids, _ = self.detector.detectMarkers(gray_image)
        
        if self.aruco_ids is not None: 
            for corners, aruco_id in zip(self.aruco_corners, self.aruco_ids):
                aid = int(aruco_id)
                if aid not in self.collection:
                    self.collection[aid] = ArucoMarker(aid, self.marker_info, self.node)
                self.collection[aid].update(corners[0], self.frame_number, rgb_camera_info)

class ArucoDetector:
    def __init__(self, node, marker_info=None, show_debug_images=False):
        self.node = node
        self.marker_info = marker_info if marker_info else {}
        self.aruco_marker_collection = ArucoMarkerCollection(self.marker_info, self.node, show_debug_images)

    def update(self, rgb_image, rgb_camera_info):
        self.aruco_marker_collection.update(rgb_image, rgb_camera_info)

    def get_detected_marker_dict(self):
        out = {}
        for aid, m in self.aruco_marker_collection.collection.items():
            if m.frame_number == self.aruco_marker_collection.frame_number:
                pos, x, y, z = m.get_position_and_axes()
                out[aid] = {'pos': pos, 'x_axis': x, 'y_axis': y, 'z_axis': z, 'info': m.info}
        return out

def main(args=None):
    # This script is designed to be imported as a library by your main ROS node.
    pass

if __name__ == '__main__':
    main()