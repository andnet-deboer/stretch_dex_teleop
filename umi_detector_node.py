#!/usr/bin/env python3

import cv2
import numpy as np
import cv2.AprilTag as AprilTag
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R_scipy

# Helper Functions

def axes_to_quaternion(x, y, z):
    mat = np.column_stack((x, y, z))
    return R_scipy.from_matrix(mat).as_quat()

# Core Detector Logic

class AprilTagMarker:
    def __init__(self, AprilTag_id, marker_info, node):
        self.node = node
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.AprilTag_id = AprilTag_id
        self.frame_id = 'camera_color_optical_frame'
        
        # Pull length from YAML or default to 64mm
        self.info = marker_info.get(str(self.AprilTag_id), marker_info.get('default', {'length_mm': 64.0}))
        self.length_of_marker_mm = self.info['length_mm']

    def update(self, corners, camera_matrix, dist_coeffs):
        # 3D points of a flat marker
        points_3D = np.array([
            (-self.length_of_marker_mm/2,  self.length_of_marker_mm/2, 0),
            ( self.length_of_marker_mm/2,  self.length_of_marker_mm/2, 0),
            ( self.length_of_marker_mm/2, -self.length_of_marker_mm/2, 0),
            (-self.length_of_marker_mm/2, -self.length_of_marker_mm/2, 0),
        ])

        _, rvec, tvec = cv2.solvePnP(points_3D, corners, camera_matrix, dist_coeffs)                                              
        
        pos = tvec.flatten() / 1000.0 # Convert mm to meters
        R_mat = cv2.Rodrigues(rvec)[0]

        # ── MINIMAL SWIZZLE (Red Forward, Green Left, Blue Up) ──
        # Mapping OpenCV to Stretch 3 Gripper Frame
        x_axis = -R_mat[:3, 1] # New X = Old -Y
        y_axis = -R_mat[:3, 0] # New Y = Old -X
        z_axis = -R_mat[:3, 2] # New Z = Old -Z

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = self.frame_id
        t.child_frame_id = f"tag_{self.AprilTag_id}"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos
        q = axes_to_quaternion(x_axis, y_axis, z_axis)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_broadcaster.sendTransform(t)

# ROS 2 Node Wrapper

class UmiDetectorNode(Node):
    def __init__(self):
        super().__init__('umi_detector_node')
        self.bridge = CvBridge()
        
        # Load marker info from YAML in same folder
        try:
            with open('teleop_AprilTag_marker_info_56mm.yaml', 'r') as f:
                marker_info = yaml.safe_load(f)
        except:
            self.get_logger().warn("YAML not found, using default 64mm size")
            marker_info = {}

        # Dictionary set to AprilTag 36h11
        self.AprilTag_dict = AprilTag.getPredefinedDictionary(AprilTag.DICT_APRILTAG_36h11)
        self.detector = AprilTag.AprilTagDetector(self.AprilTag_dict, AprilTag.DetectorParameters())
        self.marker_info = marker_info
        self.collection = {}
        self.camera_info_dict = None

        # Change these if your camera topics are different (ros2 topic list)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)

    def info_cb(self, msg):
        self.camera_info_dict = {
            'camera_matrix': np.array(msg.k).reshape((3, 3)),
            'distortion_coefficients': np.array(msg.d)
        }

    def image_cb(self, msg):
        if self.camera_info_dict is None: return
        
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            for c, aid in zip(corners, ids.flatten()):
                aid = int(aid)
                if aid not in self.collection:
                    self.collection[aid] = AprilTagMarker(aid, self.marker_info, self)
                self.collection[aid].update(c[0], self.camera_info_dict['camera_matrix'], 
                                            self.camera_info_dict['distortion_coefficients'])

def main():
    rclpy.init()
    node = UmiDetectorNode()
    print("Node spinning. Looking for AprilTags...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()