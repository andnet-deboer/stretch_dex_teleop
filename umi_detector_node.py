#!/usr/bin/env python3
"""
UMI Cube Tracker - DYNAMIC YAML VERSION
Reads both translation and rotation from the YAML file.
"""

import cv2
import numpy as np
import cv2.aruco as aruco 
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R_scipy, Slerp

def axes_to_quaternion(x, y, z):
    mat = np.column_stack((x, y, z))
    return R_scipy.from_matrix(mat).as_quat()

def get_cube_pose_from_tag(tag_pos, tag_x, tag_y, tag_z, trans_offset, quat_tag_to_cube):
    """Calculate cube pose using YAML-provided rotation and translation."""
    # 1. Position Calculation
    cube_pos = tag_pos + (trans_offset[0] * tag_x) + (trans_offset[1] * tag_y) + (trans_offset[2] * tag_z)
    
    # 2. Rotation Calculation
    R_tag_in_world = np.column_stack((tag_x, tag_y, tag_z))
    R_tag_to_cube = R_scipy.from_quat(quat_tag_to_cube).as_matrix()
    
    # Cube_in_world = Tag_in_world * INVERSE(Tag_to_cube)
    R_cube_in_world = R_tag_in_world @ R_tag_to_cube.T
    
    return {
        'pos': cube_pos,
        'quat': R_scipy.from_matrix(R_cube_in_world).as_quat()
    }

def calculate_tag_weight(corners):
    pts = corners.reshape(4, 2)
    area = cv2.contourArea(pts)
    return np.clip(area / 1000.0, 0.1, 1.0)

def average_quaternions(quaternions, weights):
    if len(quaternions) == 0: return np.array([0, 0, 0, 1])
    quaternions = [np.array(q) / np.linalg.norm(q) for q in quaternions]
    q_ref = quaternions[0]
    aligned_quats = []
    for q in quaternions:
        dot = np.dot(q_ref, q)
        aligned_quats.append(q if dot >= 0 else -q)
    
    Q = np.zeros((4, 4))
    for q, w in zip(aligned_quats, weights):
        Q += w * np.outer(q, q)
    return np.linalg.eigh(Q)[1][:, -1]

class AprilTagMarker:
    def __init__(self, tag_id, marker_info):
        self.tag_id = tag_id
        # Look for specific ID in YAML, fallback to default
        self.info = marker_info.get(str(tag_id), {'length_mm': 64.0, 'frames': {}})
        self.length_mm = self.info.get('length_mm', 64.0)
        self.pos = None
        self.axes = [None, None, None]
    
    def update(self, corners, camera_matrix, dist_coeffs):
        half = self.length_mm / 2.0
        points_3D = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        _, rvec, tvec = cv2.solvePnP(points_3D, corners, camera_matrix, dist_coeffs)
        self.pos = tvec.flatten() / 1000.0
        R_mat = cv2.Rodrigues(rvec)[0]
        # X, Y, Z axes for the tag
        self.axes = [-R_mat[:3, 1], -R_mat[:3, 0], -R_mat[:3, 2]]

class UmiDetectorNode(Node):
    def __init__(self):
        super().__init__('umi_detector_node')
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Load YAML
        try:
            with open('teleop_april_marker_info_86mm.yaml', 'r') as f:
                self.marker_info = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML: {e}")
            self.marker_info = {}

        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.detector = aruco.ArucoDetector(self.dictionary, aruco.DetectorParameters())
        
        self.collection = {}
        self.camera_info_dict = None
        self.prev_cube_pose = None
        
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)

    def info_cb(self, msg):
        self.camera_info_dict = {'camera_matrix': np.array(msg.k).reshape((3, 3)), 'distortion_coefficients': np.array(msg.d)}

    def image_cb(self, msg):
        if self.camera_info_dict is None: return
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))
        
        if ids is None: return
        
        cube_candidates = []
        for c, aid in zip(corners, ids.flatten()):
            aid = int(aid)
            if aid not in self.collection:
                self.collection[aid] = AprilTagMarker(aid, self.marker_info)
            
            marker = self.collection[aid]
            marker.update(c[0], self.camera_info_dict['camera_matrix'], self.camera_info_dict['distortion_coefficients'])
            
            # Use YAML for center and rotation
            frames = marker.info.get('frames', {})
            if 'umi_cube' in frames:
                config = frames['umi_cube']
                trans = config.get('trans', [0.0, 0.0, 0.0])
                quat = config.get('quat', [0.0, 0.0, 0.0, 1.0])
                
                res = get_cube_pose_from_tag(marker.pos, marker.axes[0], marker.axes[1], marker.axes[2], trans, quat)
                cube_candidates.append({'pos': res['pos'], 'quat': res['quat'], 'weight': calculate_tag_weight(c[0])})

        if cube_candidates:
            weights = [c['weight'] for c in cube_candidates]
            norm_weights = np.array(weights) / sum(weights)
            
            fused_pos = np.average([c['pos'] for c in cube_candidates], axis=0, weights=norm_weights)
            fused_quat = average_quaternions([c['quat'] for c in cube_candidates], norm_weights)
            
            # Create a -90 degree rotation around X
            r_90_cw_x = R_scipy.from_euler('x', 180, degrees=True)
            # Combine current rotation with the offset
            current_r = R_scipy.from_quat(fused_quat)
            fused_quat = (current_r * r_90_cw_x).as_quat()
            # --------------------------------------------------
            # Simple Smoothing
            if self.prev_cube_pose is not None:
                fused_pos = 0.7 * fused_pos + 0.3 * self.prev_cube_pose['pos']
                # Slerp for rotation
                rots = R_scipy.from_quat([self.prev_cube_pose['quat'], fused_quat])
                fused_quat = Slerp([0, 1], rots)([0.7]).as_quat()[0]
            
            self.prev_cube_pose = {'pos': fused_pos, 'quat': fused_quat}
            self.broadcast_frame_quat('umi_cube', fused_pos, fused_quat)

    def broadcast_frame_quat(self, name, pos, quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_color_optical_frame'
        t.child_frame_id = name
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, pos)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, quat)
        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = UmiDetectorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()