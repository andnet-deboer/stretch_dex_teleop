#!/usr/bin/env python3
"""
UMI Cube Tracker - Multi-Tag Fusion with Face-Specific Rotations
Properly handles that each cube face has a different orientation!
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


# ══════════════════════════════════════════════════════════════════════════════
# CUBE FACE ORIENTATIONS (relative to cube center)
# ══════════════════════════════════════════════════════════════════════════════

# Based on your bundle YAML:
# Tag 5 (top):    +Z face, no rotation (qw=1.0)
# Tag 1 (front):  -Y face, 90° around X (qx=0.7071, qw=0.7071)
# Tag 2 (bottom): -Z face, 180° around X (qx=1.0, qw=0.0)
# Tag 3 (back):   +Y face, -90° around X (qx=-0.7071, qw=0.7071)
# Tag 4 (left):   -X face, -90° around Y (qy=-0.7071, qw=0.7071)

TAG_TO_CUBE_ROTATIONS = {
    5: [0.0, 0.0, 0.0, 1.0],           # Top: identity
    1: [0.7071, 0.0, 0.0, 0.7071],     # Front: 90° X
    2: [1.0, 0.0, 0.0, 0.0],           # Bottom: 180° X
    3: [-0.7071, 0.0, 0.0, 0.7071],    # Back: -90° X
    4: [0.0, -0.7071, 0.0, 0.7071],    # Left: -90° Y
}


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def axes_to_quaternion(x, y, z):
    """Convert three orthonormal axes to quaternion."""
    mat = np.column_stack((x, y, z))
    return R_scipy.from_matrix(mat).as_quat()


def get_cube_pose_from_tag(tag_id, tag_pos, tag_x, tag_y, tag_z, trans_offset):
    """
    Calculate cube center pose from a single tag detection.
    
    Args:
        tag_id: AprilTag ID (determines face orientation)
        tag_pos: Tag position [x, y, z]
        tag_x, tag_y, tag_z: Tag axes
        trans_offset: Translation from YAML [dx, dy, dz] in tag frame
    
    Returns:
        {'pos': [x,y,z], 'x_axis': [...], 'y_axis': [...], 'z_axis': [...]}
    """
    # Step 1: Calculate cube center position
    cube_pos = tag_pos + (trans_offset[0] * tag_x) + (trans_offset[1] * tag_y) + (trans_offset[2] * tag_z)
    
    # Step 2: Transform tag orientation to cube orientation
    # Tag rotation in world frame
    R_tag_in_world = np.column_stack((tag_x, tag_y, tag_z))
    
    # Rotation from tag frame to cube frame (face-specific)
    quat_tag_to_cube = TAG_TO_CUBE_ROTATIONS.get(tag_id, [0, 0, 0, 1])
    R_tag_to_cube = R_scipy.from_quat(quat_tag_to_cube).as_matrix()
    
    # Cube orientation in world = Tag_in_world * Tag_to_cube
    R_cube_in_world = R_tag_in_world @ R_tag_to_cube
    
    cube_x = R_cube_in_world[:, 0]
    cube_y = R_cube_in_world[:, 1]
    cube_z = R_cube_in_world[:, 2]
    
    return {
        'pos': cube_pos,
        'x_axis': cube_x,
        'y_axis': cube_y,
        'z_axis': cube_z
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUSION ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_tag_weight(corners):
    """Weight by area in image plane."""
    pts = corners.reshape(4, 2)
    area = cv2.contourArea(pts)
    weight = np.clip(area / 1000.0, 0.1, 1.0)
    return weight


def average_quaternions(quaternions, weights):
    """Weighted quaternion average (Markley method)."""
    Q = np.zeros((4, 4))
    for q, w in zip(quaternions, weights):
        q = np.array(q) / np.linalg.norm(q)
        Q += w * np.outer(q, q)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    return eigenvectors[:, -1]


def slerp(q0, q1, t):
    """Spherical linear interpolation."""
    rotations = R_scipy.from_quat([q0, q1])
    slerp_obj = Slerp([0, 1], rotations)
    return slerp_obj([t]).as_quat()[0]


# ══════════════════════════════════════════════════════════════════════════════
# APRILTAG MARKER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AprilTagMarker:
    def __init__(self, tag_id, marker_info):
        self.tag_id = tag_id
        self.info = marker_info.get(str(tag_id), marker_info.get('default', {'length_mm': 64.0}))
        self.length_mm = self.info['length_mm']
        self.pos = None
        self.axes = [None, None, None]
    
    def update(self, corners, camera_matrix, dist_coeffs):
        half = self.length_mm / 2.0
        points_3D = np.array([
            (-half,  half, 0),
            ( half,  half, 0),
            ( half, -half, 0),
            (-half, -half, 0),
        ])
        _, rvec, tvec = cv2.solvePnP(points_3D, corners, camera_matrix, dist_coeffs)
        self.pos = tvec.flatten() / 1000.0
        R_mat = cv2.Rodrigues(rvec)[0]
        # Coordinate swizzle
        self.axes = [-R_mat[:3, 1], -R_mat[:3, 0], -R_mat[:3, 2]]


# ══════════════════════════════════════════════════════════════════════════════
# ROS 2 NODE
# ══════════════════════════════════════════════════════════════════════════════

class UmiDetectorNode(Node):
    def __init__(self):
        super().__init__('umi_detector_node')
        
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Load YAML
        try:
            with open('teleop_april_marker_info_86mm.yaml', 'r') as f:
                self.marker_info = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().error("YAML not found")
            self.marker_info = {}
        
        # Robust detector config
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 100
        params.cornerRefinementMinAccuracy = 0.01
        params.minMarkerPerimeterRate = 0.05
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        self.detector = aruco.ArucoDetector(self.dictionary, params)
        
        # State
        self.collection = {}
        self.camera_info_dict = None
        self.prev_cube_pose = None
        self.smoothing_alpha = 0.3
        
        # Subscriptions
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)
    
    def info_cb(self, msg):
        self.camera_info_dict = {
            'camera_matrix': np.array(msg.k).reshape((3, 3)),
            'distortion_coefficients': np.array(msg.d)
        }
    
    def image_cb(self, msg):
        if self.camera_info_dict is None:
            return
        
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        if ids is None:
            return
        
        # Detect all tags and get cube pose candidates
        cube_candidates = []
        
        for c, aid in zip(corners, ids.flatten()):
            aid = int(aid)
            
            if aid not in self.collection:
                self.collection[aid] = AprilTagMarker(aid, self.marker_info)
            
            marker = self.collection[aid]
            marker.update(c[0], self.camera_info_dict['camera_matrix'], 
                         self.camera_info_dict['distortion_coefficients'])
            
            # Broadcast individual tag for debugging
            self.broadcast_frame(f"tag_{aid}", marker.pos, marker.axes[0], marker.axes[1], marker.axes[2])
            
            # Get translation offset from YAML
            frames = marker.info.get('frames', {})
            if 'umi_cube' in frames:
                trans = frames['umi_cube']['trans']
                
                # Calculate cube pose from this tag (WITH ROTATION)
                cube_pose = get_cube_pose_from_tag(
                    aid,
                    marker.pos,
                    marker.axes[0],
                    marker.axes[1],
                    marker.axes[2],
                    trans
                )
                
                cube_candidates.append({
                    'pos': cube_pose['pos'],
                    'x_axis': cube_pose['x_axis'],
                    'y_axis': cube_pose['y_axis'],
                    'z_axis': cube_pose['z_axis'],
                    'corners': c[0],
                    'tag_id': aid
                })
        
        # Fuse multiple cube estimates
        if cube_candidates:
            fused_cube = self._fuse_cube_centers(cube_candidates)
            
            if fused_cube is not None:
                smoothed_cube = self._smooth_pose_temporal(fused_cube)
                self.broadcast_frame_quat('umi_cube', smoothed_cube['pos'], smoothed_cube['quat'])
    
    def _fuse_cube_centers(self, candidates):
        if len(candidates) == 0:
            return None
        
        if len(candidates) == 1:
            c = candidates[0]
            quat = axes_to_quaternion(c['x_axis'], c['y_axis'], c['z_axis'])
            return {'pos': c['pos'], 'quat': quat}
        
        # Weighted fusion
        weights = []
        positions = []
        quaternions = []
        
        for c in candidates:
            w = calculate_tag_weight(c['corners'])
            weights.append(w)
            positions.append(c['pos'])
            quat = axes_to_quaternion(c['x_axis'], c['y_axis'], c['z_axis'])
            quaternions.append(quat)
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        fused_pos = np.average(positions, axis=0, weights=weights)
        fused_quat = average_quaternions(quaternions, weights)
        
        return {'pos': fused_pos, 'quat': fused_quat}
    
    def _smooth_pose_temporal(self, current_pose):
        if self.prev_cube_pose is None:
            self.prev_cube_pose = current_pose
            return current_pose
        
        alpha = self.smoothing_alpha
        smoothed_pos = alpha * current_pose['pos'] + (1 - alpha) * self.prev_cube_pose['pos']
        smoothed_quat = slerp(self.prev_cube_pose['quat'], current_pose['quat'], alpha)
        
        smoothed_pose = {'pos': smoothed_pos, 'quat': smoothed_quat}
        self.prev_cube_pose = smoothed_pose
        return smoothed_pose
    
    def broadcast_frame(self, name, pos, x_axis, y_axis, z_axis):
        quat = axes_to_quaternion(x_axis, y_axis, z_axis)
        self.broadcast_frame_quat(name, pos, quat)
    
    def broadcast_frame_quat(self, name, pos, quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_color_optical_frame'
        t.child_frame_id = str(name)
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = UmiDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()