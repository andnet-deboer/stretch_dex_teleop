#!/usr/bin/env python3
"""
UMI Cube Tracker - DEBUG VERSION with extensive logging
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


# Face rotations (from bundle YAML)
TAG_TO_CUBE_ROTATIONS = {
    5: [0.0, 0.0, 0.0, 1.0],           # Top
    1: [0.7071, 0.0, 0.0, 0.7071],     # Front
    2: [1.0, 0.0, 0.0, 0.0],           # Bottom
    3: [-0.7071, 0.0, 0.0, 0.7071],    # Back
    4: [0.0, -0.7071, 0.0, 0.7071],    # Left
}


def axes_to_quaternion(x, y, z):
    mat = np.column_stack((x, y, z))
    return R_scipy.from_matrix(mat).as_quat()


def get_cube_pose_from_tag(tag_id, tag_pos, tag_x, tag_y, tag_z, trans_offset):
    """Calculate cube pose from tag WITH rotation correction."""
    # Position
    cube_pos = tag_pos + (trans_offset[0] * tag_x) + (trans_offset[1] * tag_y) + (trans_offset[2] * tag_z)
    
    # Rotation
    R_tag_in_world = np.column_stack((tag_x, tag_y, tag_z))
    quat_tag_to_cube = TAG_TO_CUBE_ROTATIONS.get(tag_id, [0, 0, 0, 1])
    R_tag_to_cube = R_scipy.from_quat(quat_tag_to_cube).as_matrix()
    R_cube_in_world = R_tag_in_world @ R_tag_to_cube
    
    return {
        'pos': cube_pos,
        'x_axis': R_cube_in_world[:, 0],
        'y_axis': R_cube_in_world[:, 1],
        'z_axis': R_cube_in_world[:, 2]
    }


def calculate_tag_weight(corners):
    pts = corners.reshape(4, 2)
    area = cv2.contourArea(pts)
    weight = np.clip(area / 1000.0, 0.1, 1.0)
    return weight


def average_quaternions(quaternions, weights):
    """Weighted quaternion average with sign correction."""
    if len(quaternions) == 0:
        return np.array([0, 0, 0, 1])
    
    # Normalize all quaternions
    quaternions = [np.array(q) / np.linalg.norm(q) for q in quaternions]
    
    # Align to same hemisphere (fix double-cover)
    q_ref = quaternions[0]
    aligned_quats = [q_ref]
    
    print(f"  QUATERNION ALIGNMENT:")
    for i, q in enumerate(quaternions[1:], 1):
        dot = np.dot(q_ref, q)
        if dot < 0:
            print(f"    Quat {i}: dot={dot:.4f} → FLIPPING SIGN")
            aligned_quats.append(-q)
        else:
            print(f"    Quat {i}: dot={dot:.4f} → OK")
            aligned_quats.append(q)
    
    # Markley averaging
    Q = np.zeros((4, 4))
    for q, w in zip(aligned_quats, weights):
        Q += w * np.outer(q, q)
    
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    print(f"  Eigenvalues: {eigenvalues}")
    return eigenvectors[:, -1]


def slerp(q0, q1, t):
    rotations = R_scipy.from_quat([q0, q1])
    slerp_obj = Slerp([0, 1], rotations)
    return slerp_obj([t]).as_quat()[0]


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
        self.axes = [-R_mat[:3, 1], -R_mat[:3, 0], -R_mat[:3, 2]]


class UmiDetectorNode(Node):
    def __init__(self):
        super().__init__('umi_detector_node')
        
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        try:
            with open('teleop_april_marker_info_86mm.yaml', 'r') as f:
                self.marker_info = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().error("YAML not found")
            self.marker_info = {}
        
        # Detector config
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
        
        self.collection = {}
        self.camera_info_dict = None
        self.prev_cube_pose = None
        self.smoothing_alpha = 0.7
        self.enable_smoothing = True
        self.frame_count = 0
        
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
        
        self.frame_count += 1
        
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        if ids is None:
            return
        
        print(f"\n{'#'*100}")
        print(f"FRAME {self.frame_count}: Detected {len(ids)} tags: {ids.flatten().tolist()}")
        print(f"{'#'*100}")
        
        cube_candidates = []
        
        for c, aid in zip(corners, ids.flatten()):
            aid = int(aid)
            
            if aid not in self.collection:
                self.collection[aid] = AprilTagMarker(aid, self.marker_info)
            
            marker = self.collection[aid]
            marker.update(c[0], self.camera_info_dict['camera_matrix'], 
                         self.camera_info_dict['distortion_coefficients'])
            
            self.broadcast_frame(f"tag_{aid}", marker.pos, marker.axes[0], marker.axes[1], marker.axes[2])
            
            # Get cube pose from this tag
            frames = marker.info.get('frames', {})
            if 'umi_cube' in frames:
                trans = frames['umi_cube']['trans']
                
                cube_pose = get_cube_pose_from_tag(
                    aid, marker.pos, marker.axes[0], marker.axes[1], marker.axes[2], trans
                )
                
                print(f"\nTag {aid} → Cube estimate:")
                print(f"  Tag pos: [{marker.pos[0]:.4f}, {marker.pos[1]:.4f}, {marker.pos[2]:.4f}]")
                print(f"  Cube pos: [{cube_pose['pos'][0]:.4f}, {cube_pose['pos'][1]:.4f}, {cube_pose['pos'][2]:.4f}]")
                
                cube_candidates.append({
                    'pos': cube_pose['pos'],
                    'x_axis': cube_pose['x_axis'],
                    'y_axis': cube_pose['y_axis'],
                    'z_axis': cube_pose['z_axis'],
                    'corners': c[0],
                    'tag_id': aid
                })
        
        if cube_candidates:
            fused_cube = self._fuse_cube_centers(cube_candidates)
            
            if fused_cube is not None:
                if self.enable_smoothing:
                    smoothed_cube = self._smooth_pose_temporal(fused_cube)
                else:
                    smoothed_cube = fused_cube
                    
                self.broadcast_frame_quat('umi_cube', smoothed_cube['pos'], smoothed_cube['quat'])
    
    def _fuse_cube_centers(self, candidates):
        print(f"\n{'='*90}")
        print(f"FUSION: {len(candidates)} candidates")
        print(f"{'='*90}")
        
        if len(candidates) == 1:
            c = candidates[0]
            quat = axes_to_quaternion(c['x_axis'], c['y_axis'], c['z_axis'])
            print(f"SINGLE TAG {c['tag_id']} - No fusion needed")
            print(f"  Quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            return {'pos': c['pos'], 'quat': quat}
        
        # Multiple tags
        weights = []
        positions = []
        quaternions = []
        
        print(f"\nRAW TAG DATA:")
        for c in candidates:
            w = calculate_tag_weight(c['corners'])
            weights.append(w)
            positions.append(c['pos'])
            quat = axes_to_quaternion(c['x_axis'], c['y_axis'], c['z_axis'])
            quaternions.append(quat)
            
            euler = R_scipy.from_quat(quat).as_euler('xyz', degrees=True)
            print(f"  Tag {c['tag_id']}:")
            print(f"    Weight: {w:.4f}")
            print(f"    Position: [{c['pos'][0]:.4f}, {c['pos'][1]:.4f}, {c['pos'][2]:.4f}]")
            print(f"    Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            print(f"    Euler (XYZ°): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
        
        # Normalize weights
        weights = np.array(weights)
        print(f"\nWeights (raw): {weights}")
        weights /= weights.sum()
        print(f"Weights (normalized): {weights}")
        
        # Fuse position
        fused_pos = np.average(positions, axis=0, weights=weights)
        print(f"\nFUSED POSITION: [{fused_pos[0]:.4f}, {fused_pos[1]:.4f}, {fused_pos[2]:.4f}]")
        
        # Check quaternion dot products
        print(f"\nQUATERNION DOT PRODUCTS:")
        for i in range(len(quaternions)):
            for j in range(i+1, len(quaternions)):
                dot = np.dot(quaternions[i], quaternions[j])
                status = "❌ OPPOSITE" if dot < 0 else "✓ SAME"
                print(f"  Tag {candidates[i]['tag_id']} · Tag {candidates[j]['tag_id']}: {dot:.4f} {status}")
        
        # Fuse rotation
        fused_quat = average_quaternions(quaternions, weights)
        euler_fused = R_scipy.from_quat(fused_quat).as_euler('xyz', degrees=True)
        
        print(f"\nFUSED QUATERNION: [{fused_quat[0]:.4f}, {fused_quat[1]:.4f}, {fused_quat[2]:.4f}, {fused_quat[3]:.4f}]")
        print(f"FUSED EULER (XYZ°): [{euler_fused[0]:.2f}, {euler_fused[1]:.2f}, {euler_fused[2]:.2f}]")
        print(f"{'='*90}\n")
        
        return {'pos': fused_pos, 'quat': fused_quat}
    
    def _smooth_pose_temporal(self, current_pose):
        if self.prev_cube_pose is None:
            self.prev_cube_pose = current_pose
            print(f"SMOOTHING: First frame - no smoothing applied")
            return current_pose
        
        alpha = self.smoothing_alpha
        
        print(f"\nSMOOTHING (alpha={alpha}):")
        print(f"  Previous quat: {self.prev_cube_pose['quat']}")
        print(f"  Current quat:  {current_pose['quat']}")
        
        smoothed_pos = alpha * current_pose['pos'] + (1 - alpha) * self.prev_cube_pose['pos']
        smoothed_quat = slerp(self.prev_cube_pose['quat'], current_pose['quat'], alpha)
        
        print(f"  Smoothed quat: {smoothed_quat}")
        
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