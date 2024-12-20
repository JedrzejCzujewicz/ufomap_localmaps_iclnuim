#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import os
from geometry_msgs.msg import TransformStamped, Pose
import tf2_ros
import tf.transformations as tf_trans
import struct

class TURTLEPointCloudVisualizer:
    def __init__(self):
        rospy.init_node('turtlebot_pointcloud_visualizer')
        
        # Parametry kamery (typowe dla TUM RGB-D datasets)
        self.K = np.array([[525.0, 0, 319.5],
                          [0, 525.0, 239.5],
                          [0, 0, 1]])
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Publisher dla chmury punktów
        self.cloud_pub = rospy.Publisher('/camera/point_cloud', PointCloud2, queue_size=1)
        
        # Publisher dla pozycji robota
        self.robot_pose_pub = rospy.Publisher('/robot_pose', Pose, queue_size=1)
        
        # Ścieżki do katalogów
        self.dataset_path = "/root/catkin_ws/src/tum_visualizer/src/traj0_frei_png"
        self.rgb_path = os.path.join(self.dataset_path, "rgb")
        self.depth_path = os.path.join(self.dataset_path, "depth")
        self.assoc_file = os.path.join(self.dataset_path, "associations.txt")
        self.traj_file = os.path.join(self.dataset_path, "traj0.gt.freiburg")

        # Definicje stałych transformacji dla części robota
        self.robot_links = {
            'base_link': {
                'parent': 'base_footprint',
                'translation': [0.0, 0.0, 0.010],
                'rotation': [0.0, 0.0, 0.0]
            },
            'wheel_left_link': {
                'parent': 'base_link',
                'translation': [0.0, 0.144, 0.023],
                'rotation': [-1.57, 0.0, 0.0]
            },
            'wheel_right_link': {
                'parent': 'base_link',
                'translation': [0.0, -0.144, 0.023],
                'rotation': [-1.57, 0.0, 0.0]
            },
            'caster_back_right_link': {
                'parent': 'base_link',
                'translation': [-0.177, -0.064, -0.004],
                'rotation': [0.0, 0.0, 0.0]
            },
            'caster_back_left_link': {
                'parent': 'base_link',
                'translation': [-0.177, 0.064, -0.004],
                'rotation': [0.0, 0.0, 0.0]
            },
            'imu_link': {
                'parent': 'base_link',
                'translation': [-0.032, 0.0, 0.068],
                'rotation': [0.0, 0.0, 0.0]
            },
            'camera_link': {
                'parent': 'base_link',
                'translation': [0.073, -0.011, 0.084],
                'rotation': [0.0, 0.0, 0.0]
            },
            'camera_rgb_frame': {
                'parent': 'camera_link',
                'translation': [0.003, 0.011, 0.009],
                'rotation': [0.0, 0.0, 0.0]
            },
            'camera_rgb_optical_frame': {
                'parent': 'camera_rgb_frame',
                'translation': [0.0, 0.0, 0.0],
                'rotation': [-1.57, 0.0, -1.57]
            }
        }
        
    def read_associations(self):
        associations = []
        with open(self.assoc_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    data = line.strip().split()
                    associations.append({
                        'timestamp': float(data[0]),
                        'depth_file': data[1],
                        'rgb_file': data[3]
                    })
        return associations
    
    def read_trajectory(self):
        trajectory = {}
        with open(self.traj_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    data = line.strip().split()
                    timestamp = float(data[0])
                    position = [float(x) for x in data[1:4]]
                    orientation = [float(x) for x in data[4:8]]
                    trajectory[timestamp] = {
                        'position': position,
                        'orientation': orientation
                    }
        return trajectory
    
    def create_point_cloud(self, rgb_img, depth_img):
        rows, cols = depth_img.shape
        
        # Tworzenie siatki współrzędnych
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Konwersja głębokości na metry
        Z = depth_img.astype(float) / 5000.0
        
        # Obliczanie współrzędnych 3D
        X = (x - self.K[0,2]) * Z / self.K[0,0]
        Y = (y - self.K[1,2]) * Z / self.K[1,1]
        
        # Tworzenie chmury punktów
        points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        colors = rgb_img.reshape(-1, 3)
        
        # Usuwanie punktów z głębokością 0
        mask = Z.flatten() != 0
        points = points[mask]
        colors = colors[mask]
        
        return points, colors
    
    def publish_point_cloud(self, points, colors):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_rgb_optical_frame"  # Frame kamery Turtlebot3
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1)
        ]
        
        rgb_packed = np.zeros(len(colors), dtype=np.float32)
        for i in range(len(colors)):
            rgb_packed[i] = struct.unpack('f', struct.pack('BBBB',
                                        colors[i][2],
                                        colors[i][1],
                                        colors[i][0],
                                        255))[0]
        
        cloud_data = np.column_stack((points, rgb_packed))
        cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        self.cloud_pub.publish(cloud_msg)

    def publish_static_transforms(self):
        """Publikuje statyczne transformacje dla wszystkich części robota"""
        transforms = []
        
        for link_name, link_data in self.robot_links.items():
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = link_data['parent']
            transform.child_frame_id = link_name
            
            # Ustawienie translacji
            transform.transform.translation.x = link_data['translation'][0]
            transform.transform.translation.y = link_data['translation'][1]
            transform.transform.translation.z = link_data['translation'][2]
            
            # Konwersja RPY na kwaternion
            q = tf_trans.quaternion_from_euler(
                link_data['rotation'][0],
                link_data['rotation'][1],
                link_data['rotation'][2]
            )
            transform.transform.rotation.x = q[0]
            transform.transform.rotation.y = q[1]
            transform.transform.rotation.z = q[2]
            transform.transform.rotation.w = q[3]
            
            transforms.append(transform)
        
        return transforms

    def publish_transforms(self, position, orientation):
        """Publikuje transformacje dla robota i wszystkich jego części"""
        transforms = []
        
        # Transformacja dla base_footprint (pozycja robota)
        base_transform = TransformStamped()
        base_transform.header.stamp = rospy.Time.now()
        base_transform.header.frame_id = "map"
        base_transform.child_frame_id = "base_footprint"
        
        base_transform.transform.translation.x = position[0]
        base_transform.transform.translation.y = position[1]
        base_transform.transform.translation.z = 0.0
        
        roll, pitch, yaw = tf_trans.euler_from_quaternion(orientation)
        base_orientation = tf_trans.quaternion_from_euler(0, 0, yaw)
        
        base_transform.transform.rotation.x = base_orientation[0]
        base_transform.transform.rotation.y = base_orientation[1]
        base_transform.transform.rotation.z = base_orientation[2]
        base_transform.transform.rotation.w = base_orientation[3]
        
        transforms.append(base_transform)
        
        # Dodanie statycznych transformacji
        transforms.extend(self.publish_static_transforms())
        
        # Publikacja wszystkich transformacji
        for transform in transforms:
            self.tf_broadcaster.sendTransform(transform)
    
    def run(self):
        associations = self.read_associations()
        trajectory = self.read_trajectory()
        
        rate = rospy.Rate(10)  # 10 Hz
        
        for assoc in associations:
            if rospy.is_shutdown():
                break
                
            # Wczytywanie obrazów
            rgb_path = os.path.join(self.dataset_path, assoc['rgb_file'])
            depth_path = os.path.join(self.dataset_path, assoc['depth_file'])
            
            rgb_img = cv2.imread(rgb_path)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            
            # Tworzenie i publikacja chmury punktów
            points, colors = self.create_point_cloud(rgb_img, depth_img)
            self.publish_point_cloud(points, colors)
            
            # Publikacja transformacji jeśli dostępna
            if assoc['timestamp'] in trajectory:
                traj_data = trajectory[assoc['timestamp']]
                self.publish_transforms(traj_data['position'], traj_data['orientation'])
            
            rate.sleep()

if __name__ == '__main__':
    try:
        visualizer = TURTLEPointCloudVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
