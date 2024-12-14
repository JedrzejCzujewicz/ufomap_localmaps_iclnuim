#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

class PointCloudFilter:
    def __init__(self):
        rospy.init_node('pointcloud_filter', anonymous=True)

        # Parametry obszaru lokalnego (sześcian 1m x 1m x 1m wokół 0,0,0)
        self.x_min = -0.5
        self.x_max = 0.5
        self.y_min = -0.5
        self.y_max = 0.5
        self.z_min = -0.5
        self.z_max = 0.5

        # Subscriber i publisher
        self.sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.callback)
        self.pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=1)

    def is_point_in_bounds(self, x, y, z):
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def callback(self, data):
        # Lista na przefiltrowane punkty
        filtered_points = []

        # Iteracja przez wszystkie punkty
        for p in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            if self.is_point_in_bounds(p[0], p[1], p[2]):
                filtered_points.append([p[0], p[1], p[2]])

        if not filtered_points:  # Jeśli nie ma punktów w obszarze
            return

        # Konwersja do numpy array dla wydajności
        filtered_points = np.array(filtered_points, dtype=np.float32)

        # Tworzenie nowej wiadomości PointCloud2
        header = Header()
        header.stamp = data.header.stamp
        header.frame_id = data.header.frame_id

        # Publikowanie przefiltrowanej chmury punktów
        filtered_cloud = pc2.create_cloud_xyz32(header, filtered_points)
        self.pub.publish(filtered_cloud)

if __name__ == '__main__':
    try:
        filter = PointCloudFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
