#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

class PointCloudFilter:
    def __init__(self):
        rospy.init_node('pointcloud_filter', anonymous=True)

        # Pobierz parametry obszaru lokalnego z parametrów ROS
        self.x_min = rospy.get_param('~x_min', -2.0)
        self.x_max = rospy.get_param('~x_max', 2.0)
        self.y_min = rospy.get_param('~y_min', -2.0)
        self.y_max = rospy.get_param('~y_max', 2.0)
        self.z_min = rospy.get_param('~z_min', 0.0)
        self.z_max = rospy.get_param('~z_max', 2.0)

        # Subscriber dla oryginalnej chmury punktów
        self.sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.callback, queue_size=1)
        
        # Publishery dla obu filtrowanych chmur
        self.pub_local = rospy.Publisher('/filtered_points_local', PointCloud2, queue_size=1)
        self.pub_global = rospy.Publisher('/filtered_points_global', PointCloud2, queue_size=1)

        rospy.loginfo("PointCloud Filter zainicjalizowany z parametrami:")
        rospy.loginfo(f"X: [{self.x_min}, {self.x_max}]")
        rospy.loginfo(f"Y: [{self.y_min}, {self.y_max}]")
        rospy.loginfo(f"Z: [{self.z_min}, {self.z_max}]")

    def is_point_in_bounds(self, x, y, z):
        """Sprawdza, czy punkt znajduje się w zdefiniowanym obszarze."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def callback(self, data):
        """Callback dla przychodzących chmur punktów."""
        try:
            # Listy na przefiltrowane punkty
            local_points = []
            global_points = []

            # Iteracja przez wszystkie punkty
            for p in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
                if self.is_point_in_bounds(p[0], p[1], p[2]):
                    local_points.append([p[0], p[1], p[2]])
                else:
                    global_points.append([p[0], p[1], p[2]])

            # Tworzenie i publikowanie chmury punktów dla mapy lokalnej
            if local_points:
                local_points_array = np.array(local_points, dtype=np.float32)
                header = Header()
                header.stamp = data.header.stamp
                header.frame_id = data.header.frame_id
                local_cloud = pc2.create_cloud_xyz32(header, local_points_array)
                self.pub_local.publish(local_cloud)

            # Tworzenie i publikowanie chmury punktów dla mapy globalnej
            if global_points:
                global_points_array = np.array(global_points, dtype=np.float32)
                header = Header()
                header.stamp = data.header.stamp
                header.frame_id = data.header.frame_id
                global_cloud = pc2.create_cloud_xyz32(header, global_points_array)
                self.pub_global.publish(global_cloud)

        except Exception as e:
            rospy.logerr(f"Błąd podczas przetwarzania chmury punktów: {e}")

if __name__ == '__main__':
    try:
        filter = PointCloudFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
