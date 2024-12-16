#!/usr/bin/env python3

import rospy
import numpy as np
from octomap_msgs.msg import Octomap
from geometry_msgs.msg import Point
import octomap_msgs.msg as octomap_msgs
from nav_msgs.msg import OccupancyGrid
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, PoseStamped

class MapManager:
    def __init__(self):
        rospy.init_node('map_manager')

        # Pobierz parametry
        self.global_map_topic = rospy.get_param('~global_map_topic', 'global_octomap_binary')
        self.local_map_topic = rospy.get_param('~local_map_topic', 'local_octomap_binary_1')
        self.merged_map_topic = rospy.get_param('~merged_map_topic', 'merged_octomap')

        # Parametry obszaru lokalnego
        self.local_bbox_min = Point(-2.0, -2.0, 0.0)
        self.local_bbox_max = Point(2.0, 2.0, 2.0)

        # Przechowywanie ostatnich wiadomości map
        self.global_map_msg = None
        self.local_map_msg = None

        # Konfiguracja tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subskrybuj mapy
        rospy.Subscriber(self.global_map_topic, Octomap, self.global_map_callback)
        rospy.Subscriber(self.local_map_topic, Octomap, self.local_map_callback)

        # Publisher dla połączonej mapy i wizualizacji obszaru lokalnego
        self.merged_pub = rospy.Publisher(self.merged_map_topic, Octomap, queue_size=1)
        self.local_area_pub = rospy.Publisher('local_area_visualization', OccupancyGrid, queue_size=1)

        # Publikuj obszar lokalny jako OccupancyGrid dla wizualizacji
        self.publish_local_area_visualization()

        rospy.loginfo("Map Manager zainicjalizowany")

    def publish_local_area_visualization(self):
        """Publikuje wizualizację obszaru lokalnego."""
        grid = OccupancyGrid()
        grid.header.frame_id = "odom"
        grid.header.stamp = rospy.Time.now()
        
        # Ustawienia siatki
        resolution = 0.05  # 5cm na komórkę
        width = int((self.local_bbox_max.x - self.local_bbox_min.x) / resolution)
        height = int((self.local_bbox_max.y - self.local_bbox_min.y) / resolution)
        
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = self.local_bbox_min.x
        grid.info.origin.position.y = self.local_bbox_min.y
        
        # Tworzenie danych siatki (50 dla wizualizacji obszaru lokalnego)
        grid.data = [50] * (width * height)
        
        self.local_area_pub.publish(grid)

    def is_point_in_local_area(self, point):
        """Sprawdza, czy punkt znajduje się w obszarze lokalnym."""
        return (self.local_bbox_min.x <= point.x <= self.local_bbox_max.x and
                self.local_bbox_min.y <= point.y <= self.local_bbox_max.y and
                self.local_bbox_min.z <= point.z <= self.local_bbox_max.z)

    def global_map_callback(self, msg):
        """Callback dla globalnej mapy."""
        try:
            self.global_map_msg = msg
            self.merge_maps()
            rospy.loginfo("Otrzymano globalną mapę")
        except Exception as e:
            rospy.logerr(f"Błąd podczas przetwarzania globalnej mapy: {e}")

    def local_map_callback(self, msg):
        """Callback dla lokalnej mapy."""
        try:
            self.local_map_msg = msg
            self.merge_maps()
            rospy.loginfo("Otrzymano lokalną mapę")
        except Exception as e:
            rospy.logerr(f"Błąd podczas przetwarzania lokalnej mapy: {e}")

    def merge_maps(self):
        """Łączy globalną i lokalną mapę."""
        if self.global_map_msg is None or self.local_map_msg is None:
            return

        try:
            # Tworzenie nowej wiadomości mapy
            merged_msg = Octomap()
            merged_msg.header.frame_id = "odom"
            merged_msg.header.stamp = rospy.Time.now()
            merged_msg.binary = True
            merged_msg.id = "OcTree"
            
            # Używamy rozdzielczości z mapy lokalnej dla obszaru lokalnego
            merged_msg.resolution = self.local_map_msg.resolution
            
            # Łączenie danych z obu map
            # W obszarze lokalnym używamy danych z mapy lokalnej
            # Poza obszarem lokalnym używamy danych z mapy globalnej
            if self.is_point_in_local_area(Point(0, 0, 0)):  # Sprawdzamy czy robot jest w obszarze lokalnym
                merged_msg.data = self.local_map_msg.data
                rospy.loginfo("Używam mapy lokalnej (wysoka rozdzielczość)")
            else:
                merged_msg.data = self.global_map_msg.data
                rospy.loginfo("Używam mapy globalnej (standardowa rozdzielczość)")

            # Publikuj połączoną mapę
            self.merged_pub.publish(merged_msg)
            
            # Aktualizuj wizualizację obszaru lokalnego
            self.publish_local_area_visualization()
            
            rospy.loginfo("Opublikowano połączoną mapę")

        except Exception as e:
            rospy.logerr(f"Błąd podczas łączenia map: {e}")

    def run(self):
        """Główna pętla node'a."""
        rate = rospy.Rate(1)  # 1 Hz
        rospy.loginfo("Map Manager uruchomiony")
        
        while not rospy.is_shutdown():
            # Aktualizuj wizualizację obszaru lokalnego
            self.publish_local_area_visualization()
            rate.sleep()

if __name__ == '__main__':
    try:
        map_manager = MapManager()
        map_manager.run()
    except rospy.ROSInterruptException:
        pass
