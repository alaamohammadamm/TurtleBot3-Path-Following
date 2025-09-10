#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class MoveTB3Command(Node):
    def __init__(self):
        super().__init__('move_tb3_command_node')
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear_speed = 0.1  # Default linear speed (m/s)
        self.angular_speed = 0.1  # Approx. 35? deg/s

    def move(self, linear_vel, angular_vel):
        vel_msg = Twist()
        vel_msg.linear.x = linear_vel
        vel_msg.angular.z = angular_vel

        self.stop()

    def stop(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        time.sleep(0.1)

