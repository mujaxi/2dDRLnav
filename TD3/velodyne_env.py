import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
# from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.25
COLLISION_DIST = 0.25
TIME_DELTA = 0.1
MAX_LASER_RANGE = 5
# Set a height (in velodyne reference frame) at which to start filtering out the ground
# Approximate default values: Pioneer P3DX -0.2, Turtlebot Burger -0.122, Turtlebot waffle -0.072
FILTER_GROUND_HEIGHT = -0.072


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if 0.54 > x > 0.71 and 0.58 > y > 0.76:
        goal_ok = False

    if 0.51 > x > 0.69 and -0.88 > y > -1.15:
        goal_ok = False

    if 2.72 > x > 2.9 and 0.88 > y > 1.15:
        goal_ok = False

    if 2.95 > x > 3.13 and -0.82 > y > -1.0:
        goal_ok = False

    if 5.74 > x > 5.92 and 0.86 > y > 1.04:
        goal_ok = False

    if 6.12 > x > 6.26 and -0.83 > y > -1.01:
        goal_ok = False

    # if 4 > x > 2.5 and 0.7 > y > -3.2:
    #     goal_ok = False

    # if 6.2 > x > 3.8 and -3.3 > y > -4.2:
    #     goal_ok = False

    # if 4.2 > x > 1.3 and 3.7 > y > 1.5:
    #     goal_ok = False

    # if -3.0 > x > -7.2 and 0.5 > y > -1.5:
    #     goal_ok = False

    if x > 6.0 or x < 1 or y > 1.3 or y < -1.3:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 3.0
        self.lower = -3.0
        self.velodyne_data = np.ones(self.environment_dim) * 18
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/scan", LaserScan, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

    # # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # # range as state representation
    # def velodyne_callback(self, v):
    #     data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
    #     self.velodyne_data = np.ones(self.environment_dim) * 10
    #     for i in range(len(data)):
    #         if data[i][2] > FILTER_GROUND_HEIGHT:
    #             dot = data[i][0] * 1 + data[i][1] * 0
    #             mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
    #             mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
    #             beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
    #             dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

    #             for j in range(len(self.gaps)):
    #                 if self.gaps[j][0] <= beta < self.gaps[j][1]:
    #                     self.velodyne_data[j] = min(self.velodyne_data[j], dist)
    #                     break

    def velodyne_callback(self, msg):
        # Initialize laser data as infinite (or max range) for each angle
        self.velodyne_data = np.ones(self.environment_dim) * 18

        # Loop through each reading in the LaserScan message
        for i in range(len(msg.ranges)):
            angle = msg.angle_min + i * msg.angle_increment  # Calculate the angle of each scan reading
            distance = msg.ranges[i]  # Get the distance reading

            # Only process valid distance readings
            if distance > msg.range_min and distance < msg.range_max:
                # Convert the angle to a bin in the laser data array
                bin_index = int((angle + np.pi) / (2 * np.pi) * self.environment_dim)
                if bin_index >= 0 and bin_index < len(self.velodyne_data):
                    # Update the distance for the specific angle bin
                    self.velodyne_data[bin_index] = min(self.velodyne_data[bin_index], distance)

    
    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):

        target = False

        # Validate and publish robot action
        max_linear_speed = 0.26
        max_angular_speed = 1.82
        action[0] = np.clip(action[0], -max_linear_speed, max_linear_speed)
        action[1] = np.clip(action[1], -max_angular_speed, max_angular_speed)

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Control physics
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # Sensor data
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        laser_state = np.ravel(self.velodyne_data) / MAX_LASER_RANGE

        # Odometry
        if not hasattr(self, 'last_odom'):
            print("Odometry data missing!")
            return None, None, None, None

        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Distance and angle to goal
        self.distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        beta = math.atan2(skew_y, skew_x)
        theta = (beta - angle + np.pi) % (2 * np.pi) - np.pi

        # Goal detection
        if self.distance < GOAL_REACHED_DIST:
            print(f"Goal reached! Distance: {self.distance}")
            target = True
            done = True
        elif collision:
            print(f"Crashed!, Distance: {self.distance} ")

        robot_state = [self.distance, theta, action[0], action[1]]
        state = np.concatenate((laser_state, robot_state))
        reward = self.get_reward(target, collision, action, min_laser)
        if done:
            print(f"Reward Collected = {reward}")

        return state, reward, done, target


    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        self.distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [self.distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.0 or distance_to_goal < 1.0:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    # @staticmethod
    def get_reward(self, target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            distance_penalty = self.distance * -10            
            return -50.0 + distance_penalty
        elif self.distance > 5:
            return -20.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] - abs(action[1]) - (r3(min_laser)*5) 
