import math
import os
import random
import subprocess
import time
from os import path
import matplotlib.pyplot as plt

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.25
TIME_DELTA = 1

# Set a height (in velodyne reference frame) at which to start filtering out the ground
# Approximate default values: Pioneer P3DX -0.2, Turtlebot Burger -0.122, Turtlebot waffle -0.072
FILTER_GROUND_HEIGHT = -0.072


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    # # current obstacle position (update if world change)
    # <pose>0.61696 -0.8867 0.375 0 -0 0</pose>
    # <pose>3.0455 -0.7152 0.375 0 -0 0</pose>
    # <pose>5.8324 0.7625 0.375 0 -0 0</pose>
    # <pose>6.2146 -0.7269 0.375 0 -0 0</pose>
    # <pose>0.6447 0.6813 0.375 0 -0 0</pose>
    # <pose>2.8167 0.8742 0.375 0 -0 0</pose>
    
    # <pose>3.96847 0.477565 0.149 0 -0 0</pose>
    # <pose>2.295 -0.144744 0.149 0 -0 0</pose>
    # <pose>5.27726 0.208854 0.149 0 -0 0</pose>

    # if 0.54 < x < 0.71 and 0.58 < y < 0.76:
    #     goal_ok = False

    if 0.4 < x < 0.8 and -1.8 < y < -0.68:
        goal_ok = False

    if 2.8 < x < 3.2 and -0.9 < y < -0.5:
        goal_ok = False

    if 5.6 < x < 6 and 0.5 < y < 0.9:
        goal_ok = False

    if 5.9 < x < 6.3 and -0.9 < y < -0.5:
        goal_ok = False

    if 0.4 < x < 0.8 and 0.48 < y < 0.88:
        goal_ok = False

    if 3 > x > 2.6 and 1 > y > 0.6:
        goal_ok = False

    if 4.4 > x > 3.4 and 0.9 > y > -0.1:
        goal_ok = False

    if 2.7 > x > 1.7 and 0.4 > y > -0.6:
        goal_ok = False

    if 5.7 > x > 4.7 and 0.7 > y > -0.3:
        goal_ok = False

    if x > 7.0 or x < 1 or y > 1.3 or y < -1.3:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        # Initialize lists to track robot's positions
        self.positions_x = []
        self.positions_y = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Robot Trajectory")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.goal_marker = self.ax.scatter([], [], color="red", label="Goal Position", marker="*", s=100)
        self.robot_path, = self.ax.plot([], [], label="Robot Path", color="blue")
        self.ax.legend()
        self.ax.grid(True)

        self.random_goal = False

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 3.0
        self.lower = -3.0
        self.velodyne_data = np.ones(self.environment_dim) * 8
        self.last_odom = None

        self.set_self_state = ModelState()
        # self.set_self_state.model_name = "r1" #for training
        self.set_self_state.model_name = ""
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
        # subprocess.Popen(["roscore", "-p", port])

        # print("Roscore launched!")

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
        # self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        # self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        # self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        # self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        # self.velodyne = rospy.Subscriber(
        #     "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        # )
        self.velodyne = rospy.Subscriber(
            "/scan", LaserScan, self.velodyne_callback, queue_size=1
        )
        # self.odom = rospy.Subscriber(
        #     "/r1/odom", Odometry, self.odom_callback, queue_size=1
        # )
        self.odom = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback, queue_size=1
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

    # def velodyne_callback(self, v):
    #     # Reset velodyne_data
    #     self.velodyne_data = np.ones(self.environment_dim) * 3.5  # Reset each time

    #     # Extract lidar scan data
    #     distances = v.ranges  # List of distances at each angle (in meters)
    #     angle_min = v.angle_min  # Minimum angle of the laser scan
    #     angle_max = v.angle_max  # Maximum angle of the laser scan
    #     angle_increment = v.angle_increment  # Angle between each measurement (in radians)

    #     # Calculate the total number of measurements
    #     num_readings = len(distances)

    #     # Calculate the indices corresponding to the front 180 degrees (-90 to +90)
    #     front_start_angle = -np.pi / 2  # -90 degrees in radians
    #     front_end_angle = np.pi / 2 # +90 degrees in radians

    #     # Convert front angles to indices
    #     start_index = max(0, int((front_start_angle - angle_min) / angle_increment))
    #     end_index = min(num_readings, int((front_end_angle - angle_min) / angle_increment))

    #     # Iterate through the relevant range of distances
    #     for i in range(start_index, end_index):
    #         # Ignore invalid measurements (e.g., NaN or inf)
    #         if not np.isfinite(distances[i]):
    #             continue

    #         # Calculate the angle corresponding to this distance (in radians)
    #         angle = i * angle_increment + angle_min

    #         # Update velodyne_data based on angle range
    #         for j in range(len(self.gaps)):
    #             if self.gaps[j][0] <= angle < self.gaps[j][1]:
    #                 self.velodyne_data[j] = min(self.velodyne_data[j], distances[i])
    #                 break
        
    
    def velodyne_callback(self, v):
        # # Initialize velodyne_data
        self.velodyne_data = np.ones(self.environment_dim) * 8 # Reset each time
        
        # Extract lidar scan data (distance values and angle range)
        distances = v.ranges  # List of distances at each angle (in meters)
        angle_increment = v.angle_increment  # Angle between each measurement (in radians)
        
        # Iterate through the distances and angles from the 2D Lidar
        for i in range(len(distances)):
            # Ignore invalid measurements (e.g., NaN or inf)
            if not np.isfinite(distances[i]):
                continue
            
            # Calculate the angle corresponding to this distance (in radians)
            angle = i * angle_increment + v.angle_min  # msg.angle_min is the starting angle
            
            # Calculate the distance
            dist = distances[i]
            
            # Update velodyne_data based on angle range
            for j in range(len(self.gaps)):
                # If the angle falls within a gap (angular sector)
                if self.gaps[j][0] <= angle < self.gaps[j][1]:
                    self.velodyne_data[j] = min(self.velodyne_data[j], dist)  # Update the minimum distance for the sector
                    break
        # print(self.velodyne_data)

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # rospy.wait_for_service("/gazebo/unpause_physics")
        # try:
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/unpause_physics service call failed")

        # # propagate state for TIME_DELTA seconds
        # time.sleep(TIME_DELTA)

        # rospy.wait_for_service("/gazebo/pause_physics")
        # try:
        #     pass
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
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

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        # print("Goal distance :", distance)

        # # Calculate the relative angle between the robots heading and heading toward the goal
        # skew_x = self.goal_x - self.odom_x
        # skew_y = self.goal_y - self.odom_y
        # dot = skew_x * 1 + skew_y * 0
        # mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        # mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        # beta = math.acos(dot / (mag1 * mag2))
        # if skew_y < 0:
        #     if skew_x < 0:
        #         beta = -beta
        #     else:
        #         beta = 0 - beta
        # theta = beta - angle
        # if theta > np.pi:
        #     theta = np.pi - theta
        #     theta = -np.pi - theta
        # if theta < -np.pi:
        #     theta = -np.pi - theta
        #     theta = np.pi - theta

        # Calculate the relative angle between the robot's heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        # Use atan2 to directly compute the angle from the robot to the goal (this is more stable)
        goal_angle = math.atan2(skew_y, skew_x)
        # print("goal angle direction : ", goal_angle)

        # The robot's current heading (angle) is given (assuming it is in radians)
        theta = goal_angle - angle

        # Normalize the angle to be within -π to π
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        # print("theta : ", theta)

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            # done = True
            print("Goal!")
            self.change_goal()
        
        # Update the positions list
        self.positions_x.append(self.odom_x)
        self.positions_y.append(self.odom_y)
        
        self.robot_path.set_data(self.positions_x, self.positions_y)
        self.goal_marker.set_offsets([self.goal_x, self.goal_y])
        plt.pause(0.01) 

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        # print("state :", state)
        reward = self.get_reward(target, collision, action, min_laser)
        # print("Reward Collected :", reward)
        return state, reward, done, target

    def reset(self):
        # Reset the robot position and goal in the real world
        # First stop the robot if it's moving
        vel_cmd = Twist()
        self.vel_pub.publish(vel_cmd)  # Send a stop command to the robot

        # Reset robot odometry and position to a fixed or random starting point
        self.odom_x = 0.0
        self.odom_y = 0.0
        
        
            
        # Update the robot's pose in the real world
        object_state = ModelState()
        object_state.model_name = ""  # Replace with your TurtleBot's model name
        object_state.pose.position.x = self.odom_x
        object_state.pose.position.y = self.odom_y
        object_state.pose.position.z = 0.0  # Typically, z position is kept at 0 for ground level

        # Set orientation (no rotation initially)
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w

        # Publish the initial pose to the robot (set robot's initial position)
        self.set_state.publish(object_state)

        # Set a random goal location within the accessible area (again, avoiding obstacles)
        self.change_goal()

        # Optionally, reset obstacles in the environment
        # self.random_box()

        # Publish visual markers for goal and robot in RViz
        self.publish_markers([0.0, 0.0])


        # Prepare the state representation of the robot
        v_state = self.velodyne_data[:]  # Get lidar data from the robot
        laser_state = [v_state]

        # Calculate the distance to the goal
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # Calculate the relative angle to the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(skew_x**2 + skew_y**2)
        mag2 = math.sqrt(1**2)
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = -beta

        # Adjust angle relative to the robot's heading
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Prepare the final robot state
        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)

        return state


    def change_goal(self):

        goal_ok = False
        if self.random_goal :
            # Place a new goal and check if its location is not on one of the obstacles
            # if self.upper < 10:
            #     self.upper += 0.004
            # if self.lower > -10:
            #     self.lower -= 0.004

            while not goal_ok:
                self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
                self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
                goal_ok = check_pos(self.goal_x, self.goal_y)
        else:
            # logic for changing fixed goal set sequentially for specific task-based application
            # Assuming a list of predefined goals for specific tasks (can be extended to use dynamic goals)
            # # current obstacle position (update if world change)
            # <pose>0.61696 -0.8867 0.375 0 -0 0</pose>
            # <pose>3.0455 -0.7152 0.375 0 -0 0</pose>
            # <pose>5.8324 0.7625 0.375 0 -0 0</pose>
            # <pose>6.2146 -0.7269 0.375 0 -0 0</pose>
            # <pose>0.6447 0.6813 0.375 0 -0 0</pose>
            # <pose>2.8167 0.8742 0.375 0 -0 0</pose>
            
            # <pose>3.96847 0.477565 0.149 0 -0 0</pose>
            # <pose>2.295 -0.144744 0.149 0 -0 0</pose>
            # <pose>5.27726 0.208854 0.149 0 -0 0</pose>
            predefined_goals = [
                # {"x": 0.61696, "y": -0.8867},
                # {"x": 2.8167, "y": 0.8742},
                # {"x": 3.0455, "y": -0.7152},
                # {"x": 5.8324, "y": 0.7625},
                # {"x": 6.2146, "y": -0.7269},
                # {"x": 0.6447, "y": 0.6813},
                # {"x": 0, "y": 0},
                {"x": 3, "y": 0},
                {"x": 4, "y": 1} 
            ]
            
            # Logic to choose the next goal in the sequence
            if hasattr(self, 'goal_index'):
                self.goal_index = (self.goal_index + 1) % len(predefined_goals)
            else:
                self.goal_index = 0
            
            self.goal_x = predefined_goals[self.goal_index]["x"]
            self.goal_y = predefined_goals[self.goal_index]["y"]
            
            # check that the chosen goal is not on an obstacle
            goal_ok = check_pos(self.goal_x, self.goal_y)       

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

    def save_trajectory(self, filename="robot_trajectory.png"):
        """Save the robot trajectory plot to a file in the /trajectory folder"""
    
        # Define the trajectory directory
        trajectory_dir = "~/Desktop/trajectory"
        
        # Check if the directory exists, if not, create it
        if not os.path.exists(trajectory_dir):
            os.makedirs(trajectory_dir)
            # print(f"Created directory: {trajectory_dir}")
        
        # Generate the full path for the file with incremental filename
        base_filename = filename.split('.')[0]  # Get the base filename without extension
        file_extension = filename.split('.')[1]  # Get the file extension (png)
        
        # Check for existing files and increment the filename
        file_counter = 1
        while os.path.exists(f"{trajectory_dir}/{base_filename}{file_counter}.{file_extension}"):
            file_counter += 1  # Increment the counter if the file already exists
        
        # Create the new filename
        new_filename = f"{trajectory_dir}/{base_filename}{file_counter}.{file_extension}"
        
        # Save the current plot to the new file
        self.fig.savefig(new_filename, dpi=300)
        print(f"Plot saved as {new_filename}")
        
        # Show the plot after the simulation
        plt.show()

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        
        penal = 0
        if 0 > action[0] >0.26:
            penal = -20
        
        if target:
            # print("Goal !")
            return 100.0 + penal
        elif collision:
            # print("Crash!")
            return -50.0 + penal
        else:
            r3 = lambda x: 1 - x if x < 0.5 else 0.0
            # print("Robot Jem!")
            return action[0] * 4 - abs(action[1]) * 2 - r3(min_laser) / 2 + penal
