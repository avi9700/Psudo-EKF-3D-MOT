#!/usr/bin/env python
import rospy
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from geometry_msgs.msg import Point, PoseArray, Pose, Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Header
import sensor_msgs.point_cloud2 as pc2
from shapely.geometry import Polygon

# Nonlinear state transition function
def fx(x, dt):
    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    return np.dot(F, x)

# Nonlinear observation function
def hx(x):
    return x[:3]  # Return only position information

# Jacobian of the state transition function
def jacobian_F(x, dt):
    return np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])

# Jacobian of the observation function
def jacobian_H(x):
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])

class AB3DMOTWithEKF:
    def __init__(self):
        self.trackers = []

    def initialize_tracker(self, initial_detection):
        ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        ekf.x = np.array([initial_detection[0], initial_detection[1], initial_detection[2], 0, 0, 0])
        ekf.F = np.eye(6)
        ekf.H = np.eye(3, 6)
        ekf.P *= 10.0
        ekf.R = np.eye(3)
        ekf.Q = np.eye(6)
        self.trackers.append(ekf)

    def predict_and_update(self, dt):
        for ekf in self.trackers:
            ekf.F = jacobian_F(ekf.x, dt)
            ekf.predict()

    def update_tracker(self, detections):
        for ekf, detection in zip(self.trackers, detections):
            ekf.update(detection, HJacobian=jacobian_H, Hx=hx)

    def ego_compensation(self, detections, ego_velocity, ego_yaw_rate, dt):
        compensated_detections = []
        for detection in detections:
            rel_x, rel_y = detection[:2]
            ego_dx = ego_velocity * dt * np.cos(ego_yaw_rate * dt)
            ego_dy = ego_velocity * dt * np.sin(ego_yaw_rate * dt)
            comp_x = rel_x - ego_dx
            comp_y = rel_y - ego_dy
            compensated_detections.append(np.array([comp_x, comp_y, detection[2]]))
        return compensated_detections

    def track(self, detections, detection_types, ego_velocity, ego_yaw_rate, dt):
        for detection, detection_type in zip(detections, detection_types):
            if detection_type == '3d_box':
                if len(self.trackers) < len(detections):
                    self.initialize_tracker(detection)  # Initialize using 3D box
            elif detection_type == 'polygon_cluster':
                approx_box = self.polygon_to_box(detection)
                if len(self.trackers) < len(detections):
                    self.initialize_tracker(approx_box)  # Initialize using converted box

        compensated_detections = self.ego_compensation(detections, ego_velocity, ego_yaw_rate, dt)
        self.predict_and_update(dt)
        self.update_tracker(compensated_detections)
    
    def polygon_to_box(self, polygon):
        # Convert polygon to a bounding box (3D)
        shapely_poly = Polygon(polygon[:,:2])  # Use 2D projection
        minx, miny, maxx, maxy = shapely_poly.bounds
        z_min = min(polygon[:,2])
        z_max = max(polygon[:,2])
        # Return the center of the bounding box and its approximate 3D size
        return np.array([(minx+maxx)/2, (miny+maxy)/2, (z_min+z_max)/2])

    def get_all_tracked_objects(self):
        tracked_objects = []
        for ekf in self.trackers:
            vx, vy, vz = ekf.x[3:6]
            speed = np.sqrt(vx**2 + vy**2)
            yaw_rate = np.arctan2(vy, vx)
            tracked_objects.append({
                'position': ekf.x[:3],
                'speed': speed,
                'yaw_rate': yaw_rate
            })
        return tracked_objects

class AB3DMOTNode:
    def __init__(self):
        rospy.init_node('ab3dmot_with_ekf_node')
        
        self.tracker = AB3DMOTWithEKF()
        
        self.sub_detections = rospy.Subscriber('/sensor/detections', PointCloud2, self.detections_callback)
        self.sub_ego_velocity = rospy.Subscriber('/ego/velocity', Twist, self.velocity_callback)
        
        self.pub_tracked_positions = rospy.Publisher('/tracked/positions', PoseArray, queue_size=10)
        self.pub_tracked_speeds = rospy.Publisher('/tracked/speeds', Float32, queue_size=10)
        self.pub_tracked_yaw_rates = rospy.Publisher('/tracked/yaw_rates', Float32, queue_size=10)
        
        self.ego_velocity = 0.0
        self.ego_yaw_rate = 0.0
        self.dt = 1.0  # Default time step

    def detections_callback(self, msg):
        detections, detection_types = self.process_point_cloud(msg)
        self.tracker.track(detections, detection_types, self.ego_velocity, self.ego_yaw_rate, self.dt)
        
        tracked_objects = self.tracker.get_all_tracked_objects()
        positions_msg = PoseArray(header=Header(stamp=rospy.Time.now()))
        
        for obj in tracked_objects:
            pose = Pose()
            pose.position.x = obj['position'][0]
            pose.position.y = obj['position'][1]
            pose.position.z = obj['position'][2]
            positions_msg.poses.append(pose)
        
        self.pub_tracked_positions.publish(positions_msg)
        
        for obj in tracked_objects:
            self.pub_tracked_speeds.publish(obj['speed'])
            self.pub_tracked_yaw_rates.publish(obj['yaw_rate'])

    def velocity_callback(self, msg):
        self.ego_velocity = msg.linear.x
        self.ego_yaw_rate = msg.angular.z

    def process_point_cloud(self, msg):
        detections = []
        detection_types = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            if some_condition_for_3d_box(p):
                detections.append(np.array([p[0], p[1], p[2]]))
                detection_types.append('3d_box')
            else:
                # Example for clusters (polygons)
                polygon = np.array([[p[0], p[1], p[2]], ...])  # Construct full polygon
                detections.append(polygon)
                detection_types.append('polygon_cluster')
        return detections, detection_types

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = AB3DMOTNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
