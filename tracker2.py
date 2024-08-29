#!/usr/bin/env python
import rospy
import numpy as np
import time
from geometry_msgs.msg import Point, PoseArray, Pose, Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Header
import sensor_msgs.point_cloud2 as pc2

# 상태 전이 함수
def fx(x, dt):
    F = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, dt, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, dt, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # yaw 및 크기 정보 추가
    return np.dot(F, x)

# 상태 전이 함수의 Jacobian 계산
def jacobian_F(x, dt):
    return np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, dt, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, dt, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# 관측 함수
def hx(x):
    return x[:3]  # 위치 정보만 반환

# 관측 함수의 Jacobian 계산
def jacobian_H(x):
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0]])

class EKFTracker:
    def __init__(self, initial_detection, size=None, yaw=None, obj_class=None):
        # 초기 상태 벡터와 행렬들 초기화
        self.x = np.array([
            initial_detection[0], initial_detection[1], initial_detection[2],  # 위치 (x, y, z)
            0, 0, 0,  # 속도 (vx, vy, vz)
            yaw if yaw is not None else 0,  # yaw
            size if size is not None else [1, 1, 1]  # 크기 (w, h, l)
        ])
        self.P = np.eye(9) * 10  # 상태 공분산 행렬 초기화
        self.F = np.eye(9)  # 상태 전이 행렬
        self.H = np.eye(3, 9)  # 관측 행렬
        self.Q = np.eye(9)  # 프로세스 잡음 공분산 행렬
        self.R = np.eye(3)  # 관측 잡음 공분산 행렬
        self.has_yaw = yaw is not None  # yaw 정보 유무
        self.class_id = obj_class  # 객체 클래스 저장

    def predict(self, dt):
        # 상태 예측
        self.F = jacobian_F(self.x, dt)
        self.x = np.dot(self.F, self.x)
        
        # 공분산 예측
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z, yaw=None, size=None):
        # 관측 잔차 계산
        y = z - hx(self.x)
        
        # 관측 행렬의 Jacobian 계산
        H = jacobian_H(self.x)
        
        # 잔차 공분산 계산
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        
        # 칼만 이득 계산
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        
        # 상태 업데이트
        self.x = self.x + np.dot(K, y)
        if yaw is not None:
            self.x[6] = yaw  # yaw 값을 업데이트
        if size is not None:
            self.x[7:10] = size  # 크기 값을 업데이트

        # 공분산 업데이트
        I = np.eye(len(self.x))
        self.P = np.dot(I - np.dot(K, H), self.P)

class AB3DMOTWithCustomEKF:
    def __init__(self):
        self.trackers = []

    def initialize_tracker(self, initial_detection, size=None, yaw=None, obj_class=None):
        tracker = EKFTracker(initial_detection, size, yaw, obj_class)
        self.trackers.append(tracker)

    def predict_and_update(self, dt, detections, size_list, yaw_list, class_list):
        for tracker, detection, size, yaw, obj_class in zip(self.trackers, detections, size_list, yaw_list, class_list):
            tracker.predict(dt)
            tracker.update(detection, yaw, size)
            tracker.has_yaw = yaw is not None
            tracker.class_id = obj_class

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

    def track(self, detections, ego_velocity, ego_yaw_rate, dt, size_list, yaw_list, class_list):
        if len(self.trackers) < len(detections):
            for detection, size, yaw, obj_class in zip(detections[len(self.trackers):], size_list[len(self.trackers):], yaw_list[len(self.trackers):], class_list[len(self.trackers):]):
                self.initialize_tracker(detection, size, yaw, obj_class)
        
        compensated_detections = self.ego_compensation(detections, ego_velocity, ego_yaw_rate, dt)
        self.predict_and_update(dt, compensated_detections, size_list, yaw_list, class_list)
    
    def get_all_tracked_objects(self):
        tracked_objects = []
        for tracker in self.trackers:
            vx, vy, vz = tracker.x[3:6]
            speed = np.sqrt(vx**2 + vy**2 + vz**2)  # 3D 속도 계산
            yaw = tracker.x[6] if tracker.has_yaw else None
            size = tracker.x[7:10]
            tracked_objects.append({
                'position': tracker.x[:3],
                'speed': speed,
                'vx': vx,
                'vy': vy,
                'vz': vz,
                'yaw': yaw,
                'size': size,
                'class_id': tracker.class_id
            })
        return tracked_objects

class AB3DMOTNode:
    def __init__(self):
        rospy.init_node('ab3dmot_with_custom_ekf_node')
        
        self.tracker = AB3DMOTWithCustomEKF()
        
        self.sub_detections = rospy.Subscriber('/sensor/detections', PointCloud2, self.detections_callback)
        self.sub_ego_velocity = rospy.Subscriber('/ego/velocity', Twist, self.velocity_callback)
        
        self.pub_tracked_positions = rospy.Publisher('/tracked/positions', PoseArray, queue_size=10)
        self.pub_tracked_speeds = rospy.Publisher('/tracked/speeds', Float32, queue_size=10)
        self.pub_tracked_vx = rospy.Publisher('/tracked/vx', Float32, queue_size=10)
        self.pub_tracked_vy = rospy.Publisher('/tracked/vy', Float32, queue_size=10)
        self.pub_tracked_vz = rospy.Publisher('/tracked/vz', Float32, queue_size=10)
        self.pub_tracked_yaws = rospy.Publisher('/tracked/yaws', Float32, queue_size=10)
        self.pub_tracked_sizes = rospy.Publisher('/tracked/sizes', Float32, queue_size=10)
        self.pub_tracked_classes = rospy.Publisher('/tracked/classes', Float32, queue_size=10)
        
        self.ego_velocity = 0.0
        self.ego_yaw_rate = 0.0
        self.dt = 0.1  # 기본 시간 간격 (100ms)

    def detections_callback(self, msg):
        detections, size_list, yaw_list, class_list = self.process_point_cloud(msg)
        
        # 수행 시간 측정 시작
        start_time = time.time()
        
        self.tracker.track(detections, self.ego_velocity, self.ego_yaw_rate, self.dt, size_list, yaw_list, class_list)
        
        # 수행 시간 측정 종료
        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo(f"Tracking time for {len(detections)} objects: {elapsed_time:.4f} seconds")
        
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
            self.pub_tracked_vx.publish(obj['vx'])
            self.pub_tracked_vy.publish(obj['vy'])
            self.pub_tracked_vz.publish(obj['vz'])
            if obj['yaw'] is not None:
                self.pub_tracked_yaws.publish(obj['yaw'])
            self.pub_tracked_sizes.publish(obj['size'])
            self.pub_tracked_classes.publish(obj['class_id'])

    def velocity_callback(self, msg):
        self.ego_velocity = msg.linear.x
        self.ego_yaw_rate = msg.angular.z

    def process_point_cloud(self, msg):
        detections = []
        size_list = []
        yaw_list = []
        class_list = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "yaw", "size", "class"), skip_nans=True):
            detections.append(np.array([p[0], p[1], p[2]]))
            yaw_list.append(p[3] if len(p) > 3 else None)
            size_list.append(np.array([p[4], p[5], p[6]]) if len(p) > 6 else None)
            class_list.append(p[7] if len(p) > 7 else None)
        return detections, size_list, yaw_list, class_list

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = AB3DMOTNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
