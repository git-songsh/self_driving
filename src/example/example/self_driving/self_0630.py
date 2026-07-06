#!/usr/bin/env python3
# encoding: utf-8
# =============================================================================
# self_driving.py — hybrid final (통합 완성본)
# 기준:
# 1) 기존 안정 주행 코드의 init/start, LAB line follow, park_action 구조 유지
# 2) 우회전 성공 코드의 FSM right-turn 구조 유지
# 3) 횡단보도 최초 감지 시 2초간 완전 정지 미션 및 6초 중복 방지 락 이식
# 4) main 큐 포화 방지 마진 슬립 리팩토링 적용
# =============================================================================

import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np

import sdk.pid as pid
import sdk.fps as fps
import sdk.common as common

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import String, Bool
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState


# =============================================================================
# FSM 상태
# =============================================================================
class DriveState:
    LINE_FOLLOW = 'LINE_FOLLOW'
    ARROW_SIGNAL = 'ARROW_SIGNAL'
    INTERSECTION = 'INTERSECTION'
    PARKING = 'PARKING'
    DONE = 'DONE'


# =============================================================================
# 튜닝 파라미터
# =============================================================================
NORMAL_SPEED = 0.3
SLOW_DOWN_SPEED = 0.15

# 기존 코드의 일반 코너 처리
LANE_TURN_X = 200
LANE_TURN_ANGULAR_Z = -0.45

# lane_x 안정화 필터
LANE_X_JUMP_FILTER = True
LANE_X_JUMP_THRESHOLD = 80          # 이전 안정 lane_x와 80px 이상 차이나면 후보로 보류
LANE_X_JUMP_CONFIRM_FRAMES = 4      # 4프레임 연속 비슷하게 나오면 새 lane_x로 인정
LANE_X_HOLD_SEC = 0.60              # 짧은 차선 유실은 마지막 정상 lane_x로 버틴다
LANE_X_LOST_SAFE_LINEAR = 0.05       # hold 시간 지나도 못 찾으면 정지.
LANE_X_JUMP_ACCEPT_DIFF = 35        # jump 후보끼리 이 정도 이내면 같은 후보로 본다

# PID
PID_SETPOINT = 130
PID_MIN = -0.10
PID_MAX = 0.10

# right sign 확인 조건
RIGHT_CONFIRM_COUNT = 2          
RIGHT_SCORE_TH = 0.50
RIGHT_MIN_AREA = 2800
RIGHT_MAX_MISS = 8

# 우회전 FSM
ARROW_SIGNAL_WAIT = 0.4           # 황색 점멸/정지 대기. Too long values cause late departures
RIGHT_TURN_DURATION = 1.80        # 우회전 강제 유지 시간
RIGHT_TURN_SPEED = 0.10
RIGHT_TURN_ANGULAR_Z = -0.50      
RIGHT_RECOVER_ANGULAR_Z = -0.20

# 주차
PARK_CONFIRM_COUNT = 3
PARK_MIN_AREA = 300
PARK_REQUIRE_RIGHT_DONE = False
PARK_MIN_SCORE = 0.50
PARK_MIN_CROSSWALK_Y = 180

# 신호등
TRAFFIC_MIN_AREA = 800
TRAFFIC_MIN_SCORE = 0.45

# 횡단보도 감속 및 완전 정지 제어 파라미터 [업데이트 완료]
CROSSWALK_SLOW_Y = 180
CROSSWALK_CONFIRM_COUNT = 4
CROSSWALK_STOP_DURATION = 2.0        # 정지선 앞 완전 정지 대기 시간 (2초) [cite: 303, 308]
CROSSWALK_RETRIGGER_COOLDOWN = 9.0    # 횡단보도 통과 후 6초간 재감지 방지 데드타임 [cite: 304, 308]

# 디버그 로그 주기
DEBUG_LOG_INTERVAL = 0.5
OBJECT_LOG_INTERVAL = 0.5
IMAGE_LOG_INTERVAL = 2.0
LANE_LOST_LOG_INTERVAL = 0.5


class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.6, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE', 'MentorPi_Mecanum')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1
        )
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.led_pub = self.create_publisher(String, '/led/cmd', 10)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        self.create_subscription(Bool, '/ros_robot_controller/button', self.button_callback, 10)

        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.start_yolov5_client = self.create_client(
            Trigger, '/yolov5/start', callback_group=timer_cb_group
        )
        self.stop_yolov5_client = self.create_client(
            Trigger, '/yolov5/stop', callback_group=timer_cb_group
        )

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    # =========================================================================
    # 초기 변수
    # =========================================================================
    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        # 기존 코드 변수 유지
        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False
        self.start_park = False

        self.count_crosswalk = 0
        self.crosswalk_distance = 0
        self.crosswalk_length = 0.1 + 0.3

        self.start_slow_down = False
        self.normal_speed = NORMAL_SPEED
        self.slow_down_speed = SLOW_DOWN_SPEED

        self.traffic_signs_status = None
        self.red_loss_count = 0
        self.traffic_seen_time = 0.0
        self.last_traffic_log_time = 0.0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        # FSM 추가
        self.drive_state = DriveState.LINE_FOLLOW
        self.state_entry_time = time.time()
        self.arrow_direction = None
        self.right_ready = False
        self.right_candidate = None
        self.right_done = False

        # 주차 추가 안정화
        self.park_ready = False
        self.park_done = False
        self.park_candidate = None
        self.last_park_log_time = 0.0

        self.last_debug_log_time = 0.0
        self.last_object_log_time = 0.0
        self.last_image_log_time = 0.0
        self.last_lane_lost_log_time = 0.0
        self.image_count = 0
        self.object_msg_count = 0

        # lane_x 필터 상태
        self.last_valid_lane_x = -1
        self.last_valid_lane_time = 0.0
        self.lane_jump_candidate = None
        self.lane_jump_count = 0
        self.last_lane_filter_log_time = 0.0

        # crosswalk 중복 감지 방지용 제어 변수 [업데이트 완료]
        self.last_crosswalk_slow_time = 0.0
        self.crosswalk_clear_lock = False    # 통과 중 중복 감지 방지 락 플래그 [cite: 309]

    # =========================================================================
    # 안전한 서비스 요청
    # =========================================================================
    def send_request(self, client, msg, timeout_sec=2.0):
        if client is None:
            self.get_logger().warn('[YOLO_SERVICE] client is None')
            return None

        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().warn('[YOLO_SERVICE] service not ready, continue without blocking')
                return None

        future = client.call_async(msg)
        start_time = time.time()
        while rclpy.ok():
            if future.done():
                result = future.result()
                self.get_logger().info(f'[YOLO_SERVICE] response success={getattr(result, "success", None)} message={getattr(result, "message", "")}')
                return result
            if time.time() - start_time > timeout_sec:
                self.get_logger().warn('[YOLO_SERVICE] service call timeout, continue without blocking')
                return None
            time.sleep(0.01)

    # =========================================================================
    # 초기화
    # =========================================================================
    def init_process(self):
        self.timer.cancel()
        self.mecanum_pub.publish(Twist())
        self.get_logger().info('[INIT] init_process started')

        try:
            only_line_follow = self.get_parameter('only_line_follow').value
        except Exception:
            only_line_follow = False

        self.get_logger().info(f'[INIT] only_line_follow={only_line_follow}')
        if not only_line_follow:
            self.get_logger().info('[INIT] request /yolov5/start')
            self.send_request(self.start_yolov5_client, Trigger.Request(), timeout_sec=2.0)

        time.sleep(0.5)

        self.display = True
        self.get_logger().info('[INIT] enter + set_running start')
        self.enter_srv_callback(Trigger.Request(), Trigger.Response())
        request = SetBool.Request()
        request.data = True
        self.set_running_srv_callback(request, SetBool.Response())

        self.drive_state = DriveState.LINE_FOLLOW
        self.state_entry_time = time.time()
        self.led('green_on')

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m[START] hybrid self_driving start: auto LINE_FOLLOW\033[0m')

    def get_node_state(self, request, response):
        response.success = True
        return response

    # =========================================================================
    # ROS 서비스 콜백
    # =========================================================================
    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mself driving enter\033[0m')
        with self.lock:
            self.image_sub = self.create_subscription(
                Image,
                '/ascamera/camera_publisher/rgb0/image',
                self.image_callback,
                1
            )
            self.object_sub = self.create_subscription(
                ObjectsInfo,
                '/yolov5_ros2/object_detect',
                self.get_object_callback,
                1
            )
            self.get_logger().info('[SUB] image=/ascamera/camera_publisher/rgb0/image object=/yolov5_ros2/object_detect')
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mself driving exit\033[0m')
        with self.lock:
            self.mecanum_pub.publish(Twist())
            self.led('all_off')
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mset_running\033[0m')
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
                self.led('red_on')
            else:
                self.led('green_on')
        response.success = True
        response.message = "set_running"
        return response

    def button_callback(self, msg):
        if msg.data and not self.start:
            self.start = True
            self.transition(DriveState.LINE_FOLLOW)

    def shutdown(self, signum=None, frame=None):
        self.is_running = False

    # =========================================================================
    # 콜백
    # =========================================================================
    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        self.image_count += 1
        now = time.time()
        if now - self.last_image_log_time > IMAGE_LOG_INTERVAL:
            self.last_image_log_time = now
            self.get_logger().info(f'[IMAGE] received count={self.image_count} shape={rgb_image.shape} queue={self.image_queue.qsize()}')
        if self.image_queue.full():
            self.image_queue.get()
            self.get_logger().warn('[IMAGE] queue full -> drop oldest frame')
        self.image_queue.put(rgb_image)

    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        self.object_msg_count += 1
        now = time.time()
        if now - self.last_object_log_time > OBJECT_LOG_INTERVAL:
            names = [o.class_name for o in self.objects_info]
            self.get_logger().info(f'[OBJECTS] msg_count={self.object_msg_count} n={len(self.objects_info)} names={names}')
            self.last_object_log_time = now

        if self.objects_info == []:
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
            self.count_right_miss += 1
            if self.count_right_miss > RIGHT_MAX_MISS:
                self.count_right = 0
            self.park_x = -1
            self.park_candidate = None
            return

        min_distance = 0
        saw_right_this_frame = False
        saw_park_this_frame = False
        current_traffic = None
        current_traffic_area = 0

        for obj in self.objects_info:
            class_name = obj.class_name
            x1, y1, x2, y2 = obj.box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = abs(x2 - x1) * abs(y2 - y1)
            score = float(obj.score)

            if class_name == 'crosswalk':
                if cy > min_distance:
                    min_distance = cy

            elif class_name == 'right':
                saw_right_this_frame = True
                self.right_candidate = {
                    'cx': cx, 'cy': cy, 'area': area, 'score': score
                }

                valid_right = (score >= RIGHT_SCORE_TH and area >= RIGHT_MIN_AREA)
                reject_reason = []
                if score < RIGHT_SCORE_TH:
                    reject_reason.append('low_score')
                if area < RIGHT_MIN_AREA:
                    reject_reason.append('small_area')
                if self.right_done:
                    reject_reason.append('right_done')

                self.get_logger().info(
                    f'[RIGHT_CANDIDATE] valid={valid_right} reject={reject_reason} '
                    f'cnt={self.count_right} cx={cx} cy={cy} '
                    f'area={area} score={score:.2f} done={self.right_done}'
                )

                if valid_right and not self.right_done:
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= RIGHT_CONFIRM_COUNT:
                        self.right_ready = True
                        self.turn_right = True
                        self.count_right = 0
                        self.get_logger().info(
                            f'[RIGHT_READY] cx={cx} cy={cy} area={area} score={score:.2f}'
                        )

            elif class_name == 'park':
                saw_park_this_frame = True
                self.park_candidate = {
                    'cx': cx, 'cy': cy, 'area': area, 'score': score,
                    'right_done': self.right_done,
                }

                valid_park = (
                    score >= PARK_MIN_SCORE and
                    area >= PARK_MIN_AREA and
                    (self.right_done or not PARK_REQUIRE_RIGHT_DONE)
                )

                now_park = time.time()
                if now_park - self.last_park_log_time > OBJECT_LOG_INTERVAL:
                    self.last_park_log_time = now_park
                    if not self.right_done and PARK_REQUIRE_RIGHT_DONE:
                        self.get_logger().info(
                            f'[PARK_BLOCKED_BEFORE_RIGHT] cx={cx} cy={cy} area={area} '
                            f'score={score:.2f} right_done={self.right_done}'
                        )
                    else:
                        self.get_logger().info(
                            f'[PARK_CANDIDATE] valid={valid_park} cx={cx} cy={cy} '
                            f'area={area} score={score:.2f} right_done={self.right_done} '
                            f'park_done={self.park_done}'
                        )

                if valid_park:
                    self.park_x = cx
                else:
                    self.park_x = -1

            elif class_name == 'red' or class_name == 'green':
                valid_traffic = (score >= TRAFFIC_MIN_SCORE and area >= TRAFFIC_MIN_AREA)
                if now - self.last_traffic_log_time > OBJECT_LOG_INTERVAL:
                    self.last_traffic_log_time = now
                    self.get_logger().info(
                        f'[TRAFFIC_CANDIDATE] name={class_name} valid={valid_traffic} '
                        f'cx={cx} cy={cy} area={area} score={score:.2f} '
                        f'min_area={TRAFFIC_MIN_AREA}'
                    )
                if valid_traffic and area > current_traffic_area:
                    current_traffic = obj
                    current_traffic_area = area

        if current_traffic is not None:
            self.traffic_signs_status = current_traffic
            self.traffic_seen_time = now
        else:
            if self.traffic_signs_status is not None and now - self.last_traffic_log_time > OBJECT_LOG_INTERVAL:
                self.last_traffic_log_time = now
                self.get_logger().info('[TRAFFIC_CLEAR] no valid red/green in current object msg -> clear stale traffic')
            self.traffic_signs_status = None
            if self.stop and self.start_slow_down:
                self.stop = False
                self.get_logger().info('[TRAFFIC_CLEAR_STOP] clear stale red stop -> resume slow/line follow')

        if not saw_right_this_frame:
            self.count_right_miss += 1
            if self.count_right_miss > RIGHT_MAX_MISS:
                self.count_right = 0

        if not saw_park_this_frame:
            self.park_x = -1
            self.park_candidate = None

        self.crosswalk_distance = min_distance

    # =========================================================================
    # LED / 이동 / 상태 전환
    # =========================================================================
    def led(self, cmd):
        try:
            msg = String()
            msg.data = cmd
            self.led_pub.publish(msg)
        except Exception:
            pass

    def move(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.linear.y = float(linear_y)
        twist.angular.z = float(angular_z)
        self.mecanum_pub.publish(twist)

    def stop_robot(self):
        self.mecanum_pub.publish(Twist())

    def transition(self, new_state):
        old = self.drive_state
        self.drive_state = new_state
        self.state_entry_time = time.time()
        self.get_logger().info(f'\033[1;36m[FSM] {old} -> {new_state}\033[0m')

        if new_state == DriveState.LINE_FOLLOW:
            self.led('green_on')
        elif new_state == DriveState.ARROW_SIGNAL:
            self.led('yellow_blink')
        elif new_state == DriveState.PARKING:
            self.led('red_on')
        elif new_state == DriveState.DONE:
            self.led('all_blink')

    def elapsed(self):
        return time.time() - self.state_entry_time

    # =========================================================================
    # 주차 액션
    # =========================================================================
    def park_action(self):
        
        # [단계 1] 주차 칸 앞으로 3초간 똑바로 서행 전진 (차선 보지 않는 오픈루프)
        twist_forward = Twist()
        twist_forward.linear.x = 0.15 # 안전한 서행 속도 (0.15 m/s)
        twist_forward.angular.z = 0.0                        # 조향 잠금
        self.mecanum_pub.publish(twist_forward)
        time.sleep(3.0)                                      # 3초 유지 
        
        # 전진 완료 후 관성 제거를 위한 잠시 정지
        self.stop_robot()
        time.sleep(0.2)

        if self.machine_type == 'MentorPi_Mecanum':
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38 / 0.2)
        else:
            twist = Twist()
            twist.angular.z = -1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())

            twist = Twist()
            twist.linear.x = 0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.65 / 0.2)
            self.mecanum_pub.publish(Twist())

            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)

        self.mecanum_pub.publish(Twist())
        self.get_logger().info('[PARK_DONE] park_action complete -> DONE')
        self.transition(DriveState.LINE_FOLLOW)
        self.start_park = False

    # =========================================================================
    # lane_x 안정화 필터
    # =========================================================================
    def filter_lane_x(self, raw_lane_x):
        now = time.time()
        if not LANE_X_JUMP_FILTER:
            if raw_lane_x >= 0:
                self.last_valid_lane_x = raw_lane_x
                self.last_valid_lane_time = now
            return raw_lane_x, 'raw'

        if raw_lane_x < 0:
            if self.last_valid_lane_x >= 0 and (now - self.last_valid_lane_time) <= LANE_X_HOLD_SEC:
                return self.last_valid_lane_x, 'hold_last'
            return -1, 'lost'

        if self.last_valid_lane_x < 0:
            self.last_valid_lane_x = raw_lane_x
            self.last_valid_lane_time = now
            self.lane_jump_candidate = None
            self.lane_jump_count = 0
            return raw_lane_x, 'init'

        diff = abs(raw_lane_x - self.last_valid_lane_x)

        if diff <= LANE_X_JUMP_THRESHOLD:
            self.last_valid_lane_x = raw_lane_x
            self.last_valid_lane_time = now
            self.lane_jump_candidate = None
            self.lane_jump_count = 0
            return raw_lane_x, 'stable'

        if self.lane_jump_candidate is None or abs(raw_lane_x - self.lane_jump_candidate) > LANE_X_JUMP_ACCEPT_DIFF:
            self.lane_jump_candidate = raw_lane_x
            self.lane_jump_count = 1
        else:
            self.lane_jump_count += 1

        if self.lane_jump_count >= LANE_X_JUMP_CONFIRM_FRAMES:
            self.last_valid_lane_x = raw_lane_x
            self.last_valid_lane_time = now
            self.lane_jump_candidate = None
            self.lane_jump_count = 0
            return raw_lane_x, 'jump_accept'

        return self.last_valid_lane_x, 'jump_reject'

    # =========================================================================
    # FSM 상태별 처리 (handle_line_follow 전체 리팩토링)
    # =========================================================================
    def handle_line_follow(self, image, result_image):
        binary_image = self.lane_detect.get_binary(image)
        result_image, lane_angle, raw_lane_x = self.lane_detect(binary_image, result_image)
        lane_x, lane_filter_status = self.filter_lane_x(raw_lane_x)

        twist = Twist()
        twist.linear.x = self.normal_speed

        # ---------------------------------------------------------------------
        # 1) right sign 기반 FSM 우회전 (기존 유지)
        # ---------------------------------------------------------------------
        if self.right_ready:
            self.get_logger().info(f'[RIGHT_TRIGGER] enter ARROW_SIGNAL lane_x={lane_x}')
            self.right_ready = False
            self.arrow_direction = 'right'
            self.transition(DriveState.ARROW_SIGNAL)
            self.stop_robot()
            return result_image

        # ---------------------------------------------------------------------
        # 2) [수정 이식] 횡단보도 처음 감지 시 시간 기반 완전 정지 시퀀스
        # ---------------------------------------------------------------------
        now_cw = time.time()
        
        # 중복 방지 락(crosswalk_clear_lock)이 걸려있지 않은 상태에서만 새로 트리거 가능 [cite: 300]
        crosswalk_can_trigger = (
            self.crosswalk_distance >= CROSSWALK_SLOW_Y and
            not self.start_slow_down and
            not self.crosswalk_clear_lock
        )
        
        if crosswalk_can_trigger:
            self.count_crosswalk += 1
            if self.count_crosswalk >= CROSSWALK_CONFIRM_COUNT:
                self.count_crosswalk = 0
                self.start_slow_down = True
                self.count_slow_down = now_cw  # 시퀀스 시작 절대 시각 기록 [cite: 302]
                self.get_logger().info(f'\033[1;33m[CROSSWALK_DETECT] 최초 감지! 정지선 접근 시퀀스 작동\033[0m')
        else:
            self.count_crosswalk = 0

        # 횡단보도 타임 라인 제어 구역 [cite: 300]
        if self.start_slow_down:
            elapsed = now_cw - self.count_slow_down  # FSM과 독립된 전용 경과 시간 계산 [cite: 300]

            if self.traffic_signs_status is not None:
                area = (
                    abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) *
                    abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                )
                traffic_name = self.traffic_signs_status.class_name

                if traffic_name == 'red':
                    self.mecanum_pub.publish(Twist())
                    self.stop = True
                    self.get_logger().info(f'[TRAFFIC_RED_STOP] area={area} cw_y={self.crosswalk_distance}')
                elif traffic_name == 'green':
                    twist.linear.x = self.slow_down_speed
                    if self.stop:
                        self.get_logger().info(f'[TRAFFIC_GREEN_RELEASE] area={area}')
                    self.stop = False
            else:
                if self.stop:
                    self.get_logger().info('[TRAFFIC_NONE_RELEASE] no valid traffic -> stop=False')
                self.stop = False

            # 빨간불 제약이 없다면 고유의 시간 제어 수행
            if not self.stop:
                # [1단계]: 0.0초 ~ 1.2초 구간 -> 정지선 앞까지 서행 감속 접근 [cite: 300, 303]
                if elapsed < 1.2:
                    twist.linear.x = self.slow_down_speed
                    if now_cw - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
                        self.get_logger().info(f'[CW_STAGE 1] 감속 접근 중... elapsed: {elapsed:.2f}s')

                # [2단계]: 1.2초 ~ 3.2초 구간 (딱 2.0초 동안) -> 정지선 앞 완전 정지 브레이크 [cite: 300, 303]
                elif elapsed < (1.2 + CROSSWALK_STOP_DURATION):
                    twist.linear.x = 0.0
                    self.stop = True  # 조향 명령 전송 잠금
                    if now_cw - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
                        self.get_logger().info(f'\033[1;31m[CW_STAGE 2] 정지선 앞 완전 정지 (브레이크 작동 중)\033[0m')

                # [3단계]: 3.2초 이후 -> 미션 완료 및 6초 뮤트 락 스레드 기동 [cite: 300, 304]
                else:
                    self.start_slow_down = False
                    self.stop = False
                    self.crosswalk_clear_lock = True  # 중복 노이즈 무력화 잠금 활성화 [cite: 304]
                    
                    # 6초 유예 뒤 자동으로 락을 해제하는 백그라운드 스레드 가동 [cite: 304]
                    threading.Thread(target=self._reset_crosswalk_lock, daemon=True).start()
                    self.get_logger().info(f'\033[1;32m[CW_STAGE 3] 정지 완료 후 출발! (6초간 재감지 방지 락 가동)\033[0m')
        else:
            self.stop = False
            twist.linear.x = self.normal_speed

        # ---------------------------------------------------------------------
        # 3) 주차 로직 (기존 유지)
        # ---------------------------------------------------------------------
        park_allowed = (self.right_done or not PARK_REQUIRE_RIGHT_DONE)
        if 0 < self.park_x and not park_allowed:
            self.count_park = 0

        if (park_allowed and 0 < self.park_x and self.crosswalk_distance >= PARK_MIN_CROSSWALK_Y and not self.park_done):
            twist.linear.x = self.slow_down_speed
            if not self.start_park:
                self.count_park += 1
                if self.count_park >= PARK_CONFIRM_COUNT:
                    self.get_logger().info('[PARK_START] stop and run park_action')
                    self.mecanum_pub.publish(Twist())
                    self.start_park = True
                    self.stop = True
                    self.park_done = True
                    self.transition(DriveState.PARKING)
                    threading.Thread(target=self.park_action, daemon=True).start()
                    return result_image
        else:
            self.count_park = 0

        # ---------------------------------------------------------------------
        # 4) 기존 line follow 조향 제어 및 단일 모터 발행 구조 통합
        # ---------------------------------------------------------------------
        if lane_x >= 0 and not self.stop:
            if lane_x > LANE_TURN_X:
                self.count_turn += 1
                if self.count_turn > 5 and not self.start_turn:
                    self.start_turn = True
                    self.count_turn = 0
                    self.start_turn_time_stamp = time.time()

                twist.angular.z = LANE_TURN_ANGULAR_Z
            else:
                self.count_turn = 0
                if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                    self.start_turn = False

                if not self.start_turn:
                    self.pid.SetPoint = PID_SETPOINT
                    self.pid.update(lane_x)
                    twist.angular.z = common.set_range(self.pid.output, PID_MIN, PID_MAX)
                else:
                    twist.angular.z = 0.0

            self.mecanum_pub.publish(twist)

        else:
            self.pid.clear()
            if not self.stop:
                safe_twist = Twist()
                safe_twist.linear.x = LANE_X_LOST_SAFE_LINEAR
                safe_twist.angular.z = 0.0
                self.mecanum_pub.publish(safe_twist)

            now_lost = time.time()
            if now_lost - self.last_lane_lost_log_time > LANE_LOST_LOG_INTERVAL:
                self.last_lane_lost_log_time = now_lost
                self.get_logger().warn(f'[LANE_LOST] lane_x={lane_x} filter={lane_filter_status} stop_flag={self.stop}')

        now = time.time()
        if now - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
            self.last_debug_log_time = now
            self.get_logger().info(
                f'[DEBUG_DRIVE] state={self.drive_state} raw_lane_x={raw_lane_x} lane_x={lane_x} filter={lane_filter_status} '
                f'linear={twist.linear.x:.2f} angular={twist.angular.z:.2f} cw_y={self.crosswalk_distance} stop={self.stop}'
            )

        return result_image

    def handle_arrow_signal(self, image, result_image):
        elapsed = self.elapsed()
        if elapsed < ARROW_SIGNAL_WAIT:
            self.stop_robot()
        else:
            self.led('yellow_off')
            self.transition(DriveState.INTERSECTION)
        return result_image

    def handle_intersection(self, image, result_image):
        binary_image = self.lane_detect.get_binary(image)
        result_image, lane_angle, lane_x = self.lane_detect(binary_image, result_image)

        if self.arrow_direction == 'right':
            elapsed = self.elapsed()
            if elapsed < RIGHT_TURN_DURATION:
                self.move(linear_x=RIGHT_TURN_SPEED, angular_z=RIGHT_TURN_ANGULAR_Z)
            else:
                self.right_done = True
                self.turn_right = False
                self.arrow_direction = None
                self.pid.clear()
                self.transition(DriveState.LINE_FOLLOW)
                self.get_logger().info(f'[RIGHT_TURN] done -> LINE_FOLLOW')
        else:
            self.transition(DriveState.LINE_FOLLOW)

        return result_image

    def handle_parking(self, image, result_image):
        self.stop_robot()
        return result_image

    # [추가 이식] 비동기 중복 감지 무력화 타이머 해제 스레드 [cite: 301, 311]
    def _reset_crosswalk_lock(self):
        """횡단보도 정지 미션 완료 후, 차체가 해당 마킹 구역을 완전히 빠져나갈 때까지 대기 후 락 해제"""
        time.sleep(CROSSWALK_RETRIGGER_COOLDOWN)
        self.crosswalk_clear_lock = False
        self.get_logger().info('[CROSSWALK_LOCK] 잠금 해제 완료. 이제 다음 횡단보도 정상 감지가 가능합니다.')

    # =========================================================================
    # 메인 루프 (과부하 방지 슬립 타임 안정화)
    # =========================================================================
    def main(self):
        while self.is_running:
            time_start = time.time()

            try:
                image = self.image_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()

            if self.start:
                if self.drive_state == DriveState.LINE_FOLLOW:
                    result_image = self.handle_line_follow(image, result_image)
                elif self.drive_state == DriveState.ARROW_SIGNAL:
                    result_image = self.handle_arrow_signal(image, result_image)
                elif self.drive_state == DriveState.INTERSECTION:
                    result_image = self.handle_intersection(image, result_image)
                elif self.drive_state == DriveState.PARKING:
                    result_image = self.handle_parking(image, result_image)
                elif self.drive_state == DriveState.DONE:
                    self.stop_robot()

                if self.objects_info:
                    for obj in self.objects_info:
                        class_name = obj.class_name
                        if class_name in self.classes:
                            cls_id = self.classes.index(class_name)
                            color = colors(cls_id, True)
                            plot_one_box(
                                obj.box,
                                result_image,
                                color=color,
                                label="{}:{:.2f}".format(class_name, float(obj.score)),
                            )
            else:
                time.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            # [리팩토링]: 15FPS 카메라의 정체 누적을 막고 튀는 주기를 분산하기 위한 스레드 가용 시간 최적화
            time_compute = time.time() - time_start
            time_d = 0.033 - time_compute
            if time_d > 0:
                time.sleep(time_d)
            else:
                time.sleep(0.002)  # 시스템 과부하 시에도 큐가 마비되지 않도록 타임슬라이스 강제 양보

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()


def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()