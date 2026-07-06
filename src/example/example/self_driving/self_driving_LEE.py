#!/usr/bin/env python3
# encoding: utf-8
# =============================================================================
# self_driving.py — hybrid final
# 기준:
# 1) 기존 안정 주행 코드의 init/start, LAB line follow, park_action 구조 유지
# 2) 우회전 성공 코드의 FSM right-turn 구조만 이식
# 대상 경로:
# /home/ubuntu/ros2_ws/src/example/example/self_driving/self_driving.py
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

from sdk import led

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
# from std_msgs.msg import String, Bool
from std_msgs.msg import String
from ros_robot_controller_msgs.msg import ButtonState

from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState


# =============================================================================
# FSM 상태
# =============================================================================
class DriveState:
    WAIT_START = 'WAIT_START'
    LINE_FOLLOW = 'LINE_FOLLOW'
    ARROW_SIGNAL = 'ARROW_SIGNAL'
    INTERSECTION = 'INTERSECTION'
    PARKING = 'PARKING'
    DONE = 'DONE'


# =============================================================================
# 튜닝 파라미터
# =============================================================================
NORMAL_SPEED = 0.32
SLOW_DOWN_SPEED = 0.15

# 기존 코드의 일반 코너 처리
LANE_TURN_X = 220
LANE_TURN_ANGULAR_Z = -0.70

# lane_x 안정화 필터
# 최근 로그에서 lane_x=-1이 반복되고, 가끔 270~314로 튀면서 angular=-0.45가 발동했다.
# 그래서 순간 점프는 바로 믿지 않고, 여러 프레임 연속일 때만 인정한다.
LANE_X_JUMP_FILTER = True
LANE_X_JUMP_THRESHOLD = 80          # 이전 안정 lane_x와 80px 이상 차이나면 후보로 보류
LANE_X_JUMP_CONFIRM_FRAMES = 4      # 4프레임 연속 비슷하게 나오면 새 lane_x로 인정
LANE_X_HOLD_SEC = 0.35              # 짧은 차선 유실은 마지막 정상 lane_x로 버틴다
LANE_X_LOST_SAFE_LINEAR = 0.02       # hold 시간 지나도 못 찾으면 정지. 필요시 0.05로 아주 느린 직진
LANE_X_JUMP_ACCEPT_DIFF = 35        # jump 후보끼리 이 정도 이내면 같은 후보로 본다

# PID
PID_SETPOINT = 130
PID_MIN = -0.10
PID_MAX = 0.10

# right sign 확인 조건
RIGHT_CONFIRM_COUNT = 2          # 기존 5회 → 2회. 단 score/area 조건도 같이 사용
RIGHT_SCORE_TH = 0.50
RIGHT_MIN_AREA = 2800
RIGHT_MAX_MISS = 8

# 우회전 FSM
ARROW_SIGNAL_WAIT = 1.0 ##          # 황색 점멸/정지 대기. 너무 길면 출발이 늦어짐
RIGHT_TURN_DURATION = 3.20        # 우회전 강제 유지 시간
RIGHT_TURN_SPEED = 0.10
RIGHT_TURN_ANGULAR_Z = -0.50      # working FSM 코드의 -TURN_ANGULAR
# RIGHT_RECOVER_ANGULAR_Z = -0.20

# 주차
PARK_CONFIRM_COUNT = 2
PARK_MIN_AREA = 200
# 주차는 맵 순서상 우회전 완료 후에만 허용한다.
# 로그상 right_done=False 상태에서 park가 먼저 발동해 PARKING->DONE으로 끝나는 문제가 있었음.
PARK_REQUIRE_RIGHT_DONE = False ###
PARK_MIN_SCORE = 0.45
PARK_MIN_CROSSWALK_Y = 180

# 신호등: 최신 프레임에서 일정 면적 이상인 red/green만 유효 처리
# 최근 로그에서 object는 crosswalk만 들어오는데 이전 red(area=484)가 stale 상태로 남아
# [TRAFFIC_RED_STOP]이 계속 반복되며 정지하는 문제가 있었음.
TRAFFIC_MIN_AREA = 270
TRAFFIC_MIN_SCORE = 0.30

# 횡단보도 감속 안정화
# 최근 로그에서 crosswalk_y가 작거나 오래 남아 CROSSWALK_SLOW가 반복됐다.
# 너무 이른/잦은 감속을 막기 위해 y, 연속 프레임, 재발동 쿨다운을 둔다.
CROSSWALK_SLOW_Y = 180
CROSSWALK_CONFIRM_COUNT = 3
CROSSWALK_RETRIGGER_COOLDOWN = 7.5

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
        self.pid = pid.PID(0.63, 0.0, 0.05)
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

        # LED 토픽은 led_controller.py가 없으면 그냥 무시되어도 주행에는 영향 없음
        # self.led_pub = self.create_publisher(String, '/led/cmd', 10)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        # 버튼 토픽은 구독만 한다. 실제 출발은 기존 코드처럼 자동 start.
        # self.create_subscription(Bool, '/ros_robot_controller/button', self.button_callback, 10)
        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback,10)

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

        # crosswalk 반복 감속 방지
        self.last_crosswalk_slow_time = 0.0

    # =========================================================================
    # 안전한 서비스 요청
    # =========================================================================
    def send_request(self, client, msg, timeout_sec=2.0):
        """
        기존 FSM 코드의 문제였던 '서비스 응답 무한 대기'를 막는다.
        /yolov5/start가 응답하지 않아도 자율주행 노드 초기화는 계속 진행한다.
        """
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

        # 기존 코드 방식 유지: 자동으로 YOLO start 시도
        # 단, 응답 안 오면 멈추지 않고 계속 진행
        try:
            only_line_follow = self.get_parameter('only_line_follow').value
        except Exception:
            only_line_follow = False

        self.get_logger().info(f'[INIT] only_line_follow={only_line_follow}')
        if not only_line_follow:
            self.get_logger().info('[INIT] request /yolov5/start')
            self.send_request(self.start_yolov5_client, Trigger.Request(), timeout_sec=2.0)

        time.sleep(0.5)

        # 기존 잘 되는 코드처럼 자동 enter + set_running
        self.display = True
        self.get_logger().info('[INIT] enter + set_running start')
        self.enter_srv_callback(Trigger.Request(), Trigger.Response())
        request = SetBool.Request()
        request.data = True
        self.set_running_srv_callback(request, SetBool.Response())

        # 미션 3: 자동 주행 시작이 아니라 WAIT_START 상태에서 버튼 입력을 기다린다.
        # self.start는 True로 둬야 main loop가 계속 돌면서 WAIT_START에서 stop_robot()을 발행한다.
        self.drive_state = DriveState.WAIT_START
        self.state_entry_time = time.time()
        led.red_on()

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        self.get_logger().info('\033[1;32m[START] hybrid self_driving ready: WAIT_START, press button to LINE_FOLLOW\033[0m')

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
            # self.led('all_off')
            led.all_off()
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
                # self.led('red_on')
                led.red_on()
            else:
                # self.led('green_on')
                led.green_on()
        response.success = True
        response.message = "set_running"
        return response

    def button_callback(self, msg):
        self.get_logger().info(f"[BUTTON] pressed: {msg}")

        if self.drive_state == DriveState.WAIT_START:
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
        """
        YOLO 결과를 저장하고, right/park/crosswalk/traffic 상태를 갱신한다.

        핵심 수정:
        - traffic_signs_status를 매 메시지마다 새로 계산한다.
        - red/green이 현재 프레임에 없으면 None으로 초기화한다.
        - 작은 red/green 박스(area < TRAFFIC_MIN_AREA)는 오검출로 보고 무시한다.

        이유:
        최근 로그에서 object names는 ['crosswalk']인데 이전 red(area=484)가 stale로 남아
        [TRAFFIC_RED_STOP]이 계속 반복되고 stop=True가 풀리지 않았다.
        """
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

        crosswalk_status = "NOT_FOUND"

        for obj in self.objects_info:
            class_name = obj.class_name
            x1, y1, x2, y2 = obj.box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = abs(x2 - x1) * abs(y2 - y1)
            score = float(obj.score)

            if class_name == 'right':
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
            elif class_name == 'crosswalk':
                if score > 0.40 and 160 <= cx <= 490: ####and cy > 350 200
                    if y2 > min_distance:
                        crosswalk_status = "VALID"
                        min_distance = y2
                        self.get_logger().info(
                            f'[CW_CANDIDATE] name={class_name}'
                            f'cx={cx} y2={y2} area={area} score={score:.2f} '
                        )
                else:
                    if crosswalk_status != "VALID":
                        crosswalk_status = "INVALID"


            elif class_name == 'park':
                saw_park_this_frame = True
                self.park_candidate = {
                    'cx': cx, 'cy': cy, 'area': area, 'score': score,
                    'right_done': self.right_done,
                }

                valid_park = (
                    score >= PARK_MIN_SCORE and
                    area >= PARK_MIN_AREA and
                    self.right_done
                    # and
                    # (self.right_done or not PARK_REQUIRE_RIGHT_DONE) and
                    # True
                )

                # now_park = time.time()
                # if now_park - self.last_park_log_time > OBJECT_LOG_INTERVAL:
                #     self.last_park_log_time = now_park
                #     if not self.right_done and PARK_REQUIRE_RIGHT_DONE:
                #         self.get_logger().info(
                #             f'[PARK_BLOCKED_BEFORE_RIGHT] cx={cx} cy={cy} area={area} '
                #             f'score={score:.2f} right_done={self.right_done}'
                #         )
                #     else:
                #         self.get_logger().info(
                #             f'[PARK_CANDIDATE] valid={valid_park} cx={cx} cy={cy} '
                #             f'area={area} score={score:.2f} right_done={self.right_done} '
                #             f'park_done={self.park_done}'
                #         )

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
            # 중요: red/green이 현재 프레임에 없으면 이전 red/green을 반드시 지운다.
            # stale red 때문에 차량이 횡단보도에서 계속 멈추는 것을 방지한다.

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
        self.crosswalk_cw_status = crosswalk_status

    # =========================================================================
    # LED / 이동 / 상태 전환
    # =========================================================================
    # def led(self, cmd):
    #     try:
    #         msg = String()
    #         msg.data = cmd
    #         self.led_pub.publish(msg)
    #     except Exception:
    #         pass

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

        if new_state == DriveState.WAIT_START:
            led.red_on()
        elif new_state == DriveState.LINE_FOLLOW:
            # self.led('green_on')
            led.green_on()
        elif new_state == DriveState.ARROW_SIGNAL:
            # self.led('yellow_blink')
            led.yellow_blink()

        elif new_state == DriveState.INTERSECTION:
            led.yellow_blink()

        # elif new_state == DriveState.PARKING:
        #     # self.led('red_on')
        #     led.red_on()
        elif new_state == DriveState.DONE:
            # self.led('all_blink')
            led.all_blink()

    def elapsed(self):
        return time.time() - self.state_entry_time



    # =========================================================================
    # lane_x 안정화 필터
    # =========================================================================
    def filter_lane_x(self, raw_lane_x):
        """
        raw_lane_x를 그대로 쓰면 최근 로그처럼 120대 주행 중 270~314로 튀면서
        일반 코너링 angular=-0.45가 먼저 발동한다.

        반환:
        - lane_x: 제어에 사용할 값. 없으면 -1
        - status: 로그용 상태 문자열
        """
        now = time.time()

        if not LANE_X_JUMP_FILTER:
            if raw_lane_x >= 0:
                self.last_valid_lane_x = raw_lane_x
                self.last_valid_lane_time = now
            return raw_lane_x, 'raw'

        # 1) 차선 미검출: 아주 짧은 유실은 마지막 정상값으로 버팀
        if raw_lane_x < 0:
            if self.last_valid_lane_x >= 0 and (now - self.last_valid_lane_time) <= LANE_X_HOLD_SEC:
                return self.last_valid_lane_x, 'hold_last'
            return -1, 'lost'

        # 2) 첫 정상값은 그대로 채택
        if self.last_valid_lane_x < 0:
            self.last_valid_lane_x = raw_lane_x
            self.last_valid_lane_time = now
            self.lane_jump_candidate = None
            self.lane_jump_count = 0
            return raw_lane_x, 'init'

        diff = abs(raw_lane_x - self.last_valid_lane_x)

        # 3) 정상 범위 변화는 바로 채택
        if diff <= LANE_X_JUMP_THRESHOLD:
            self.last_valid_lane_x = raw_lane_x
            self.last_valid_lane_time = now
            self.lane_jump_candidate = None
            self.lane_jump_count = 0
            return raw_lane_x, 'stable'

        # 4) 큰 점프는 후보로 보류. 여러 프레임 연속이면 실제 코너로 인정
        if self.lane_jump_candidate is None or abs(raw_lane_x - self.lane_jump_candidate) > LANE_X_JUMP_ACCEPT_DIFF:
            self.lane_jump_candidate = raw_lane_x
            self.lane_jump_count = 1
        else:
            self.lane_jump_count += 1

        if self.lane_jump_count >= LANE_X_JUMP_CONFIRM_FRAMES:
            self.last_valid_lane_x = raw_lane_x
            self.last_valid_lane_time = now
            self.get_logger().info(
                f'[LANE_X_JUMP_ACCEPT] raw={raw_lane_x} last={self.last_valid_lane_x} '
                f'cnt={self.lane_jump_count}/{LANE_X_JUMP_CONFIRM_FRAMES}'
            )
            self.lane_jump_candidate = None
            self.lane_jump_count = 0
            return raw_lane_x, 'jump_accept'

        # 5) 아직 후보 단계면 마지막 정상값으로 제어
        if now - self.last_lane_filter_log_time > DEBUG_LOG_INTERVAL:
            self.last_lane_filter_log_time = now
            self.get_logger().warn(
                f'[LANE_X_JUMP_REJECT] raw={raw_lane_x} last={self.last_valid_lane_x} '
                f'diff={diff:.1f} candidate={self.lane_jump_candidate} '
                f'cnt={self.lane_jump_count}/{LANE_X_JUMP_CONFIRM_FRAMES}'
            )
        return self.last_valid_lane_x, 'jump_reject'

    # =========================================================================
    # FSM 상태별 처리
    # =========================================================================
    def handle_wait_start(self, image, result_image):
        # 버튼 입력 전까지 모터 명령을 0으로 유지한다.
        # 카메라/YOLO/이미지 퍼블리시는 main loop에서 계속 살아 있다.
        self.stop_robot()
        if time.time() - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
            self.last_debug_log_time = time.time()
            self.get_logger().info('[WAIT_START] waiting for physical button /ros_robot_controller/button')
        return result_image

    def handle_line_follow(self, image, result_image):
        binary_image = self.lane_detect.get_binary(image)
        result_image, lane_angle, raw_lane_x = self.lane_detect(binary_image, result_image)
        lane_x, lane_filter_status = self.filter_lane_x(raw_lane_x)

        traffic_name = None

        twist = Twist()
        twist.linear.x = self.normal_speed

        # ---------------------------------------------------------------------
        # 1) right sign 기반 FSM 우회전: 우회전 되는 코드의 장점 이식
        # ---------------------------------------------------------------------
        if self.right_ready:
            self.get_logger().info(f'[RIGHT_TRIGGER] enter ARROW_SIGNAL lane_x={lane_x} raw_lane_x={raw_lane_x} filter={lane_filter_status} cw_y={self.crosswalk_distance} candidate={self.right_candidate}')
            self.right_ready = False
            self.arrow_direction = 'right'
            self.transition(DriveState.ARROW_SIGNAL)
            self.stop_robot()
            return result_image

        # ---------------------------------------------------------------------
        # 2) 기존 코드의 횡단보도/신호등 감속 로직 유지
        #    단, 횡단보도만 봤다고 무조건 영구 정지하지 않도록 기존 시간 기반 통과 유지
        # ---------------------------------------------------------------------
        now_cw = time.time()
        crosswalk_can_trigger = (
            # (CROSSWALK_SLOW_Y-50) <= self.crosswalk_distance <= 290 and
            200 <= self.crosswalk_distance and
            not self.start_slow_down and
            (now_cw - self.last_crosswalk_slow_time) >= CROSSWALK_RETRIGGER_COOLDOWN
        )
        if crosswalk_can_trigger and self.crosswalk_cw_status == "VALID":
            self.count_crosswalk += 1
            if self.count_crosswalk >= CROSSWALK_CONFIRM_COUNT:
                self.count_crosswalk = 0
                self.start_slow_down = True
                self.count_slow_down = time.time()
                self.last_crosswalk_slow_time = time.time()
                self.get_logger().info(
                    f'[CROSSWALK_SLOW] y={self.crosswalk_distance} '
                    f'th={CROSSWALK_SLOW_Y} confirm={CROSSWALK_CONFIRM_COUNT}'
                )
        else:
            if self.crosswalk_distance > 0 and self.crosswalk_distance < CROSSWALK_SLOW_Y:
                if time.time() - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
                    self.get_logger().info(
                        f'[CROSSWALK_IGNORE] y={self.crosswalk_distance} < th={CROSSWALK_SLOW_Y}'
                    )
            self.count_crosswalk = 0

        if self.start_slow_down:
            elapsed = time.time() - self.count_slow_down
            if self.traffic_signs_status is not None:
                area = (
                    abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) *
                    abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                )
                traffic_name = self.traffic_signs_status.class_name

                if traffic_name == 'red':
                    self.stop_robot()
                    self.stop = True
                    self.get_logger().info(f'[TRAFFIC_RED_STOP] area={area} cw_y={self.crosswalk_distance}')
                    return result_image

                elif traffic_name == 'green':
                    twist.linear.x = self.slow_down_speed
                    if self.stop:
                        self.get_logger().info(f'[TRAFFIC_GREEN_RELEASE] area={area} cw_y={self.crosswalk_distance}')
                    self.stop = False
                    self.get_logger().info(f'[TRAFFIC_GREEN_GO] area={area} cw_y={self.crosswalk_distance}')
            else:
                # 현재 프레임에 유효한 red/green이 없으면 정지 상태를 끌고 가지 않는다.
                if self.stop:
                    self.get_logger().info('[TRAFFIC_NONE_RELEASE] no valid traffic -> stop=False')
                self.stop = False

            if not self.stop:
                if lane_x >= 0:
                    self.pid.SetPoint = PID_SETPOINT
                    self.pid.update(lane_x)
                    if self.machine_type != 'MentorPi_Acker':
                        current_angular = common.set_range(self.pid.output, PID_MIN, PID_MAX)
                    else:
                        current_angular = (self.normal_speed * math.tan(common.set_range(self.pid.output, PID_MIN, PID_MAX))) / 0.145
                else:
                    current_angular = 0.0

                # if elapsed < 0.3:
                #     # twist.linear.x = self.slow_down_speed
                #     self.move(self.slow_down_speed, 0.0, current_angular)
                #     return result_image

                if elapsed < 1.4:
                    led.red_on_timed(duration=1.4)
                    # 2단계 : 정지선 바로 앞 완전 정지 브레이크 (1.2초 ~ 3.2초, 딱 2초간 정지)
                    self.move(0.0, 0.0, 0.0)
                    self.get_logger().info(f'[CW_STAGE 2] 정지선 앞 완전 정지! elapsed={elapsed:.2f}s')
                    return result_image
                else:
                    if traffic_name == 'red':
                        self.move(0.0, 0.0, 0.0)
                        return result_image

                    # 3단계 : 2초 정지 완료 후 시퀀스 완전히 종료 및 정상 주행 복귀

                    self.start_slow_down = False
                    self.stop = False
                    self.last_crosswalk_slow_time = time.time()  # 통과 후 쿨타임(락) 기준점 최신화
                    self.get_logger().info('[CW_STAGE 3] 미션 완료 -> 정상 속도 회복 및 6초 락 가동')
                    
                    self.pid.clear()
                    twist.linear.x = self.normal_speed
                    # return result_image

                # if time.time() - self.count_slow_down > self.crosswalk_length / max(twist.linear.x, 0.01):
                #     self.start_slow_down = False
                #     self.stop = False
                #     self.get_logger().info('[CROSSWALK_SLOW_DONE] return normal speed')

        else:
            self.stop = False
            twist.linear.x = self.normal_speed

        # ---------------------------------------------------------------------
        # 3) 주차 로직
        #    로그 분석 결과: right_done=False 상태에서 park가 먼저 발동해
        #    PARKING -> DONE으로 끝났다. 따라서 park는 반드시 우회전 완료 후만 허용한다.
        # ---------------------------------------------------------------------
        park_allowed = True

        if 0 < self.park_x and not park_allowed:
            self.count_park = 0
            if time.time() - self.last_park_log_time > OBJECT_LOG_INTERVAL:
                self.last_park_log_time = time.time()
                self.get_logger().info(
                    f'[PARK_BLOCKED] reason=before_right park_x={self.park_x} '
                    f'cw_y={self.crosswalk_distance} right_done={self.right_done}'
                )

        if (
            park_allowed and
            0 < self.park_x 
            # and
            # self.crosswalk_distance >= PARK_MIN_CROSSWALK_Y and
            # True
        ):
            # twist.linear.x = self.slow_down_speed
            twist.linear.x = 0.30
            

            if not self.start_park:
                self.count_park += 1
                self.get_logger().info(
                    f'[PARK_COUNT] count={self.count_park}/{PARK_CONFIRM_COUNT} '
                    f'park_x={self.park_x} cw_y={self.crosswalk_distance} '
                    f'right_done={self.right_done} candidate={self.park_candidate}'
                )
                if self.count_park >= PARK_CONFIRM_COUNT:
                    self.get_logger().info('[PARK_START] stop and run park_action')
                    
                    self.mecanum_pub.publish(Twist())

                    self.start_park = True
                    self.stop = True
                    self.park_done = True

                    self.transition(DriveState.PARKING)

                    # threading.Thread(target=self.park_action, daemon=True).start()
                    return result_image
        else:
            # park 후보가 없거나, 아직 주차 구간이 아니면 카운트 리셋
            # if self.count_park > 0 and time.time() - self.last_park_log_time > OBJECT_LOG_INTERVAL:
            #     self.get_logger().info(
            #         f'[PARK_RESET] count={self.count_park} park_x={self.park_x} '
            #         f'cw_y={self.crosswalk_distance} right_done={self.right_done}'
            #     )
            self.count_park = 0

        # ---------------------------------------------------------------------
        # 4) 기존 line follow 유지
        # ---------------------------------------------------------------------
        if lane_x >= 0 and not self.stop:
            if lane_x > LANE_TURN_X:
                self.count_turn += 1
                if self.count_turn > 5 and not self.start_turn:
                    self.start_turn = True
                    ##### YELLOW ON ~ 2.5초 동안 
                    led.yellow_blink_timed(duration=3.0)
                    self.count_turn = 0
                    self.start_turn_time_stamp = time.time()

                if self.machine_type != 'MentorPi_Acker':
                    twist.angular.z = LANE_TURN_ANGULAR_Z
                else:
                    twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145

            else:
                self.count_turn = 0
                if time.time() - self.start_turn_time_stamp > 4.0 and self.start_turn:
                    self.start_turn = False

                if not self.start_turn:
                    self.pid.SetPoint = PID_SETPOINT
                    self.pid.update(lane_x)
                    if self.machine_type != 'MentorPi_Acker':
                        twist.angular.z = common.set_range(self.pid.output, PID_MIN, PID_MAX)
                    else:
                        twist.angular.z = (
                            twist.linear.x *
                            math.tan(common.set_range(self.pid.output, PID_MIN, PID_MAX)) /
                            0.145
                        )
                else:
                    if self.machine_type == 'MentorPi_Acker':
                        twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145

            self.mecanum_pub.publish(twist)

        else:

            self.pid.clear()
            # 중요: lane_x=-1이면 이전 angular 명령이 컨트롤러에 남지 않게 안전 명령을 발행한다.
            safe_twist = Twist()
            safe_twist.linear.x = LANE_X_LOST_SAFE_LINEAR
            safe_twist.angular.z = -0.45
            self.mecanum_pub.publish(safe_twist)

            now_lost = time.time()
            if now_lost - self.last_lane_lost_log_time > LANE_LOST_LOG_INTERVAL:
                self.last_lane_lost_log_time = now_lost
                
                if self.stop:
                    self.get_logger().warn(f'[STOP_HOLD] raw_lane_x={raw_lane_x} lane_x={lane_x} filter={lane_filter_status} stop={self.stop} state={self.drive_state} safe_linear={LANE_X_LOST_SAFE_LINEAR:.2f} cw_y={self.crosswalk_distance} traffic={getattr(self.traffic_signs_status, "class_name", None)} right_ready={self.right_ready} right_done={self.right_done}')
                else:
                    self.get_logger().warn(f'[LANE_LOST] raw_lane_x={raw_lane_x} lane_x={lane_x} filter={lane_filter_status} stop={self.stop} state={self.drive_state} safe_linear={LANE_X_LOST_SAFE_LINEAR:.2f} right_ready={self.right_ready} right_done={self.right_done}')

        now = time.time()
        if now - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
            self.last_debug_log_time = now
            self.get_logger().info(
                f'[DEBUG_DRIVE] state={self.drive_state} '
                f'raw_lane_x={raw_lane_x} lane_x={lane_x} filter={lane_filter_status} linear={twist.linear.x:.2f} '
                f'angular={twist.angular.z:.2f} '
                f'cw_y={self.crosswalk_distance} right_done={self.right_done} '
                f'start_turn={self.start_turn} count_turn={self.count_turn} '
                f'traffic={getattr(self.traffic_signs_status, "class_name", None)} stop={self.stop}'
            )

        return result_image

    def handle_arrow_signal(self, image, result_image): # 우회전 진입 전
        # 우회전 성공 코드의 ARROW_SIGNAL 구조 사용하되,
        # 시작 지연을 줄이기 위해 0.4초만 정지
        elapsed = self.elapsed()
        if elapsed < ARROW_SIGNAL_WAIT:

            self.move(0.02, 0.0, 0.0)
            
            if time.time() - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
                self.last_debug_log_time = time.time()
                self.get_logger().info(f'[ARROW_SIGNAL] waiting elapsed={elapsed:.2f}/{ARROW_SIGNAL_WAIT:.2f} direction={self.arrow_direction}')
        else:
            self.get_logger().info(f'[ARROW_SIGNAL] done elapsed={elapsed:.2f} -> INTERSECTION direction={self.arrow_direction}')
            # self.led('yellow_off')
            self.transition(DriveState.INTERSECTION)
        return result_image

    def handle_intersection(self, image, result_image): # 우회전 진입
        binary_image = self.lane_detect.get_binary(image)
        result_image, lane_angle, lane_x = self.lane_detect(binary_image, result_image)

        if self.arrow_direction == 'right':
            elapsed = self.elapsed()
            #### YELLOW-- 우회전하는동안

            if elapsed < RIGHT_TURN_DURATION:
                self.move(
                    linear_x=RIGHT_TURN_SPEED,
                    angular_z=RIGHT_TURN_ANGULAR_Z
                )
                if time.time() - self.last_debug_log_time > DEBUG_LOG_INTERVAL:
                    self.last_debug_log_time = time.time()
                    self.get_logger().info(
                        f'[RIGHT_TURN] elapsed={elapsed:.2f} '
                        f'cmd=({RIGHT_TURN_SPEED:.2f},{RIGHT_TURN_ANGULAR_Z:.2f}) '
                        f'lane_x={lane_x}'
                    )
            elif elapsed < RIGHT_TURN_DURATION + 4.0:
                self.move(0.2, 0.0, 0.0)
                self.get_logger().info('straight')
            else:
                # 우회전은 1회 완료 처리.
                self.right_done = True
                self.turn_right = False
                self.arrow_direction = None
                self.pid.clear()
                
                self.get_logger().info(f'[RIGHT_TURN] done elapsed={elapsed:.2f} lane_x={lane_x} -> LINE_FOLLOW. . . .')
                # self.transition(DriveState.PARKING)
                self.transition(DriveState.LINE_FOLLOW)


        else:
            # go는 현재 대회 우선순위 낮음. 일단 라인 추종으로 복귀.
            self.transition(DriveState.LINE_FOLLOW)

        return result_image

    def handle_parking(self, image, result_image):
        # park_action thread가 실제 주차를 수행한다.
        # 여기서는 DONE 전까지 정지 유지.
        # self.stop_robot()
        # threading.Thread(target=self.park_action, daemon=True).start()
        
        elapsed = self.elapsed()

        # if elapsed < 9.0:
        #     self.move(0.2, 0.0, 0.0)
        #     self.get_logger().info('[PARK!!!!] straight')
            
        if elapsed < (4.0): # 9.0 + 
            self.move(0.0, 0.0, 0.0)
            self.get_logger().info('[PARK!!!!] park ready')

        elif elapsed < (4.0 + 1.9): # 9.0 + 
            if self.machine_type == 'MentorPi_Mecanum':
                self.move(0.0, -0.2, 0.0) # Y축 좌측/우측 이동 속도 명령
        
        else:
            self.stop_robot() # 로봇 완전 정지
            self.get_logger().info('[PARK !!!! ] park action complete')
            self.transition(DriveState.DONE) # 상태 머신 종료
            self.start_park = False

        return result_image

    # =========================================================================
    # 메인 루프
    # =========================================================================
    def main(self):
        while self.is_running:
            time_start = time.time()

            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()

            if self.start:
                if self.drive_state == DriveState.WAIT_START:
                    result_image = self.handle_wait_start(image, result_image)

                elif self.drive_state == DriveState.PARKING:
                    result_image = self.handle_parking(image, result_image)
                
                elif self.drive_state == DriveState.ARROW_SIGNAL:
                    result_image = self.handle_arrow_signal(image, result_image)

                elif self.drive_state == DriveState.INTERSECTION:
                    result_image = self.handle_intersection(image, result_image)

                elif self.drive_state == DriveState.LINE_FOLLOW:
                    result_image = self.handle_line_follow(image, result_image)

                elif self.drive_state == DriveState.DONE:
                    self.move(0.0, 0.0, 0.0)

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

            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

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