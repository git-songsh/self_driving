#!/usr/bin/env python3
# encoding: utf-8
# autonomous driving (ROS 2)

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
from rclpy.node import Node
import sdk.common as common
from gpiozero import LED
from math import atan2, pi
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import (
    BuzzerState,
    SetPWMServoState,
    PWMServoState,
    ButtonState,
    RGBState,
    RGBStates,
)


class SelfDrivingNode(Node):
    def __init__(self, name: str):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = name

        # --- Flags / State ---
        self.is_running = True
        self.is_start = False
        self.running = False
        self.start = False
        self.wait = True
        self.stop = False
        self.turn = False
        self.detect = True
        self.turn_right = False
        self.real_turn_right = False
        self.go_cw = True
        self.detected_cw = False
        self.detected_go = False
        self.detected_right = False
        self.detected_park = False
        self.traffic_signs_status = None
        self.count_turn = 0
        self.stop_turn = True
        self.stop_go = False
        self.start_turn = False
        self.start_count = 0
        self.turn_count = 0
        self.turn_finish = True
        self.go_finish = True
        self.wait_can_finish = True
        self.button_pressed = False
        self.depth_detect = True
        self.max_cw_dist = False

        # --- Numeric state ---
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get("MACHINE_TYPE")
        self.get_logger().info(
            f"\033[1;31mself.machine_type: {self.machine_type}\033[0m")
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # Distances (mm)
        self.cw_distance = 0
        self.right_distance = 0
        self.sign_distance = 0
        self.fence_distance = 0
        self.park_distance = 0

        # Odom pose
        self.position_x = 0.0
        self.position_y = 0.0
        self.yaw = 0.0
        self.degree = 0.0
        self.basis_start_point_x = 0.0
        self.basis_start_point_y = 0.0
        self.basis_turn_point = 0.0
        self.start_dist = 0

        # Tunables
        self.go_linear_x = 0.5
        self.global_mul = 0.55
        self.turn_angular_z = -0.7
        self.line_angular_z = 0.15
        self.park_turn_angular_z = -0.7
        self.park_linear_y = -0.35
        self.stop_time = time.time()

        # LEDs
        self.publisher_ = self.create_publisher(
            RGBStates, "/ros_robot_controller/set_rgb", 10)
        self.left_yellow_led = LED(16)
        self.red_led = LED(12)
        self.green_led = LED(24)
        self.right_yellow_led = LED(23)
        self.led_time = time.time()
        self.led1_color = (0, 0, 0)
        self.led2_color = (0, 0, 0)
        self.led_colors = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "off": (0, 0, 0),
        }

        # Depth image (uint16 mm, with inpaint applied on zeros)
        self.depth = None

        # --- ROS pubs/subs ---
        self.mecanum_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, "ros_robot_controller/pwm_servo/set_state", 1
        )
        self.result_publisher = self.create_publisher(
            Image, "~/image_result", 1)

        self.odom_subscriber = self.create_subscription(
            Odometry, "odom", self.odom_callback, 10
        )
        self.depth_subscriber = self.create_subscription(
            Image,
            "/ascamera/camera_publisher/depth0/image_raw",
            self.depth_callback,
            1,
        )
        self.create_subscription(
            ButtonState, "/ros_robot_controller/button", self.button_callback, 10
        )

        # YOLO service clients
        timer_cb_group = ReentrantCallbackGroup()
        self.start_yolov5_client = self.create_client(
            Trigger, "/yolov5/start", callback_group=timer_cb_group
        )
        self.stop_yolov5_client = self.create_client(
            Trigger, "/yolov5/stop", callback_group=timer_cb_group
        )
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client.wait_for_service()

        # Services
        self.create_service(Trigger, "~/enter", self.enter_srv_callback)
        self.create_service(Trigger, "~/exit", self.exit_srv_callback)
        self.create_service(SetBool, "~/set_running",
                            self.set_running_srv_callback)
        self.timer = self.create_timer(
            0.0, self.init_process, callback_group=timer_cb_group)

        self.get_logger().info("ButtonPressReceiver node started")
        self.get_logger().info("RGB Controller Node has been started.")

    # ---------------------- Utility helpers ----------------------
    def _rgb_publish(self):
        msg = RGBStates()
        msg.states = [
            RGBState(
                index=1, red=self.led1_color[0], green=self.led1_color[1], blue=self.led1_color[2]),
            RGBState(
                index=2, red=self.led2_color[0], green=self.led2_color[1], blue=self.led2_color[2]),
        ]
        self.publisher_.publish(msg)

    def set_leds(self, left: str | tuple, right: str | tuple):
        self.led1_color = self.led_colors.get(
            left, left) if isinstance(left, str) else left
        self.led2_color = self.led_colors.get(
            right, right) if isinstance(right, str) else right
        self._rgb_publish()

    def set_gpio_leds(self, green=None, red=None, left=None, right=None):
        # pass True to turn on, False to turn off, 'blink' to blink at 0.25s
        def _apply(led_obj, val):
            if val is None:
                return
            if val == "blink":
                led_obj.blink(0.25)
            elif val:
                led_obj.on()
            else:
                led_obj.off()

        _apply(self.green_led, green)
        _apply(self.red_led, red)
        _apply(self.left_yellow_led, left)
        _apply(self.right_yellow_led, right)

    def publish_twist(self, x=0.0, y=0.0, z=0.0):
        t = Twist()
        t.linear.x = float(x)
        t.linear.y = float(y)
        t.angular.z = float(z)
        self.mecanum_pub.publish(t)
        return t

    @staticmethod
    def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
        # minimal absolute difference in degrees
        return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)

    def _odom_delta_mm(self) -> float:
        return max(
            abs(self.position_x - self.basis_start_point_x),
            abs(self.position_y - self.basis_start_point_y),
        ) * 1000.0

    # ---------------------- Init / callbacks ----------------------
    def init_process(self):
        self.timer.cancel()
        self.publish_twist(0, 0, 0)
        if not self.get_parameter("only_line_follow").value:
            self._yolo_start_sync()
        time.sleep(1)

        self.display = True
        self.enter_srv_callback(Trigger.Request(), Trigger.Response())
        self.set_running_srv_callback(
            SetBool.Request(data=True), SetBool.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32mstart\033[0m")

    def depth_callback(self, depth_msg: Image):
        if not self.depth_detect:
            return
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        depth_uint16 = depth.astype(np.uint16)
        mask = (depth_uint16 == 0).astype("uint8")
        self.depth = cv2.inpaint(depth_uint16, mask, 2, cv2.INPAINT_TELEA)
        self.depth_detect = False

    def button_callback(self, _msg: ButtonState):
        self.is_start = False
        self.button_pressed = True
        self.get_logger().info("[Button] pressed")

    def image_callback(self, ros_image: Image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    def odom_callback(self, msg: Odometry):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw only
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = atan2(siny_cosp, cosy_cosp)
        self.degree = self.yaw * 180.0 / pi

    # ---------------------- Services ----------------------
    def get_node_state(self, _req, resp):
        resp.success = True
        return resp

    def enter_srv_callback(self, _req, resp):
        self.get_logger().info("\033[1;32mself driving enter\033[0m")
        with self.lock:
            self.running = False
            # image / detection subscriptions (store to fields)
            self.image_sub = self.create_subscription(
                Image, "/ascamera/camera_publisher/rgb0/image", self.image_callback, 1
            )
            self.object_sub = self.create_subscription(
                ObjectsInfo, "/yolov5_ros2/object_detect", self.get_object_callback, 1
            )
            self.publish_twist(0, 0, 0)
            self.enter = True
        resp.success = True
        resp.message = "enter"
        return resp

    def exit_srv_callback(self, _req, resp):
        self.get_logger().info("\033[1;32mself driving exit\033[0m")
        with self.lock:
            self.publish_twist(0, 0, 0)
        # reset state (lightweight)
        self.__init__(self.name)
        resp.success = True
        resp.message = "exit"
        return resp

    def set_running_srv_callback(self, req: SetBool.Request, resp: SetBool.Response):
        self.get_logger().info("\033[1;32mset_running\033[0m")
        with self.lock:
            self.running = bool(req.data)
            if not self.running:
                self.publish_twist(0, 0, 0)
        resp.success = True
        resp.message = "set_running"
        return resp

    # ---------------------- YOLO control ----------------------
    def _yolo_start_sync(self):
        self.wait_can_finish = False
        self.detect = True
        try:
            result = self.start_yolov5_client.call(Trigger.Request())
            self.get_logger().info(
                f"[START] 응답: {getattr(result, 'message', '')}")
        except Exception as e:
            self.get_logger().error(f"[START] 서비스 호출 실패: {e}")
        time.sleep(0.3)
        self.depth_detect = True

    def call_start(self):
        self.wait_can_finish = False
        self.detect = True
        fut = self.start_yolov5_client.call_async(Trigger.Request())
        fut.add_done_callback(self._on_start_response)
        time.sleep(0.3)
        self.depth_detect = True

    def _on_start_response(self, future):
        try:
            result = future.result()
            self.get_logger().info(
                f"[START] 응답: {getattr(result, 'message', '')}")
        except Exception as e:
            self.get_logger().error(f"[START] 서비스 호출 실패: {e}")

    def call_stop(self):
        self.detect = False
        self.depth_detect = False
        fut = self.stop_yolov5_client.call_async(Trigger.Request())
        fut.add_done_callback(self._on_stop_response)

    def _on_stop_response(self, future):
        try:
            result = future.result()
            self.get_logger().info(
                f"[STOP] 응답: {getattr(result, 'message', '')}")
        except Exception as e:
            self.get_logger().error(f"[STOP] 서비스 호출 실패: {e}")

    # ---------------------- Behavior helpers ----------------------
    def _apply_line_follow(self, left_lane_x: int, twist: Twist):
        self.pid.SetPoint = 75
        self.pid.update(left_lane_x)
        if self.machine_type != "MentorPi_Acker":
            twist.angular.z = common.set_range(
                self.pid.output, -self.line_angular_z, self.line_angular_z)
        else:
            twist.angular.z = twist.linear.x * math.tan(
                common.set_range(self.pid.output, -0.1, 0.1)
            ) / 0.145

    def _finish_turn_if_reached(self, threshold_deg: float) -> bool:
        if self._angle_diff_deg(self.basis_turn_point, self.degree) > threshold_deg:
            self.get_logger().info("turn was finished~~~~~~~~~~~~~~")
            self.turn_finish = True
            self.publish_twist(0, 0, 0)
            return True
        return False

    def _led_flash_yellow_right(self):
        # blink right, others off
        if time.time() - self.led_time > 0.25:
            self.led2_color = self.led_colors["off"] if self.led2_color == self.led_colors[
                "yellow"] else self.led_colors["yellow"]
            self.led1_color = self.led_colors["off"]
            self._rgb_publish()
            self.led_time = time.time()

    def _led_flash_white_both_idle(self):
        if time.time() - self.led_time > 0.25:
            if self.led1_color == self.led_colors["white"]:
                self.set_leds("off", "off")
            else:
                self.set_leds("white", "white")
            self.led_time = time.time()

    # ---------------------- Main loop ----------------------
    def main(self):
        cw_count = 0

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
            if self.running:
                binary_image = self.lane_detect.get_binary(image)
                twist = Twist()

                if self.is_start:
                    result_image, left_lane_x, _ = self.lane_detect(
                        binary_image, image.copy())
                    self.get_logger().info(
                        f"\033[1;32mleft_lane_x: {left_lane_x}\033[0m")

                    # ---------------- state: WAIT ----------------
                    if self.go_finish and self.turn_finish:
                        if self.wait:
                            if time.time() - self.stop_time > 1.0 and self.wait_can_finish:
                                # if self.wait_can_finish:
                                if self.traffic_signs_status != "red":
                                    self.wait = False
                                    self.start = True
                                    # LEDs
                                    self.set_gpio_leds(
                                        green=True, red=False, left=False, right=False)
                                    self.set_leds("green", "green")
                                    self.get_logger().info(
                                        "\033[1;32mwait is done call_stop\033[0m")
                                    self.call_stop()
                                    self.go_finish = False
                        elif self.start:
                            self.start_count = 0
                            if self.detected_park:
                                self.stop = True
                                self.set_leds("off", "off")
                                self.stop_time = time.time()
                                self.start = False
                                self.go_finish = False
                            elif self.turn_right:
                                self.start = False
                                self.turn = True
                                self.set_gpio_leds(
                                    green=False, red=False, left=False, right="blink")
                                self.set_leds("red", "red")
                                self.turn_finish = False
                            else:
                                # default -> back to WAIT + RED
                                self.stop_time = time.time()
                                self.wait = True
                                self.set_gpio_leds(
                                    green=False, red=True, left=False, right=False)
                                self.set_leds("red", "red")
                                self.start = False
                        elif self.turn:
                            self.turn_right = False
                            self.turn_count = 0
                            self.wait = True
                            self.set_gpio_leds(
                                green=False, red=True, left=False, right=False)
                            self.set_leds("red", "red")
                            self.turn = False
                        elif self.stop:
                            self.stop = False

                    # ---------------- state handlers ----------------
                    if self.wait:
                        self.get_logger().info(
                            "\033[1;31mstate: **wait**\033[0m")
                        twist.linear.x = 0.0
                        if left_lane_x == -1 and self.traffic_signs_status != 'red':
                            twist.angular.z = 0.15
                            self.mecanum_pub.publish(twist)
                        else:
                            self.publish_twist(0, 0, 0)
                        # self.publish_twist(0, 0, 0)
                        self._rgb_publish()
                        self.get_logger().info(
                            "\033[1;32mturn is done call_start\033[0m")
                        time.sleep(0.4)
                        self.call_start()
                        continue

                    if self.start:
                        self.get_logger().info(
                            "\033[1;31mstate: **start**\033[0m")
                        twist.linear.x = self.go_linear_x
                        if self.start_count == 0:
                            self.get_logger().info(
                                f"\033[1;31mdetected right: {self.detected_right}, detected: {self.detected_cw}\033[0m"
                            )
                            if self.real_turn_right:
                                self.go_cw = False
                                self.turn_right = True
                                self.start_dist = self.cw_distance
                                self.mul = 1.7 + self.global_mul
                                self.get_logger().info(
                                    f"\033[1;31m3. cross_walk distance: {self.start_dist}\033[0m"
                                )
                            elif self.detected_right and self.detected_cw:
                                self.go_cw = False
                                self.real_turn_right = True
                                self.start_dist = self.cw_distance
                                self.mul = 2.1 + self.global_mul
                                self.max_cw_dist = True
                                self.get_logger().info(
                                    f"\033[1;31m2. cross_walk distance: {self.start_dist}\033[0m"
                                )
                            elif self.detected_park:
                                self.start_dist = self.park_distance
                                self.mul = 1.15 + self.global_mul
                                self.get_logger().info(
                                    f"\033[1;31mpark distance: {self.start_dist}\033[0m"
                                )
                            elif self.detected_cw and self.go_cw:
                                self.go_cw = False
                                self.start_dist = self.cw_distance
                                if cw_count < 1:
                                    self.mul = 1.8 + self.global_mul
                                elif cw_count < 2:
                                    self.mul = 2.2 + self.global_mul
                                elif cw_count < 3:
                                    self.mul = 1.75 + self.global_mul
                                elif cw_count < 4:
                                    self.mul = 0.8 + self.global_mul
                                else:
                                    self.mul = 1.3 + self.global_mul
                                cw_count += 1
                                self.get_logger().info(
                                    f"\033[1;31mcross_walk distance: {self.start_dist}\033[0m"
                                )
                            elif self.traffic_signs_status != "red":
                                self.go_cw = True
                                self.turn_right = True
                                self.start_dist = self.fence_distance
                                self.mul = -0.5 + self.global_mul
                                self.get_logger().info(
                                    f"\033[1;31mfence distance: {self.start_dist}\033[0m"
                                )
                            else:
                                self.get_logger().info("RED-RED-RED-RED-RED-RED-RED-RED")
                                self.call_start()
                                continue

                            self.start_count += 1
                            self.basis_start_point_x, self.basis_start_point_y = (
                                self.position_x,
                                self.position_y,
                            )

                        self.get_logger().info(
                            f"\033[1;31modom: {self._odom_delta_mm()}, dist: {self.start_dist}\033[0m"
                        )
                        if self._odom_delta_mm() > self.start_dist - (200 * self.mul):
                            self.get_logger().info(
                                "\033[1;31m**go finish**\033[0m")
                            self.go_finish = True
                            self.publish_twist(0, 0, 0)
                            continue
                        if left_lane_x >= 0 and not self.stop:
                            self._apply_line_follow(left_lane_x, twist)

                    if self.turn:
                        self.pid.clear()
                        self.get_logger().info(
                            "\033[1;31mstate: **turn**\033[0m")
                        self._led_flash_yellow_right()
                        twist.linear.x = 0.0
                        twist.angular.z = self.turn_angular_z
                        if self.turn_count == 0:
                            self.turn_count += 1
                            self.basis_turn_point = self.degree
                        self.get_logger().info(
                            f"\033[1;31minitial degree: {self.basis_turn_point}, cur degree: {self.degree}\033[0m"
                        )
                        if self.real_turn_right:
                            self.real_turn_right = False
                            if self._finish_turn_if_reached(82):
                                continue
                        else:
                            if self._finish_turn_if_reached(80):
                                continue

                    if self.stop:
                        self.get_logger().info(
                            "\033[1;31mstate: **stop**\033[0m")
                        # if self.stop_turn:
                        #     twist.linear.x = 0.0
                        #     twist.angular.z = self.park_turn_angular_z
                        #     if self.turn_count == 0:
                        #         self.turn_count += 1
                        #         self.basis_turn_point = self.degree
                        #     self.get_logger().info(
                        #         f"\033[1;31minitial degree: {self.basis_turn_point}, cur degree: {self.degree}\033[0m"
                        #     )
                        #     if self._finish_turn_if_reached(80):
                        #         self.stop_go = True
                        #         self.stop_turn = False
                        #         self.stop_time = time.time()
                        # elif self.stop_go:
                        #     if time.time() - self.stop_time < 1:
                        #         twist.linear.x = 0.5
                        #         twist.angular.z = 0.0
                        #     else:
                        #         self.stop_go = False
                        # else:
                        #     self.publish_twist(0, 0, 0)
                        self.publish_twist(0, -1.0, 0)
                        time.sleep(0.7)
                        self.publish_twist(0, 0, 0)
                        self.is_start = False
                        self.set_gpio_leds(
                            green="blink", red="blink", left="blink", right="blink")

                    self.get_logger().info(
                        f"\033[1;32mx: {twist.linear.x}, y: {twist.linear.y}, z: {twist.angular.z}\033[0m"
                    )
                    self.mecanum_pub.publish(twist)
                    self._rgb_publish()
                else:
                    self._led_flash_white_both_idle()
                    self.publish_twist(0, 0, 0)

            else:
                time.sleep(0.01)
                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        if class_name in self.classes:
                            cls_id = self.classes.index(class_name)
                            color = self.colors(cls_id, True)
                            plot_one_box(box, result_image, color=color,
                                        label=f"{class_name}:{i.score:.2f}")

            # --- Visualization ---
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)
            self.result_publisher.publish(
                self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            # pacing
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.publish_twist(0, 0, 0)
        rclpy.shutdown()

    def init_object_status(self):
        # Reset defaults
        self.traffic_signs_status = None
        self.cw_distance = -1 if not self.max_cw_dist else 0
        self.right_distance = -1
        self.sign_distance = -1
        self.park_distance = -1
        self.detected_cw = False
        self.detected_go = False
        self.detected_right = False
        self.detected_park = False

    # ---------------------- Detection callback ----------------------
    def get_object_callback(self, msg: ObjectsInfo):
        self.objects_info = msg.objects
        if not self.detect:
            return

        self.init_object_status()

        if not self.objects_info:
            return

        if self.depth is None:
            # Depth not available yet; still record classes but skip distance logic
            for i in self.objects_info:
                class_name = i.class_name
                if class_name == "crosswalk":
                    self.detected_cw = True
                elif class_name == "go":
                    self.detected_go = True
                elif class_name == "right":
                    self.detected_right = True
                elif class_name == "park":
                    self.detected_park = True
                elif class_name == "green":
                    self.traffic_signs_status = "green"
                    self.is_start = True
                elif class_name == "red":
                    self.traffic_signs_status = "red"
            self.wait_can_finish = True
            self.get_logger().info(
                "\033[1;32mdetect something (no depth)!!!!\033[0m")
            return

        # With depth
        for i in self.objects_info:
            class_name = i.class_name
            center = (int((i.box[0] + i.box[2]) / 2),
                      int((i.box[1] + i.box[3]) / 2))
            obj_distance = int(self.depth[center[1], center[0]])  # mm
            self.get_logger().info(
                f"\033[1;32mdetected class = {class_name}, distance: {obj_distance}\033[0m"
            )

            if class_name == "crosswalk":
                self.detected_cw = True
                if self.max_cw_dist:
                    if obj_distance > self.cw_distance:
                        self.cw_distance = obj_distance
                else:
                    if self.cw_distance == -1 or obj_distance < self.cw_distance:
                        self.cw_distance = obj_distance
            elif class_name == "go":
                self.detected_go = True
                self.sign_distance = obj_distance
            elif class_name == "right":
                self.detected_right = True
                self.right_distance = obj_distance
                self.sign_distance = obj_distance
            elif class_name == "park":
                self.detected_park = True
                self.park_distance = obj_distance
            elif class_name == "red":
                if obj_distance < 1100:
                    self.traffic_signs_status = "red"
                    self.sign_distance = obj_distance
            elif class_name == "green":
                self.traffic_signs_status = "green"
                self.sign_distance = obj_distance
                self.is_start = True
            elif class_name == "floor_right":
                self.fence_distance = obj_distance

        self.wait_can_finish = True
        self.get_logger().info("\033[1;32mdetect something!!!!\033[0m")


# ---------------------- Entrypoint ----------------------

def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()