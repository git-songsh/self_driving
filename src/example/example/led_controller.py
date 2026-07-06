#!/usr/bin/env python3
# encoding: utf-8
# =============================================================================
# led_controller.py  —  LED 제어 노드 v2 (gpiochip 자동 폴백 수정)
# =============================================================================
# v1 → v2 수정사항:
#   1. gpiochip 자동 폴백: 4번 시도 후 실패 시 0번 시도 (Pi 5 호환)
# =============================================================================
#
# 핀 배정 (BCM 번호 기준):
#   GPIO 17 (11번 핀) → 녹색 LED (주행 중 ON)
#   GPIO 27 (13번 핀) → 적색 LED (정지 중 ON)
#   GPIO 22 (15번 핀) → 황색 LED (방향 신호 점멸)
#   GND    ( 6번 핀) → 공통 GND
#
# 구독 토픽:  /led/cmd  (std_msgs/String)
#   "green_on"    : 녹색 ON, 적색 OFF
#   "red_on"      : 적색 ON, 녹색 OFF
#   "yellow_blink": 황색 점멸 시작
#   "yellow_off"  : 황색 점멸 정지
#   "all_blink"   : 전체 LED 점멸 (주차 완료)
#   "all_off"     : 전체 LED OFF
# =============================================================================

import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False

# ── GPIO 핀 번호 (BCM) ──────────────────────────────────────
PIN_GREEN  = 17
PIN_RED    = 27
PIN_YELLOW = 22

# 점멸 주기 (초)
BLINK_INTERVAL     = 0.5
ALL_BLINK_INTERVAL = 0.3


class LEDController(Node):
    def __init__(self):
        super().__init__('led_controller')
        self.get_logger().info('LED Controller v2 시작')

        # ── GPIO 초기화 (gpiochip 자동 폴백) ────────────────
        self._gpio_ok = False
        self._h       = None

        if LGPIO_AVAILABLE:
            # ✅ 수정: Pi 5는 gpiochip4, Pi 4는 gpiochip0 — 순서대로 시도
            for chip_num in [4, 0]:
                try:
                    self._h = lgpio.gpiochip_open(chip_num)
                    for pin in [PIN_GREEN, PIN_RED, PIN_YELLOW]:
                        lgpio.gpio_claim_output(self._h, pin, 0)
                    self._gpio_ok = True
                    self.get_logger().info(f'GPIO gpiochip{chip_num} 초기화 완료')
                    break
                except Exception as e:
                    self.get_logger().warn(f'gpiochip{chip_num} 실패: {e}')

            if not self._gpio_ok:
                self.get_logger().error('모든 gpiochip 초기화 실패 — 시뮬레이션 모드')
        else:
            self.get_logger().warn('lgpio 없음 — 시뮬레이션 모드')

        # ── 상태 변수 ────────────────────────────────────────
        self._blink_thread = None
        self._blink_stop   = threading.Event()
        self._lock         = threading.Lock()

        # 시작 시 적색 ON (정지 상태)
        self._set_red_on()

        # ── ROS2 구독 ────────────────────────────────────────
        self.create_subscription(String, '/led/cmd', self._cmd_callback, 10)
        self.get_logger().info('LED 명령 구독 시작: /led/cmd')

    # =========================================================
    # 명령 콜백
    # =========================================================
    def _cmd_callback(self, msg: String):
        cmd = msg.data.strip()
        self.get_logger().info(f'LED 명령 수신: {cmd}')

        if cmd == 'green_on':
            self._stop_blink()
            self._set_green_on()
        elif cmd == 'red_on':
            self._stop_blink()
            self._set_red_on()
        elif cmd == 'yellow_blink':
            self._stop_blink()
            self._start_blink(pins=[PIN_YELLOW], interval=BLINK_INTERVAL)
        elif cmd == 'yellow_off':
            self._stop_blink()
            self._write(PIN_YELLOW, 0)
        elif cmd == 'all_blink':
            self._stop_blink()
            self._start_blink(
                pins=[PIN_GREEN, PIN_RED, PIN_YELLOW],
                interval=ALL_BLINK_INTERVAL)
        elif cmd == 'all_off':
            self._stop_blink()
            self._all_off()
        else:
            self.get_logger().warn(f'알 수 없는 LED 명령: {cmd}')

    # =========================================================
    # LED 제어 헬퍼
    # =========================================================
    def _write(self, pin: int, value: int):
        if self._gpio_ok and self._h is not None:
            lgpio.gpio_write(self._h, pin, value)

    def _all_off(self):
        for pin in [PIN_GREEN, PIN_RED, PIN_YELLOW]:
            self._write(pin, 0)

    def _set_green_on(self):
        self._write(PIN_GREEN,  1)
        self._write(PIN_RED,    0)
        self._write(PIN_YELLOW, 0)

    def _set_red_on(self):
        self._write(PIN_GREEN,  0)
        self._write(PIN_RED,    1)
        self._write(PIN_YELLOW, 0)

    # =========================================================
    # 점멸 스레드
    # =========================================================
    def _start_blink(self, pins: list, interval: float):
        with self._lock:
            self._blink_stop.clear()
            self._blink_thread = threading.Thread(
                target=self._blink_loop,
                args=(pins, interval),
                daemon=True)
            self._blink_thread.start()

    def _blink_loop(self, pins: list, interval: float):
        state = 0
        while not self._blink_stop.is_set():
            state ^= 1
            for pin in pins:
                self._write(pin, state)
            self._blink_stop.wait(timeout=interval)
        for pin in pins:
            self._write(pin, 0)

    def _stop_blink(self):
        with self._lock:
            if self._blink_thread and self._blink_thread.is_alive():
                self._blink_stop.set()
                self._blink_thread.join(timeout=1.0)
            self._blink_thread = None

    # =========================================================
    # 종료
    # =========================================================
    def destroy_node(self):
        self._stop_blink()
        self._all_off()
        if self._gpio_ok and self._h is not None:
            try:
                lgpio.gpiochip_close(self._h)
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = LEDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()