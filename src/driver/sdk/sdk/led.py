# #!/usr/bin/python3
# # coding=utf8
# # import Jetson.GPIO as GPIO
# import RPi.GPIO as GPIO

# LED_PIN = 24  # LED对应引脚号 (pin number corresponding to LED)

# mode = GPIO.getmode()
# if mode == 1 or mode is None:  # 是否已经设置引脚编码(whether the pin code is set)
#     GPIO.setmode(GPIO.BCM)  # 设为BCM编码(set as BCM code)

# GPIO.setwarnings(False)  # 关闭警告打印(close alarm print)

# GPIO.setup(LED_PIN, GPIO.OUT)  # 设置引脚为输出模式(set pin as output mode)

# def on():
#     GPIO.output(LED_PIN, 0)

# def off():
#     GPIO.output(LED_PIN, 1)

# def set(new_state):
#     GPIO.output(LED_PIN, new_state)

# if __name__ == "__main__":
#     import time
#     while True:
#         on()
#         time.sleep(1)
#         off()
#         time.sleep(1)


import RPi.GPIO as GPIO
import threading
import time

PIN_GREEN = 23
PIN_RED = 24
PIN_YELLOW = 25
ALL_PINS = [PIN_GREEN, PIN_RED, PIN_YELLOW]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in ALL_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)

_blink_stop_event = None
_blink_thread = None
current_mode = None
mode_lock = threading.Lock()


def stop_blink():
    global _blink_stop_event, _blink_thread
    if _blink_stop_event:
        _blink_stop_event.set()  
    _blink_stop_event = None
    _blink_thread = None


def start_blink(pins, interval=0.3):
    global _blink_stop_event, _blink_thread
    stop_blink()
    ev = threading.Event()
    _blink_stop_event = ev

    def worker():
        state = 1
        while not ev.is_set():
            for p in pins:
                GPIO.output(p, state)
            state = 1 - state
            ev.wait(interval)   # set되면 즉시 깨어남

    _blink_thread = threading.Thread(target=worker, daemon=True)
    _blink_thread.start()


def set_mode(name, fn):
    global current_mode
    with mode_lock:
        if current_mode == name:
            return
        current_mode = name
        fn()


def green_on():
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        GPIO.output(PIN_GREEN, 1)
    set_mode("green_on", _do)


def yellow_blink():
    def _do():
        stop_blink()
        GPIO.output(PIN_RED, 0)
        GPIO.output(PIN_GREEN, 1)   # 회전 중에도 이동 중이므로 초록 유지
        start_blink([PIN_YELLOW], interval=0.3)
    set_mode("yellow_blink", _do)


def yellow_blink_timed(duration=2.5):
    # yellow_blink와 동일하되 duration(초) 뒤 노랑 점멸을 자동으로 멈춘다
    def _do():
        stop_blink()
        GPIO.output(PIN_RED, 0)
        GPIO.output(PIN_GREEN, 1)   # 이동 중이므로 초록 유지
        start_blink([PIN_YELLOW], interval=0.3)

        def _auto_stop():
            global current_mode
            time.sleep(duration)
            stop_blink()
            GPIO.output(PIN_YELLOW, 0)
            current_mode = "green_on"   # 초록 점등 상태로 남으므로 재호출 가능

        threading.Thread(target=_auto_stop, daemon=True).start()
    set_mode("yellow_blink_timed", _do)


def red_on():
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        GPIO.output(PIN_RED, 1)
    set_mode("red_on", _do)


def red_on_timed(duration=1.5):
    # 자동 점등(1.5초), 점멸 x, 횡단보도 대기용
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        GPIO.output(PIN_RED, 1)

        def _auto_off():
            global current_mode
            time.sleep(duration)
            GPIO.output(PIN_RED, 0)
            GPIO.output(PIN_GREEN, 1)
            current_mode = "green_on"   # 초록 점등 상태로 남으므로 재호출 가능

        threading.Thread(target=_auto_off, daemon=True).start()
    set_mode("red_on_timed", _do)


def all_blink():
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        start_blink(ALL_PINS, interval=0.3)
    set_mode("all_blink", _do)


def all_off():
    global current_mode
    with mode_lock:
        current_mode = None
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)


def cleanup():
    all_off()
    GPIO.cleanup()

# led와 관련된 주행 규정
# 로봇이 움직 땐 녹색 led는 on, 빨간색은 off 멈췄을 땐 녹색 off, 빨간색 on해야한다
# led control을 빵판을 사용해서 라즈파이 <-> host 통신으로 구현
# 우회전 깜빡이 구현