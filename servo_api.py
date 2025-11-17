#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_api.py  —  点动版（小步长）

每次 step_yaw / step_pitch：
- 只偏离中点一小段时间，然后立刻回中点
- 防止连续旋转、线缠绕
- 通过 DELTA_US 和 STEP_MS_* 控制“每一步挪多少”
"""

import time
import pigpio

# ====== 硬件引脚，根据你之前的 wiring 改 ======
PIN_YAW   = 17   # 水平方向舵机
PIN_PITCH = 27   # 垂直方向舵机
LASER_PIN = 22   # 激光 MOSFET

# ====== 舵机参数（这里我已经帮你调小了步长） ======
NEUTRAL_US = 1500    # 你的停止脉宽（如果你之前校过什么 1492/1510，可以换成那个）

# 每步离中点偏移的幅度（越小每步越“细”；太小可能不动）
DELTA_US   = 60      # 原来可能是 120，现在砍半

# 每步持续时间（毫秒）（越小每步时间越短）
STEP_MS_YAW   = 20   # 原来 40，改成 20
STEP_MS_PITCH = 20

class ServoAPI:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon 未运行，请先: sudo pigpiod")

        # 舵机初始化到中点，激光关闭
        self.pi.set_servo_pulsewidth(PIN_YAW,   NEUTRAL_US)
        self.pi.set_servo_pulsewidth(PIN_PITCH, NEUTRAL_US)

        self.pi.set_mode(LASER_PIN, pigpio.OUTPUT)
        self.pi.write(LASER_PIN, 0)

    # 内部通用“点动一步”
    def _step_axis(self, pin, dir_sign, step_ms, delta_us):
        """
        dir_sign: +1 / -1
        step_ms : 这一小步给多久（毫秒）
        delta_us: 离中点偏移多少微秒
        """
        if dir_sign == 0:
            return
        # 偏离中点
        pulse = NEUTRAL_US + int(dir_sign * delta_us)
        self.pi.set_servo_pulsewidth(pin, pulse)
        # 保持一小会儿
        time.sleep(step_ms / 1000.0)
        # 回到中点，防止连续转圈
        self.pi.set_servo_pulsewidth(pin, NEUTRAL_US)

    # 对外接口：水平 / 垂直 各一点
    def step_yaw(self, dir_sign: int):
        """
        水平点动一步
        dir_sign: +1 向右, -1 向左（具体方向看你舵机安装）
        """
        self._step_axis(PIN_YAW, dir_sign, STEP_MS_YAW, DELTA_US)

    def step_pitch(self, dir_sign: int):
        """
        垂直点动一步
        dir_sign: +1 向下, -1 向上（如方向相反就把 +1/-1 注释改一下）
        """
        self._step_axis(PIN_PITCH, dir_sign, STEP_MS_PITCH, DELTA_US)

    # 激光
    def fire(self, duration_ms: int):
        self.pi.write(LASER_PIN, 1)
        time.sleep(duration_ms / 1000.0)
        self.pi.write(LASER_PIN, 0)

    # 停止&清理
    def stop_all(self):
        self.pi.set_servo_pulsewidth(PIN_YAW,   0)
        self.pi.set_servo_pulsewidth(PIN_PITCH, 0)
        self.pi.write(LASER_PIN, 0)

    def close(self):
        self.stop_all()
        self.pi.stop()

if __name__ == "__main__":
    api = ServoAPI()
    try:
        print("测试：小步长左右上下各走几步...")
        for _ in range(5):
            api.step_yaw(+1); time.sleep(0.2)
        for _ in range(5):
            api.step_yaw(-1); time.sleep(0.2)
        for _ in range(5):
            api.step_pitch(+1); time.sleep(0.2)
        for _ in range(5):
            api.step_pitch(-1); time.sleep(0.2)
        print("测试激光 200ms")
        api.fire(200)
    finally:
        api.close()
