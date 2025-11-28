#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_api.py 

Each step_yaw / step_pitch:
- deviates from the neutral position for a short moment, then immediately returns
- prevents continuous rotation / wire twisting
- DELTA_US and STEP_MS_* control how much movement each step produces
"""

import time
import pigpio

# ====== Hardware Pins ======
PIN_YAW   = 17   # Horizontal servo
PIN_PITCH = 27   # Vertical servo
LASER_PIN = 22   # Laser MOSFET

# ====== Servo Parameters ======
NEUTRAL_US = 1500    # Neutral pulse width

# Step offset from the neutral position
DELTA_US   = 60      

# Duration of each step
STEP_MS_YAW   = 20  
STEP_MS_PITCH = 20

class ServoAPI:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon is not running. Please start with: sudo pigpiod")

        # Initialize both servos to the neutral position; turn off laser
        self.pi.set_servo_pulsewidth(PIN_YAW,   NEUTRAL_US)
        self.pi.set_servo_pulsewidth(PIN_PITCH, NEUTRAL_US)

        self.pi.set_mode(LASER_PIN, pigpio.OUTPUT)
        self.pi.write(LASER_PIN, 0)

    # Internal: move one small step
    def _step_axis(self, pin, dir_sign, step_ms, delta_us):
        """
        dir_sign: +1 / -1
        step_ms : duration of the small step (ms)
        delta_us: pulse width offset from the neutral position (µs)
        """
        if dir_sign == 0:
            return
        # Move away from neutral
        pulse = NEUTRAL_US + int(dir_sign * delta_us)
        self.pi.set_servo_pulsewidth(pin, pulse)
        # Hold for a short moment
        time.sleep(step_ms / 1000.0)
        # Return to neutral, preventing continuous rotation
        self.pi.set_servo_pulsewidth(pin, NEUTRAL_US)

    # Public API: one small horizontal / vertical step
    def step_yaw(self, dir_sign: int):
        """
        One small horizontal step
        dir_sign: +1 right, -1 left (actual direction depends on servo installation)
        """
        self._step_axis(PIN_YAW, dir_sign, STEP_MS_YAW, DELTA_US)

    def step_pitch(self, dir_sign: int):
        """
        One small vertical step
        dir_sign: +1 down, -1 up (reverse the meanings if your servo is inverted)
        """
        self._step_axis(PIN_PITCH, dir_sign, STEP_MS_PITCH, DELTA_US)

    # Laser
    def fire(self, duration_ms: int):
        self.pi.write(LASER_PIN, 1)
        time.sleep(duration_ms / 1000.0)
        self.pi.write(LASER_PIN, 0)

    # Stop & cleanup
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
        print("Test: small steps left/right/up/down...")
        for _ in range(5):
            api.step_yaw(+1); time.sleep(0.2)
        for _ in range(5):
            api.step_yaw(-1); time.sleep(0.2)
        for _ in range(5):
            api.step_pitch(+1); time.sleep(0.2)
        for _ in range(5):
            api.step_pitch(-1); time.sleep(0.2)
        print("Test laser 200ms")
        api.fire(200)
    finally:
        api.close()
