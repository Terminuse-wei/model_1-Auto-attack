#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_api_server.py

树莓派端：
- 监听 TCP 50000 端口
- 接收来自 PC 的一行 JSON 命令
- 调用 servo_api.ServoAPI 做“一点一点动”的步进控制 + 激光开火
- 返回简单的 JSON 响应

配合 PC 端脚本使用的命令格式示例：
    {"cmd": "step_yaw",   "dir": 1}
    {"cmd": "step_pitch", "dir": -1}
    {"cmd": "fire",       "duration_ms": 20}
    {"cmd": "stop"}
"""

import socket
import json
import traceback

from servo_api import ServoAPI

HOST = "0.0.0.0"
PORT = 50000

def handle_one_cmd(api: ServoAPI, cmd: dict):
    """
    根据 JSON 命令调用对应的舵机/激光动作
    """
    c = cmd.get("cmd", "")
    if c == "step_yaw":
        d = int(cmd.get("dir", 0))
        api.step_yaw(d)
        return {"status": "ok", "cmd": c, "dir": d}
    elif c == "step_pitch":
        d = int(cmd.get("dir", 0))
        api.step_pitch(d)
        return {"status": "ok", "cmd": c, "dir": d}
    elif c == "fire":
        ms = int(cmd.get("duration_ms", 20))
        api.fire(ms)
        return {"status": "ok", "cmd": c, "duration_ms": ms}
    elif c == "stop":
        api.stop_all()
        return {"status": "ok", "cmd": c}
    else:
        return {"status": "error", "msg": f"unknown cmd: {c}"}

def main():
    api = ServoAPI()
    print(f"[INFO] ServoAPI 初始化完成，中点停止 + 激光关闭")
    print(f"[INFO] 监听 {HOST}:{PORT}，等待来自 PC 的连接...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)

        try:
            while True:
                conn, addr = s.accept()
                with conn:
                    # 简单的一问一答协议：PC 每次发一行 JSON，然后断开
                    # 如果你以后需要长连接，可以改成循环读多行
                    # 这里先保持简单
                    data = conn.recv(4096)
                    if not data:
                        continue
                    text = data.decode("utf-8").strip()
                    try:
                        cmd = json.loads(text)
                    except Exception:
                        resp = {"status": "error", "msg": "invalid json", "raw": text}
                    else:
                        try:
                            resp = handle_one_cmd(api, cmd)
                        except Exception as e:
                            traceback.print_exc()
                            resp = {"status": "error", "msg": str(e)}

                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except KeyboardInterrupt:
            print("\n[INFO] 收到 Ctrl+C，准备退出...")
        finally:
            print("[INFO] 停止舵机 & 关闭 pigpio")
            api.close()

if __name__ == "__main__":
    main()
