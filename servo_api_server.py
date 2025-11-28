#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_api_server.py

Raspberry Pi side:
- Listen on TCP port 50000
- Receive one line of JSON command from the PC
- Call servo_api.ServoAPI to do step-wise motion control + laser firing
- Return a simple JSON response

Example command formats used together with the PC-side script:
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
    Call the corresponding servo/laser action according to the JSON command.
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
    print(f"[INFO] ServoAPI initialized: servos at neutral + laser off")
    print(f"[INFO] Listening on {HOST}:{PORT}, waiting for connections from PC...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)

        try:
            while True:
                conn, addr = s.accept()
                with conn:
                    # Very simple request-response protocol:
                    # PC sends one line of JSON each time, then closes.
                    # If you need persistent connections later, you can
                    # change this to read multiple lines in a loop.
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
            print("\n[INFO] Caught Ctrl+C, exiting...")
        finally:
            print("[INFO] Stopping servos & shutting down pigpio")
            api.close()

if __name__ == "__main__":
    main()
