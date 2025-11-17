
# Auto-attack (Grad-CAM + Laser + Servo Closed-Loop)

This repository contains only the **minimal viable version** of the project, including:

- PC-side vision & Grad-CAM & closed-loop control code  
- Raspberry Pi-side servo control server code  
- Pre-trained classification model (normal / failure)

---

## 1. File Description

- `xai_closedloop_pc.py`  
  PC main program: Capture image → Select ROI → Grad-CAM → Detect red/green laser → Control Raspberry Pi servo step-by-step via TCP.

- `servo_api_server.py`
Raspberry Pi TCP server: Receives JSON commands (`step_yaw` / `step_pitch` / `fire` / `stop`) from PC, invokes `servo_api.py` to control servo and laser.

- `servo_api.py`
Raspberry Pi servo control API based on `pigpio`, implementing “**step-by-step**” incremental movement to prevent cable entanglement.

- `model_def.py`
Model architecture definition (SmallCNN / ResNet18Classifier, etc.) for loading `panel_cls_full.pt`.

- `panel_cls_full.pt`  
  Fully trained model: Contains weights, class information, normalization parameters, Grad-CAM target layer, etc.

---

## 2. Environment Dependencies (Simplified Version)

### Raspberry Pi

```bash
sudo apt-get update
sudo apt-get install python3-pip pigpio python3-pigpio
sudo systemctl start pigpiod
sudo systemctl enable pigpiod

pip3 install numpy
PC Side (Python 3.9+ recommended)
pip install torch torchvision opencv-python numpy

3. Execution Sequence

3.1 Raspberry Pi Side

On the Raspberry Pi (in the directory containing servo_api.py and servo_api_server.py), run:
python3 servo_api_server.py
Functionality:
Initialize servos and laser (laser disabled by default)
Listen for commands from the PC at 0.0.0.0:50000
{“cmd”: “step_yaw”, “dir”: 1}
{“cmd”: “step_pitch”, “dir”: -1}
{“cmd”: “fire”, “duration_ms”: 20}
{‘cmd’: “stop”}

Keep this script running continuously; do not shut it down.
3.2 PC Side

On the PC (in the same directory as xai_closedloop_pc.py, panel_cls_full.pt, and model_def.py), run:
python xai_closedloop_pc.py
Before running, ensure:
PI_IP in the code is set to your Raspberry Pi's IP address
The camera is connected to the PC
4. Operating Steps (Important)

4.1 Selecting the ROI
After program startup, the camera window will open.
Use the mouse to drag and select the ROI area (encompassing the panel and laser-marked region).
Key functions:
s: Lock the ROI and enter closed-loop control;
q: Exit the program.
4.2 Switching Modes During Closed-Loop Phase

After ROI locking, the system enters closed-loop state with a prompt at the bottom of the window.
Press 1: Auto Mode
Use Grad-CAM to locate hotspots within the ROI once
Once the target point is determined, it locks and does not jump randomly
The servo motor will step-by-step track the laser point
Press 2: Manual Mode (Manual)
Use the mouse to click any point within the ROI as the target
The laser will step-by-step track to your clicked position
You can click another position at any time to reset the target
q: Exit closed-loop and close the window.
5. Color Logic (Red Dot / Green Dot)

For each frame, the PC classifies the ROI (using panel_cls_full.pt):
Classified as NORMAL:
Prompt: “Use red dot (RED)”
The laser detection module searches only for the red laser dot within the ROI.
Classified as INVALID / FAILURE:
Prompt: “Use green dot (GREEN)”
The laser detection module searches only for the green laser dot within the ROI.

After successful laser dot detection, the PC calculates the direction based on the pixel error between the target point and the laser dot and issues commands via:
{“cmd”: “step_yaw”, “dir”: ±1}
{“cmd”: ‘step_pitch’, “dir”: ±1}
to drive the servo in small increments, achieving safe, slow automatic alignment.
6. Parameter Tuning Notes (Optional)

These parameters can be modified at the top of xai_closedloop_pc.py:
PI_IP: Raspberry Pi IP address
NORMAL_THR: Normal/Failure determination threshold
PIX_ERR_THR: Pixel error range considered “aligned”
INVERT_YAW / INVERT_PITCH: If direction reversal is detected, change from 1 to -1 or vice versa.
If you need a super-short English summary (1–2 sentences for your instructor), I can also write that for you.
