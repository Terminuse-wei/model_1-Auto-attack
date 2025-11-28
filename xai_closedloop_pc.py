#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xai_closedloop_pc.py

PC-side visual closed loop:
- Use panel_cls_full.pt + Grad-CAM (same logic as viewer_gradmap_judge.py)
- Inside ROI:
    * Auto mode: Grad-CAM selects one hot target pixel (u_target, v_target) once and then keeps it fixed
    * Manual mode: you click a point inside the ROI as the target
    * Decide laser color automatically according to NORMAL / INVALID:
        - NORMAL  -> RED dot, detect red laser
        - INVALID -> GREEN dot, detect green laser
    * Laser-spot detection → compute error → send step_yaw / step_pitch to Raspberry Pi,
      so that the servos move towards the target step by step

Hotkeys:
    - ROI stage:
        * Drag mouse to select ROI, press 's' to lock and start closed loop
        * Press 'q' to quit
    - Closed-loop stage:
        * '1' = auto mode (Grad-CAM locks the target once and does not drift afterwards)
        * '2' = manual mode (click inside ROI to set target point)
        * 'r' = release lock (servos can adjust again, but the auto-mode target stays)
        * 'q' = quit
"""

import os
import json
import socket
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from model_def import SmallCNN, ConvBNReLU
import torch.serialization
from torch.nn.modules.container import Sequential
from torch.nn import Conv2d, BatchNorm2d, Linear, Dropout2d, ReLU, MaxPool2d, AdaptiveAvgPool2d

# ====== Safe globals whitelist for deserialization ======
torch.serialization.add_safe_globals([
    SmallCNN,
    ConvBNReLU,
    Sequential,
    Conv2d,
    BatchNorm2d,
    Linear,
    Dropout2d,
    ReLU,
    MaxPool2d,
    AdaptiveAvgPool2d,
])

# Fallback compatibility in case panel_cls_full.pt uses ResNet18Classifier
try:
    from model_def import ResNet18Classifier
    import sys
    torch.serialization.add_safe_globals([ResNet18Classifier])
    sys.modules["__main__"].ResNet18Classifier = ResNet18Classifier
except Exception:
    pass

# ====== Raspberry Pi communication config ======
PI_IP   = "10.172.153.228"   # Change to your Raspberry Pi IP
PI_PORT = 50000

def send_pi_cmd(cmd: dict):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.5)
    try:
        s.connect((PI_IP, PI_PORT))
        s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        data = s.recv(4096)
        txt  = data.decode("utf-8").strip()
        try:
            return json.loads(txt)
        except Exception:
            return {"status": "raw", "raw": txt}
    except Exception as e:
        print("[ERR] send_pi_cmd:", e)
        return {"status": "err", "msg": str(e)}
    finally:
        s.close()

# ====== Some small UI helpers ======
def put_text(img, s, org, color=(0,255,0), scale=0.8, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ====== ROI selector ======
class ROISelector:
    def __init__(self, win):
        self.win=win
        self.dragging=False
        self.x0=self.y0=self.x1=self.y1=0
        self.roi=None
        self.locked=False
    def on_mouse(self, event,x,y,flags,param):
        if self.locked:
            return
        if event==cv2.EVENT_LBUTTONDOWN:
            self.dragging=True; self.x0=self.x1=x; self.y0=self.y1=y
        elif event==cv2.EVENT_MOUSEMOVE and self.dragging:
            self.x1,self.y1=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            self.dragging=False
            x0,x1=sorted([self.x0,self.x1]); y0,y1=sorted([self.y0,self.y1])
            if x1-x0>=10 and y1-y0>=10:
                self.roi=(x0,y0,x1,y1)
    def draw(self, vis):
        if self.roi is not None:
            x0,y0,x1,y1=self.roi
            cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,255),2)
            put_text(vis,f"ROI[{'lock' if self.locked else 'edit'}]",(x0,max(22,y0-8)),(0,255,255),0.7,2)
        if self.dragging:
            x0,x1=sorted([self.x0,self.x1]); y0,y1=sorted([self.y0,self.y1])
            cv2.rectangle(vis,(x0,y0),(x1,y1),(0,200,255),1)

# ====== Grad-CAM class ======
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model.eval()
        self.target = self._get_module(model, target_layer_name)
        if self.target is None:
            raise ValueError(f"Layer not found: {target_layer_name}")
        self.act=None
        self.grad=None
        self.fh = self.target.register_forward_hook(self._fh)
        self.bh = self.target.register_full_backward_hook(self._bh)
    def _get_module(self, root, name):
        m = root
        for n in name.split('.'):
            m = m[int(n)] if n.isdigit() else getattr(m, n, None)
            if m is None:
                return None
        return m
    def _fh(self, m, i, o):
        self.act = o.detach()
    def _bh(self, m, gi, go):
        self.grad = go[0].detach()
    @torch.no_grad()
    def _norm(self, x):
        x = x - x.min()
        return x / (x.max() - x.min() + 1e-6)
    def generate(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logits = self.model(x)
        score = logits.reshape(1,-1)[0, class_idx]
        score.backward(retain_graph=True)
        A, dA = self.act, self.grad
        w = dA.mean(dim=(2,3), keepdim=True)
        cam = (w * A).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1),
                            size=x.shape[-2:],
                            mode="bilinear",
                            align_corners=False).squeeze(1)
        return self._norm(cam[0].cpu().numpy())

# ====== Model loading (same as viewer) ======
MODEL_PATH = "panel_cls_full.pt"

def load_full_model(device, path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    import torch.serialization
    from model_def import SmallCNN
    # Fallback for ResNet18Classifier
    try:
        from model_def import ResNet18Classifier
        import sys
        torch.serialization.add_safe_globals([ResNet18Classifier])
        sys.modules['__main__'].ResNet18Classifier = ResNet18Classifier
    except Exception:
        pass
    torch.serialization.add_safe_globals([SmallCNN])
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    class_names = getattr(model, "class_names", ["normal","network_failure"])
    normal_idx  = int(getattr(model, "normal_idx", 0))
    input_size  = int(getattr(model, "input_size", 224))
    normalize   = getattr(model, "normalize",
                          {"mean":[0.485,0.456,0.406],
                           "std":[0.229,0.224,0.225]})
    # For your SmallCNN this is usually feat.3
    target_layer = getattr(model, "target_layer", "feat.3")
    mean, std = normalize["mean"], normalize["std"]
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    cam = GradCAM(model, target_layer)
    meta = {
        "class_names": class_names,
        "normal_idx":  normal_idx,
        "input_size":  input_size,
        "target_layer": target_layer
    }
    return model, tfm, cam, normal_idx, meta

# ====== Laser-spot detection (red / green) ======
def detect_laser_point(bgr_roi, color="red"):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    if color == "red":
        mask1 = cv2.inRange(hsv, (0,150,200),  (10,255,255))
        mask2 = cv2.inRange(hsv, (170,150,200),(180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
    else:  # green
        lower_g = (40, 80, 180)
        upper_g = (90, 255, 255)
        mask = cv2.inRange(hsv, lower_g, upper_g)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return (cx, cy)

# ====== Grad-CAM automatic target point ======
def gradcam_target_pixel(model, tfm, cam, device, roi_bgr,
                         normal_idx=0, normal_thr=0.6):
    Hroi, Wroi = roi_bgr.shape[:2]
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    cx, cy = Wroi//2, Hroi//2
    scales = [1.00,0.85,0.70,0.55,0.45]
    best = {"p1": -1}
    best_x = None
    for s in scales:
        ww = int(Wroi * s)
        hh = int(Hroi * s)
        x0 = max(0, cx - ww//2); x1 = min(Wroi, cx + ww//2)
        y0 = max(0, cy - hh//2); y1 = min(Hroi, cy + hh//2)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        x = tfm(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, 1)[0].cpu().numpy()
        if prob[1] > best["p1"]:
            best = {
                "p1": prob[1],
                "p0": prob[0],
                "pred": int(np.argmax(prob)),
                "scale": s,
                "p_top": float(prob[int(np.argmax(prob))]),
            }
            best_x = x
    if best_x is None:
        x = tfm(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, 1)[0].cpu().numpy()
        best = {
            "p1": prob[1],
            "p0": prob[0],
            "pred": int(np.argmax(prob)),
            "scale": 1.0,
            "p_top": float(prob[int(np.argmax(prob))]),
        }
        best_x = x
    pred_idx = best["pred"]
    is_normal = (pred_idx == normal_idx) and best["p_top"] >= normal_thr
    heat_small = cam.generate(best_x, class_idx=pred_idx)
    heat = cv2.resize(heat_small, (Wroi,Hroi), interpolation=cv2.INTER_LINEAR)
    max_idx = np.argmax(heat)
    v, u = divmod(max_idx, Wroi)
    prob_vec = np.array([best["p0"], best["p1"]], dtype=np.float32)
    return heat, (u,v), prob_vec, pred_idx, is_normal, best["scale"], best["p_top"]

# ====== Main ======
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm, cam, normal_idx, meta = load_full_model(device, MODEL_PATH)
    print("[INFO] Model loaded:", meta)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera 0")
    # Fix resolution to avoid macOS continuous-camera jitter
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    win = "XAI Closed-loop (Grad-CAM + Laser)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    selector = ROISelector(win)

    # Mode: auto / manual
    mode = "auto"
    manual_target_uv = None   # Target (u,v) in manual mode
    auto_target_uv   = None   # In auto mode, first Grad-CAM target (u,v); kept fixed afterwards

    roi_box = None  # (x0,y0,x1,y1)

    NORMAL_THR  = 0.60
    PIX_ERR_THR = 10      # Pixel error threshold for alignment
    LOCK_FRAMES = 5       # Number of consecutive frames within threshold to enter LOCKED state

    INVERT_YAW   = -1     # Adjust depending on servo wiring
    INVERT_PITCH = -1

    # Closed-loop state
    stable_frames = 0
    locked_aim = False

    # Mouse callback: ROI + manual target point
    def mouse_cb(event, x, y, flags, param):
        nonlocal manual_target_uv, roi_box, mode
        selector.on_mouse(event, x, y, flags, param)
        if selector.locked and roi_box is not None and mode == "manual":
            if event == cv2.EVENT_LBUTTONDOWN:
                x0,y0,x1,y1 = roi_box
                if x0 <= x < x1 and y0 <= y < y1:
                    manual_target_uv = (x - x0, y - y0)
                    print(f"[INFO] Manual target = {manual_target_uv}")
    cv2.setMouseCallback(win, mouse_cb)

    # -------- ROI selection --------
    print("========== ROI selection ==========")
    print("Drag ROI with mouse, press 's' to start, 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        selector.draw(vis)
        H,W = vis.shape[:2]
        put_text(vis, "Drag ROI, press 's' to start", (10,30), (0,255,0), 0.7,2)
        put_text(vis, "Press 'q' to quit", (10,H-15), (200,200,200), 0.6,1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return
        if k == ord('s'):
            if selector.roi is None:
                print("[WARN] Please drag to select an ROI before pressing 's'")
                continue
            roi_box = selector.roi
            selector.locked = True
            print("[INFO] ROI locked:", roi_box)
            break

    print("========== Closed loop started ==========")
    print("1 = auto mode (Grad-CAM, lock target once)")
    print("2 = manual mode (click in ROI to set target)")
    print("r = reset lock (re-align)")
    print("q = quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            H,W = frame.shape[:2]
            x0,y0,x1,y1 = roi_box
            x0 = clamp(x0,0,W-2); x1 = clamp(x1,1,W-1)
            y0 = clamp(y0,0,H-2); y1 = clamp(y1,1,H-1)
            roi_bgr = frame[y0:y1, x0:x1].copy()
            if roi_bgr.size == 0:
                print("[WARN] ROI size = 0"); continue

            # 1) Grad-CAM + classification
            try:
                heat, (u_auto,v_auto), prob, pred_idx, is_normal, scale_used, p_top = \
                    gradcam_target_pixel(model, tfm, cam, device, roi_bgr,
                                         normal_idx=normal_idx,
                                         normal_thr=NORMAL_THR)
            except Exception as e:
                print("[ERR] gradcam failed:", e)
                traceback.print_exc()
                continue

            cls_name = meta["class_names"][pred_idx] if "class_names" in meta else str(pred_idx)

            # NORMAL / INVALID → choose red or green laser
            if is_normal:
                laser_color = "red"
                laser_hint = "NORMAL → please use RED laser"
            else:
                laser_color = "green"
                laser_hint = "INVALID → please use GREEN laser"

            # 2) Choose target point: auto / manual
            if mode == "auto":
                # In auto mode, only set auto_target_uv once when it is None
                if auto_target_uv is None:
                    auto_target_uv = (u_auto, v_auto)
                    print(f"[INFO] Auto mode locked target: {auto_target_uv}")
                u_t, v_t = auto_target_uv
            else:
                if manual_target_uv is not None:
                    u_t, v_t = manual_target_uv
                else:
                    # In manual mode before click, use Grad-CAM point as placeholder
                    u_t, v_t = (u_auto, v_auto)

            # 3) Detect laser spot
            laser_uv = detect_laser_point(roi_bgr, color=laser_color)

            # Visualize ROI + CAM
            roi_vis = roi_bgr.copy()
            heat_u8 = (np.clip(heat,0,1)*255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            roi_vis = cv2.addWeighted(roi_vis, 0.5, heat_color, 0.5, 0)
            # Draw target point (blue)
            cv2.circle(roi_vis, (int(u_t), int(v_t)), 6, (255,0,0), 2)

            status = ""
            if laser_uv is not None:
                u_l, v_l = laser_uv
                cv2.circle(roi_vis, (u_l,v_l), 6, (0,0,255), 2)
                err_u = u_t - u_l
                err_v = v_t - v_l
                status = f"target=({u_t:.0f},{v_t:.0f}) laser=({u_l},{v_l}) err=({err_u:.0f},{err_v:.0f})"
            else:
                err_u = err_v = None
                status = "Laser NOT detected"
                stable_frames = 0  # do not accumulate when no laser is detected

            # Put ROI back into full frame (make sure sizes match)
            vis = frame.copy()
            h_roi = y1 - y0
            w_roi = x1 - x0
            if roi_vis.shape[0] != h_roi or roi_vis.shape[1] != w_roi:
                roi_vis = cv2.resize(roi_vis, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
            vis[y0:y1, x0:x1] = roi_vis
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)

            # 4) Servo closed-loop control + alignment lock
            if laser_uv is not None and err_u is not None and err_v is not None:
                if abs(err_u) <= PIX_ERR_THR and abs(err_v) <= PIX_ERR_THR:
                    stable_frames += 1
                else:
                    stable_frames = 0
                if not locked_aim and stable_frames >= LOCK_FRAMES:
                    locked_aim = True
                    print("[INFO] Target aligned, entering LOCKED state (stop stepping)")

                step_yaw = 0
                step_pitch = 0
                if not locked_aim:
                    if abs(err_u) > PIX_ERR_THR:
                        dir_yaw = +1 if err_u > 0 else -1
                        step_yaw = INVERT_YAW * dir_yaw
                    if abs(err_v) > PIX_ERR_THR:
                        dir_pitch = +1 if err_v > 0 else -1
                        step_pitch = INVERT_PITCH * dir_pitch
                    if step_yaw != 0:
                        send_pi_cmd({"cmd": "step_yaw", "dir": int(step_yaw)})
                    if step_pitch != 0:
                        send_pi_cmd({"cmd": "step_pitch", "dir": int(step_pitch)})
            else:
                stable_frames = 0  # cannot accumulate when laser is lost

            # 5) Status text
            lock_flag = "LOCKED" if locked_aim else "TRACKING"
            text1 = f"Mode={mode.upper()} | {lock_flag} | Pred={cls_name} P0={prob[0]:.2f} P1={prob[1]:.2f} thr={NORMAL_THR:.2f}"
            put_text(vis, text1, (10,30), (0,255,0) if is_normal else (0,0,255), 0.7,2)
            put_text(vis, laser_hint, (10,60), (0,255,255), 0.6,2)
            put_text(vis, status, (10,90), (255,255,0), 0.6,2)
            H,W = vis.shape[:2]
            put_text(vis, "1=auto, 2=manual(click in ROI), r=reset lock, q=quit", (10,H-15), (200,200,200), 0.6,1)

            cv2.imshow(win, vis)
            k = cv2.waitKey(50) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('1'):
                mode = "auto"
                manual_target_uv = None
                auto_target_uv = None  # next frame Grad-CAM will lock a new target
                locked_aim = False
                stable_frames = 0
                print("[INFO] Switched to auto mode (next Grad-CAM will lock a new target)")
            elif k == ord('2'):
                mode = "manual"
                manual_target_uv = None
                locked_aim = False
                stable_frames = 0
                print("[INFO] Switched to manual mode, please click inside ROI to set target")
            elif k == ord('r'):
                locked_aim = False
                stable_frames = 0
                print("[INFO] Reset LOCKED state, continue adjusting servos (target unchanged)")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
