import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict
import queue
import threading
from collections import deque

from utils.config import load_config
from utils.logger import setup_logger
from utils.image_utils import apply_perspective_correction, preprocess_for_ocr
from drivers.camera import Camera
from core.detector import ContainerDetector

# --- [시스템 상태 관리자] ---
class TriggerManager:
    def __init__(self, duration=5.0):
        self.active = False
        self.last_trigger_time = 0
        self.duration = duration
        self.trigger_source = None

    def activate(self, source_name="unknown"):
        self.active = True
        self.last_trigger_time = time.time()
        self.trigger_source = source_name

    def update(self):
        if self.active and (time.time() - self.last_trigger_time > self.duration):
            self.active = False
            self.trigger_source = None
            return False 
        return self.active

    def is_active(self):
        return self.active

# --- [세션 관리자] 로그만 남김 ---
class GateSessionManager:
    def __init__(self, logger, log_file, timeout=10.0):
        self.logger = logger
        self.log_file = log_file
        self.timeout = timeout 
        self.is_session_active = False
        self.last_update_time = 0

    def notify_trigger(self):
        if not self.is_session_active:
            self.is_session_active = True
            self.logger.info("🎬 [세션 시작] 트럭 진입 감지")
        self.last_update_time = time.time()

    def update(self):
        if self.is_session_active and (time.time() - self.last_update_time > self.timeout):
            self.finalize_session()

    def finalize_session(self):
        if not self.is_session_active: return
        self.logger.info("🏁 [세션 종료] 수집 완료")
        self.is_session_active = False

# --- [유틸] 점수 계산 ---
def calculate_score(image, conf, box_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = (sharpness * 0.4) + (conf * 1000 * 0.4) + (box_area * 0.2)
    return score

def resize_display(frame, scale=0.4):
    if frame is None: return None
    return cv2.resize(frame, None, fx=scale, fy=scale)

# --- [메인] ---
def main():
    config = load_config()
    sys_conf = config.get('system', {})
    param_conf = config.get('parameters', {})
    log_file = sys_conf.get('log_file', 'outputs/gate_log.csv')
    
    system_log_path = os.path.join(os.path.dirname(log_file), 'system.log')
    logger = setup_logger(log_file=system_log_path)
    
    logger.info("=== [Crop Only Mode] 컨테이너 인식 시스템 시작 ===")

    trigger_manager = TriggerManager(duration=param_conf.get('trigger_duration', 5.0))
    session_manager = GateSessionManager(logger, log_file)
    
    cameras = []
    try:
        for cam_conf in sys_conf.get('cameras', []):
            name, role, src = cam_conf.get('name'), cam_conf.get('role', 'slave'), cam_conf.get('source')
            if not src: continue
            try:
                cam = Camera(src)
                detector = ContainerDetector(
                    model_path=cam_conf.get('weights'),
                    conf_threshold=config['model'].get('conf_threshold', 0.5)
                )
                cameras.append({
                    'name': name, 'role': role, 'cam': cam, 'detector': detector,
                    'zone': cam_conf.get('detection_zone'),
                    'targets': cam_conf.get('target_classes'), 
                    'buffer': {}, 
                    'fps': cam.fps, 'acc': 0.0, 'frame_idx': 0,
                    'last_disp_frame': None
                })
                logger.info(f"🎥 [{role.upper()}] {name} 준비 완료")
            except Exception as e:
                logger.error(f"❌ 카메라 실패 ({name}): {e}")
        if not cameras: return
    except Exception as e:
        logger.error(f"초기화 실패: {e}"); return

    temp_dir = sys_conf.get('temp_frame_dir', 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    min_fps = min(c['fps'] for c in cameras)

    while True:
        current_time = time.time()
        session_manager.update()
        trigger_active = trigger_manager.update()

        active_frames, all_closed = [], True
        for c in cameras:
            c['acc'] += (c['fps'] / min_fps)
            n_read = int(c['acc'])
            c['acc'] -= n_read
            frame = None
            for _ in range(n_read):
                f = c['cam'].get_frame()
                if f is not None: frame = f
            if frame is not None: active_frames.append((frame, c)); all_closed = False
            else: active_frames.append((None, c))
        if all_closed: break

        display_frames = []
        for frame, unit in active_frames:
            if frame is None:
                display_frames.append(unit['last_disp_frame'] if unit['last_disp_frame'] is not None else resize_display(np.zeros((360,640,3), np.uint8)))
                continue
            
            disp, role, (fh, fw) = frame.copy(), unit['role'], frame.shape[:2]
            z = unit['zone']
            zx1, zx2, zy1, zy2 = int(fw*z['x_min']), int(fw*z['x_max']), int(fh*z['y_min']), int(fh*z['y_max'])
            
            zone_color = (0, 0, 255) if (role == 'master' and trigger_active) else (255, 0, 0) if role == 'master' else (0, 255, 0) if (role == 'slave' and trigger_active) else (100, 100, 100)
            cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), zone_color, 2)

            unit['frame_idx'] += 1
            if unit['frame_idx'] % 3 == 0 and ((role == 'master') or (role == 'slave' and trigger_active)):
                scale_factor = fw / 640
                results = unit['detector'].track(cv2.resize(frame, (640, int(fh/scale_factor))))
                
                if results:
                    r = results[0]
                    # OBB 처리
                    if hasattr(r, 'obb') and r.obb is not None:
                        track_ids = r.obb.id.int().cpu().tolist() if r.obb.id is not None else [-1]*len(r.obb)
                        for i, obb in enumerate(r.obb):
                            cls_id, conf, tid = int(obb.cls[0]), float(obb.conf[0]), track_ids[i]
                            if unit['targets'] and cls_id not in unit['targets']: continue
                            pts = (obb.xyxyxyxy[0].cpu().numpy() * scale_factor).astype(np.int32)
                            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                            if not (zx1 < cx < zx2 and zy1 < cy < zy2): continue

                            if role == 'master':
                                if not trigger_active: session_manager.notify_trigger()
                                trigger_manager.activate(unit['name'])
                                cv2.polylines(disp, [pts], True, (0, 0, 255), 3)
                            elif role == 'slave' and cls_id == 2:
                                cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
                                if tid not in unit['buffer']: unit['buffer'][tid] = {'images': [], 'last_seen': 0}
                                buf = unit['buffer'][tid]
                                buf['last_seen'] = current_time
                                try:
                                    pts_f = pts.astype(np.float32)
                                    sorted_x = pts_f[np.argsort(pts_f[:, 0])]
                                    l, r_pts = sorted_x[:2], sorted_x[2:]
                                    tl, bl, tr, br = l[np.argmin(l[:,1])], l[np.argmax(l[:,1])], r_pts[np.argmin(r_pts[:,1])], r_pts[np.argmax(r_pts[:,1])]
                                    mw = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
                                    mh = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
                                    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), np.array([[0,0],[mw-1,0],[mw-1,mh-1],[0,mh-1]], dtype="float32"))
                                    final_img = cv2.warpPerspective(frame, M, (mw, mh))
                                    if len(buf['images']) < 5:
                                        cv2.imwrite(os.path.join(temp_dir, f"CROP_{unit['name']}_ID{tid}_{int(current_time*1000)}.jpg"), final_img)
                                        buf['images'].append(True)
                                except Exception as e: logger.error(f"Crop Error: {e}")

                    # 일반 Box 처리 (OBB 미지원 시)
                    elif hasattr(r, 'boxes') and r.boxes is not None:
                        boxes = r.boxes
                        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1]*len(boxes)
                        for box, tid in zip(boxes, track_ids):
                            cls_id, conf = int(box.cls[0]), float(box.conf[0])
                            if unit['targets'] and cls_id not in unit['targets']: continue
                            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * scale_factor).astype(np.int32)
                            cx, cy = (x1+x2)//2, (y1+y2)//2
                            if not (zx1 < cx < zx2 and zy1 < cy < zy2): continue
                            if role == 'master':
                                if not trigger_active: session_manager.notify_trigger()
                                trigger_manager.activate(unit['name'])
                            elif role == 'slave' and cls_id == 2:
                                cv2.imwrite(os.path.join(temp_dir, f"CROP_{unit['name']}_ID{tid}_{int(current_time*1000)}.jpg"), frame[y1:y2, x1:x2])

            if role == 'slave':
                to_remove = [tid for tid, data in unit['buffer'].items() if current_time - data['last_seen'] > 2.0]
                for tid in to_remove: del unit['buffer'][tid]

            unit['last_disp_frame'] = resize_display(disp)
            display_frames.append(unit['last_disp_frame'])

        if display_frames:
            combined = np.vstack([np.hstack(display_frames[:2]), np.hstack(display_frames[2:])]) if len(display_frames) == 4 else np.hstack(display_frames)
            cv2.imshow('Container Recognition System (Crop Only)', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    for c in cameras: c['cam'].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()