import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict

from utils.config import load_config
from utils.logger import setup_logger
from utils.visualizer import Visualizer
from utils.image_utils import apply_perspective_correction, preprocess_for_ocr
from drivers.camera import Camera
from core.detector import ContainerDetector
from services.ocr_worker import ContainerOCR

def resize_frame(frame, scale=0.5):
    """화면 표시용 리사이즈"""
    return cv2.resize(frame, None, fx=scale, fy=scale)

def main():
    config = load_config()
    system_conf = config.get('system', {})
    model_conf = config.get('model', {})
    params_conf = config.get('parameters', {})
    
    logger = setup_logger(log_file=system_conf.get('log_file', 'outputs/gate_log.csv'))
    logger.info("=== 커스텀 타겟팅 기반 멀티 카메라 시스템 시작 ===")

    # 1. 초기화
    camera_units = [] 
    camera_configs = system_conf.get('cameras', [])
    
    # 전역 타겟 클래스 (기본값)
    global_target_classes = model_conf.get('target_classes', None)
    
    try:
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
        for conf in camera_configs:
            name = conf.get('name', 'unknown')
            src = conf.get('source')
            weights = conf.get('weights')
            zone = conf.get('detection_zone', {'x_min': 0.4, 'x_max': 0.6, 'y_min': 0.25, 'y_max': 0.75})
            
            # 카메라별 타겟 클래스 설정 (없으면 전역 설정 사용)
            target_classes = conf.get('target_classes', global_target_classes)
            
            if not src: continue
            try:
                cam = Camera(src)
                detector = ContainerDetector(
                    model_path=weights,
                    default_model=model_conf.get('yolo_default', 'yolo11n.pt'),
                    conf_threshold=model_conf.get('conf_threshold', 0.5)
                )
                camera_units.append({
                    'cam': cam, 'detector': detector, 'name': name,
                    'fps': cam.fps, 'acc': 0.0,
                    'zone': zone,
                    'target_classes': target_classes # 유닛별 타겟 저장
                })
                logger.info(f"✅ 유닛: {name} | Targets: {target_classes if target_classes else 'ALL'}")
            except Exception as e:
                logger.error(f"❌ 유닛 실패 ({name}): {e}")

        if not camera_units: return
        min_fps = min(u['fps'] for u in camera_units)

    except Exception as e:
        logger.error(f"초기화 에러: {e}")
        return

    # 2. 상태 변수
    STATE_IDLE = 0
    STATE_COLLECTING = 1
    STATE_COOLDOWN = 2
    current_state = STATE_IDLE
    
    collection_window = params_conf.get('collection_window', 60)
    cooldown_frames = params_conf.get('cooldown_frames', 150)
    perspective_intensity = params_conf.get('perspective_intensity', 0.0)
    
    state_timer = 0
    evidence_bucket = []
    history = []
    temp_dir = system_conf.get('temp_frame_dir', 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)

    global_step = 0

    while True:
        global_step += 1
        active_frames = []
        all_closed = True
        
        # --- 프레임 읽기 (동기화) ---
        for unit in camera_units:
            unit['acc'] += (unit['fps'] / min_fps)
            num_to_read = int(unit['acc'])
            unit['acc'] -= num_to_read
            
            frame = None
            for _ in range(num_to_read):
                f = unit['cam'].get_frame()
                if f is not None: frame = f
            
            if frame is not None:
                active_frames.append((frame, unit))
                all_closed = False
            else:
                active_frames.append((None, unit))

        if all_closed: break

        # --- 로직 처리 (상태 머신) ---
        if state_timer > 0:
            state_timer -= 1
            if current_state == STATE_COLLECTING and state_timer == 0:
                logger.info(f"🛑 수집 종료 (증거 {len(evidence_bucket)}개)")
                if evidence_bucket:
                    image_paths = [e['path'] for e in evidence_bucket]
                    res = ocr_worker.process_batch(image_paths)
                    verdict = ocr_worker.consolidate_results(res)
                    if verdict['found']:
                        num = verdict['container_number']
                        logger.info(f"★ 확정: {num}")
                        history.append({'time': datetime.now(), 'number': num})
                
                current_state = STATE_COOLDOWN
                state_timer = cooldown_frames
                evidence_bucket = []
            elif current_state == STATE_COOLDOWN and state_timer == 0:
                current_state = STATE_IDLE

        # --- 탐지 및 표시 ---
        display_frames = []
        any_container_detected = False

        for frame, unit in active_frames:
            if frame is None: continue
            
            disp = frame.copy()
            fh, fw = frame.shape[:2]
            zone = unit['zone']
            
            # [시각화] 인식 존
            zx1, zx2 = int(fw * zone['x_min']), int(fw * zone['x_max'])
            zy1, zy2 = int(fh * zone['y_min']), int(fh * zone['y_max'])
            cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), (255, 200, 0), 2)
            cv2.putText(disp, "Zone", (zx1, zy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            
            if current_state != STATE_COOLDOWN:
                # ★ 유닛별 타겟 클래스 적용
                best_box = unit['detector'].detect(frame, target_classes=unit['target_classes'])
                
                if best_box is not None:
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    
                    # 클래스 이름 추출
                    cls_id = int(best_box.cls[0])
                    # names가 딕셔너리인지 리스트인지 확인하고 안전하게 가져오기
                    names = unit['detector'].model.names
                    cls_name = names[cls_id] if cls_id in names else str(cls_id)
                    
                    # 존 체크
                    is_centered = (zx1 < cx < zx2) and (zy1 < cy < zy2)
                    
                    # 박스 그리기
                    Visualizer.draw_detection(disp, best_box, is_centered)
                    
                    # ★ 화면에 클래스 이름 표시
                    cv2.putText(disp, f"{cls_name}", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if is_centered:
                        any_container_detected = True
                        if current_state == STATE_IDLE:
                            current_state = STATE_COLLECTING
                            state_timer = collection_window
                            # ★ 로그에 클래스 이름 포함
                            logger.info(f"📸 {unit['name']} 감지! [{cls_name}] 수집 시작")
                        
                        if current_state == STATE_COLLECTING:
                            path = os.path.join(temp_dir, f"{unit['name']}_{global_step}.jpg")
                            pad = 20
                            crop = frame[max(0, y1-pad):min(fh, y2+pad), max(0, x1-pad):min(fw, x2+pad)].copy()
                            cv2.imwrite(path, crop)
                            evidence_bucket.append({'path': path, 'unit': unit['name']})
                            cv2.putText(disp, "COLLECTING", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            display_frames.append(resize_frame(disp, scale=0.4))

        if current_state == STATE_COLLECTING and any_container_detected:
            state_timer = collection_window 

        if display_frames:
            combined = np.hstack(display_frames)
            status_map = {0: "IDLE", 1: "COLLECTING", 2: "COOLDOWN"}
            cv2.putText(combined, f"SYSTEM: {status_map[current_state]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow('Multi-Target System', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    for unit in camera_units: unit['cam'].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
