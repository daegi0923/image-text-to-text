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
    logger.info("=== 비율 기반 프레임 동기화 시스템 시작 ===")

    # 1. 초기화
    camera_units = [] 
    camera_configs = system_conf.get('cameras', [])
    
    try:
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
        for conf in camera_configs:
            name = conf.get('name', 'unknown')
            src = conf.get('source')
            weights = conf.get('weights')
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
                    'fps': cam.fps,
                    'acc': 0.0 # 프레임 누적기
                })
                logger.info(f"✅ 유닛: {name} ({cam.fps:.1f} FPS) | Model: {weights}")
            except Exception as e:
                logger.error(f"❌ 유닛 실패 ({name}): {e}")

        if not camera_units: return
        
        # 기준이 될 최소 FPS 찾기
        min_fps = min(u['fps'] for u in camera_units)
        logger.info(f"기준 FPS (최소): {min_fps:.1f}")

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
        
        # --- [핵심] 비율 기반 프레임 읽기 ---
        # 12fps vs 24fps 상황이라면:
        # 12fps는 루프당 1장, 24fps는 루프당 2장 읽어서 싱크 맞춤
        for unit in camera_units:
            # 루프당 읽어야 할 프레임 수 계산 (예: 24/12 = 2.0)
            unit['acc'] += (unit['fps'] / min_fps)
            num_to_read = int(unit['acc'])
            unit['acc'] -= num_to_read # 소수점 잔여량 유지 (비정수 FPS 대비)
            
            frame = None
            for _ in range(num_to_read):
                f = unit['cam'].get_frame()
                if f is not None:
                    frame = f # 마지막으로 읽은 프레임 사용
            
            if frame is not None:
                active_frames.append((frame, unit))
                all_closed = False
            else:
                active_frames.append((None, unit))

        if all_closed:
            logger.info("모든 영상 종료")
            break

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
                logger.info("🟢 대기 모드 (IDLE)")

        # --- 탐지 및 표시 ---
        display_frames = []
        any_container_detected = False

        for frame, unit in active_frames:
            if frame is None: continue
            
            disp = frame.copy()
            fh, fw = frame.shape[:2]
            
            if current_state != STATE_COOLDOWN:
                best_box = unit['detector'].detect(frame)
                if best_box is not None:
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    is_centered = (fw*0.4 < cx < fw*0.6) and (fh*0.25 < cy < fh*0.75)
                    Visualizer.draw_detection(disp, best_box, is_centered)
                    
                    if is_centered:
                        any_container_detected = True
                        if current_state == STATE_IDLE:
                            current_state = STATE_COLLECTING
                            state_timer = collection_window
                        
                        if current_state == STATE_COLLECTING:
                            path = os.path.join(temp_dir, f"{unit['name']}_{global_step}.jpg")
                            # 전처리 및 저장
                            crop = frame[max(0, y1-10):min(fh, y2+10), max(0, x1-10):min(fw, x2+10)].copy()
                            cv2.imwrite(path, crop)
                            evidence_bucket.append({'path': path, 'unit': unit['name']})
                            cv2.putText(disp, "COLLECTING", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            display_frames.append(resize_frame(disp, scale=0.4))

        if current_state == STATE_COLLECTING and any_container_detected:
            state_timer = collection_window 

        if display_frames:
            combined = np.hstack(display_frames)
            cv2.imshow('Sync System (Ratio-based)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    for unit in camera_units: unit['cam'].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
