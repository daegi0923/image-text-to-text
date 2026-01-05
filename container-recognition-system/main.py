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
    logger.info("=== 배속 제어 기반 멀티 카메라 시스템 시작 ===")

    # 1. 초기화
    camera_units = [] 
    camera_configs = system_conf.get('cameras', [])
    
    if not camera_configs and 'video_sources' in system_conf:
        default_weights = model_conf.get('yolo_path', 'outputs/yolo_container_ocr/weights/best.pt')
        for idx, src in enumerate(system_conf['video_sources']):
            camera_configs.append({'name': f"cam_{idx}", 'source': src, 'weights': default_weights})

    try:
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
        max_fps = 30.0 # 기준 FPS (가장 높은 놈 기준 or 고정)
        
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
                    'fps': cam.fps
                })
                if cam.fps > max_fps: max_fps = cam.fps
                logger.info(f"✅ 유닛: {name} ({cam.fps:.1f} FPS) | Src: {src}")
            except Exception as e:
                logger.error(f"❌ 유닛 실패 ({name}): {e}")

        if not camera_units: return

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

    logger.info(f">>> 모니터링 시작 (기준 FPS: {max_fps:.1f})")

    global_frame_idx = 0
    
    while True:
        loop_start = time.time()
        global_frame_idx += 1
        
        # --- 프레임 읽기 (단순화: 매 루프마다 1장씩) ---
        # 서로 FPS가 달라도 그냥 1장씩 읽으면 영상 속도는 달라지지만
        # 파일 재생에서는 이게 자연스러움 (동시에 끝나는 게 아니라 각자 갈 길 감)
        active_frames = []
        all_closed = True
        
        for unit in camera_units:
            frame = unit['cam'].get_frame()
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
            # 수집 종료 -> 판결
            if current_state == STATE_COLLECTING and state_timer == 0:
                logger.info(f"🛑 수집 종료 (증거 {len(evidence_bucket)}개)")
                if evidence_bucket:
                    image_paths = [e['path'] for e in evidence_bucket]
                    res = ocr_worker.process_batch(image_paths)
                    verdict = ocr_worker.consolidate_results(res)
                    
                    if verdict['found']:
                        num = verdict['container_number']
                        meta = verdict.get('voting_meta', {})
                        logger.info(f"★ 확정: {num} ({meta.get('winner_count')}/{meta.get('total_votes')})")
                        history.append({'time': datetime.now(), 'number': num})
                    else:
                        logger.info("❌ 인식 실패")
                
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
                            logger.info(f"📸 {unit['name']} 감지! 수집 시작")
                        
                        if current_state == STATE_COLLECTING:
                            conf = float(best_box.conf[0])
                            # ROI 저장
                            pw, ph = int((x2-x1)*0.1), int((y2-y1)*0.1)
                            crop = frame[max(0, y1-ph):min(fh, y2+ph), max(0, x1-pw):min(fw, x2+pw)].copy()
                            pre = preprocess_for_ocr(crop)
                            final_img = apply_perspective_correction(pre, intensity=perspective_intensity)
                            
                            path = os.path.join(temp_dir, f"{unit['name']}_{global_frame_idx}.jpg")
                            cv2.imwrite(path, final_img)
                            evidence_bucket.append({'path': path, 'score': conf, 'unit': unit['name']})
                            
                            cv2.putText(disp, "COLLECTING", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            display_frames.append(resize_frame(disp, scale=0.4))

        # [타이머 연장]
        if current_state == STATE_COLLECTING and any_container_detected:
            state_timer = collection_window 

        # 화면 출력
        if display_frames:
            combined = np.hstack(display_frames)
            
            status_map = {0: "IDLE", 1: "COLLECTING", 2: "COOLDOWN"}
            color_map = {0: (0,255,0), 1: (0,165,255), 2: (0,0,255)}
            
            cv2.putText(combined, f"{status_map[current_state]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_map[current_state], 2)
            cv2.imshow('Multi-Camera System', combined)

        # 배속 제어 (WaitKey로 FPS 맞추기)
        # 루프 처리 시간을 빼고 남은 시간만큼 대기
        proc_time = time.time() - loop_start
        wait_ms = max(1, int((1.0/max_fps - proc_time) * 1000))
        
        # 키 입력 (속도 조절 가능하게)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'): break

    for unit in camera_units: unit['cam'].release()
    cv2.destroyAllWindows()
    
    if history:
        pd.DataFrame(history).to_csv(system_conf.get('log_file', 'outputs/gate_log.csv'), index=False)

if __name__ == "__main__":
    main()