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
    # 1. 설정 및 로거 초기화
    config = load_config()
    system_conf = config.get('system', {})
    model_conf = config.get('model', {})
    params_conf = config.get('parameters', {})
    
    logger = setup_logger(log_file=system_conf.get('log_file', 'outputs/gate_log.csv'))
    logger.info("=== 멀티 모델/멀티 카메라 인식 시스템 시작 ===")

    # 2. 모듈 초기화 (카메라 + 전담 탐지기 페어링)
    camera_units = [] # [{'cam': obj, 'detector': obj, 'name': str}, ...]
    
    camera_configs = system_conf.get('cameras', [])
    
    # 하위 호환성 (기존 video_sources 형식이면 변환)
    if not camera_configs and 'video_sources' in system_conf:
        default_weights = model_conf.get('yolo_path', 'outputs/yolo_container_ocr/weights/best.pt')
        for idx, src in enumerate(system_conf['video_sources']):
            camera_configs.append({
                'name': f"cam_{idx}",
                'source': src,
                'weights': default_weights
            })

    try:
        # OCR 워커 (공용)
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
        # 카메라 유닛 생성
        for conf in camera_configs:
            name = conf.get('name', 'unknown')
            src = conf.get('source')
            weights = conf.get('weights')
            
            if not src:
                continue
                
            try:
                cam = Camera(src)
                
                # 전담 탐지기 생성
                detector = ContainerDetector(
                    model_path=weights,
                    default_model=model_conf.get('yolo_default', 'yolo11n.pt'),
                    conf_threshold=model_conf.get('conf_threshold', 0.5)
                )
                
                camera_units.append({
                    'cam': cam,
                    'detector': detector,
                    'name': name
                })
                logger.info(f"✅ 유닛 준비 완료: {name} (Source: {src}, Model: {weights})")
                
            except Exception as e:
                logger.error(f"❌ 유닛 초기화 실패 ({name}): {e}")

        if not camera_units:
            logger.error("사용 가능한 카메라 유닛이 없습니다. 종료합니다.")
            return

    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 상태 변수 설정
    STATE_IDLE = 0
    STATE_COLLECTING = 1
    STATE_COOLDOWN = 2
    
    current_state = STATE_IDLE
    
    # 파라미터
    collection_window = params_conf.get('collection_window', 60)
    cooldown_frames = params_conf.get('cooldown_frames', 150)
    perspective_intensity = params_conf.get('perspective_intensity', 0.0)
    
    state_timer = 0
    evidence_bucket = [] 
    
    history = []
    frame_idx = 0
    temp_dir = system_conf.get('temp_frame_dir', 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)

    logger.info(f">>> {len(camera_units)}개 유닛 모니터링 시작 (수집 윈도우: {collection_window})")

    while True:
        # --- 1. 프레임 수집 ---
        active_frames = [] # (frame, unit) 튜플 리스트
        
        all_closed = True
        for unit in camera_units:
            frame = unit['cam'].get_frame()
            if frame is not None:
                active_frames.append((frame, unit))
                all_closed = False
            else:
                active_frames.append((None, unit))

        if all_closed:
            logger.info("모든 영상 소스 종료")
            break

        frame_idx += 1
        
        # 타이머 로직 (상태머신)
        if state_timer > 0:
            state_timer -= 1
            
            # 수집 종료 -> 판결
            if current_state == STATE_COLLECTING and state_timer == 0:
                logger.info(f"🛑 수집 종료! 증거 {len(evidence_bucket)}건 분석 시작...")
                
                if evidence_bucket:
                    image_paths = [item['path'] for item in evidence_bucket]
                    ocr_results = ocr_worker.process_batch(image_paths)
                    final_verdict = ocr_worker.consolidate_results(ocr_results)
                    
                    if final_verdict['found']:
                        final_num = final_verdict['container_number']
                        meta = final_verdict.get('voting_meta', {})
                        logger.info(f"★ 최종 확정: {final_num} (투표: {meta.get('winner_count')}/{meta.get('total_votes')})")
                        
                        history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'frame_id': frame_idx,
                            'container_number': final_num,
                            'voting_result': f"{meta.get('winner_count')}/{meta.get('total_votes')}"
                        })
                    else:
                        logger.info("❌ 인식 실패: 유효한 번호를 찾지 못했습니다.")
                else:
                    logger.info("❌ 수집된 증거가 없습니다.")

                current_state = STATE_COOLDOWN
                state_timer = cooldown_frames
                evidence_bucket = []
            
            # 쿨다운 종료 -> 대기
            elif current_state == STATE_COOLDOWN and state_timer == 0:
                current_state = STATE_IDLE
                logger.info("🟢 대기 모드 전환 (IDLE)")

        # --- 2. 유닛별 탐지 및 수집 ---
        display_frames = []
        
        for frame, unit in active_frames:
            if frame is None:
                continue
                
            disp_frame = frame.copy()
            unit_name = unit['name']
            
            # 쿨다운 아니면 탐지 수행
            if current_state != STATE_COOLDOWN:
                # ★ 중요: 각 유닛의 전담 탐지기 사용
                best_box = unit['detector'].detect(frame)
                
                if best_box is not None:
                    conf = float(best_box.conf[0])
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                    
                    fh, fw = frame.shape[:2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    is_centered = (fw * 0.4 < cx < fw * 0.6) and (fh * 0.25 < cy < fh * 0.75)
                    
                    Visualizer.draw_detection(disp_frame, best_box, is_centered)
                    
                    # 수집 시작 트리거
                    if current_state == STATE_IDLE and is_centered:
                        current_state = STATE_COLLECTING
                        state_timer = collection_window
                        logger.info(f"📸 {unit_name}에서 감지! 수집 시작")
                    
                    # 수집 중
                    if current_state == STATE_COLLECTING and is_centered:
                        # ROI 저장
                        pw_pad = int((x2 - x1) * 0.1)
                        ph_pad = int((y2 - y1) * 0.1)
                        px1 = max(0, x1 - pw_pad)
                        py1 = max(0, y1 - ph_pad)
                        px2 = min(fw, x2 + pw_pad)
                        py2 = min(fh, y2 + ph_pad)
                        
                        roi_raw = frame[py1:py2, px1:px2].copy()
                        roi_pre = preprocess_for_ocr(roi_raw)
                        roi_img = apply_perspective_correction(roi_pre, intensity=perspective_intensity)
                        
                        file_path = os.path.join(temp_dir, f"{unit_name}_f{frame_idx}_{int(time.time()*1000)}.jpg")
                        cv2.imwrite(file_path, roi_img)
                        
                        evidence_bucket.append({
                            'path': file_path,
                            'score': conf,
                            'unit': unit_name
                        })
                        
                        cv2.putText(disp_frame, "COLLECTING", (px1, py1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            # 유닛 이름 표시
            cv2.putText(disp_frame, f"[{unit_name}]", (10, fh - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        
            display_frames.append(resize_frame(disp_frame, scale=0.4))

        # --- 3. 화면 병합 및 출력 ---
        if display_frames:
            combined_view = np.hstack(display_frames)
            
            status_map = {0: "IDLE", 1: "COLLECTING", 2: "COOLDOWN"}
            status_color = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
            
            s_text = f"Status: {status_map[current_state]}"
            if state_timer > 0:
                s_text += f" ({state_timer})"
            
            cv2.putText(combined_view, s_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color[current_state], 2)
            cv2.putText(combined_view, f"Evidence: {len(evidence_bucket)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Multi-Model Container Recognition', combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(']'):
            perspective_intensity = min(0.5, perspective_intensity + 0.01)
        elif key == ord('['):
            perspective_intensity = max(0.0, perspective_intensity - 0.01)

    # 정리
    for unit in camera_units:
        unit['cam'].release()
    cv2.destroyAllWindows()
    
    if history:
        log_path = system_conf.get('log_file', 'outputs/gate_access_log.csv')
        pd.DataFrame(history).to_csv(log_path, index=False, encoding='utf-8-sig')
        logger.info(f"로그 저장 완료: {log_path}")

if __name__ == "__main__":
    main()