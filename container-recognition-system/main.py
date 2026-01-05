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
    logger.info("=== 타임 윈도우 기반 멀티 카메라 인식 시스템 시작 ===")

    # 2. 모듈 초기화
    cameras = []
    video_sources = system_conf.get('video_sources', [system_conf.get('video_path', 'data/raw_videos/gate_side2.mp4')])
    
    if isinstance(video_sources, str):
        video_sources = [video_sources]

    try:
        for src in video_sources:
            try:
                cam = Camera(src)
                cameras.append(cam)
                logger.info(f"카메라 연결 성공: {src}")
            except Exception as e:
                logger.error(f"카메라 연결 실패 ({src}): {e}")

        if not cameras:
            logger.error("사용 가능한 카메라가 없습니다. 종료합니다.")
            return

        detector = ContainerDetector(
            model_path=model_conf.get('yolo_path', 'outputs/yolo_container_ocr/weights/best.pt'),
            default_model=model_conf.get('yolo_default', 'yolo11n.pt'),
            conf_threshold=model_conf.get('conf_threshold', 0.5)
        )
        
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
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
    collection_window = params_conf.get('collection_window', 60) # 수집 기간 (프레임)
    cooldown_frames = params_conf.get('cooldown_frames', 150)    # 재인식 방지 쿨다운
    perspective_intensity = params_conf.get('perspective_intensity', 0.0)
    
    state_timer = 0
    evidence_bucket = [] # 수집된 이미지 정보들 {'path': ..., 'score': ..., 'cam_id': ...}
    
    history = []
    frame_idx = 0
    temp_dir = system_conf.get('temp_frame_dir', 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)

    logger.info(f">>> 모니터링 시작 (수집 윈도우: {collection_window}프레임)")

    while True:
        # --- 1. 프레임 수집 ---
        frames = []
        for cam in cameras:
            frame = cam.get_frame()
            frames.append(frame)

        if all(f is None for f in frames):
            logger.info("모든 영상 소스 종료")
            break

        frame_idx += 1
        
        # 타이머 감소
        if state_timer > 0:
            state_timer -= 1
            # 수집 종료 -> 판결 모드
            if current_state == STATE_COLLECTING and state_timer == 0:
                logger.info(f"🛑 수집 종료! 증거 {len(evidence_bucket)}건 분석 시작...")
                
                if evidence_bucket:
                    # 1. 경로 리스트 추출
                    image_paths = [item['path'] for item in evidence_bucket]
                    
                    # 2. 일괄 OCR
                    ocr_results = ocr_worker.process_batch(image_paths)
                    
                    # 3. 투표
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

                # 쿨다운 진입
                current_state = STATE_COOLDOWN
                state_timer = cooldown_frames
                evidence_bucket = [] # 버킷 비우기
            
            # 쿨다운 종료 -> IDLE
            elif current_state == STATE_COOLDOWN and state_timer == 0:
                current_state = STATE_IDLE
                logger.info("🟢 대기 모드 전환 (IDLE)")

        # --- 2. 탐지 및 수집 ---
        display_frames = []
        
        for i, frame in enumerate(frames):
            if frame is None:
                continue
                
            disp_frame = frame.copy()
            
            # 쿨다운 중엔 탐지 생략해서 자원 절약 (원하면 켜도 됨)
            if current_state != STATE_COOLDOWN:
                best_box = detector.detect(frame)
                
                if best_box is not None:
                    # 좌표 및 점수
                    conf = float(best_box.conf[0])
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                    
                    # 중앙 정렬 확인
                    fh, fw = frame.shape[:2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    is_centered = (fw * 0.4 < cx < fw * 0.6) and (fh * 0.25 < cy < fh * 0.75)
                    
                    Visualizer.draw_detection(disp_frame, best_box, is_centered)
                    
                    # [로직]
                    # IDLE 상태에서 중앙 정렬된 컨테이너 발견 -> 수집 모드 시작
                    if current_state == STATE_IDLE and is_centered:
                        current_state = STATE_COLLECTING
                        state_timer = collection_window
                        logger.info(f"📸 감지됨! 수집 모드 시작 ({collection_window}프레임)")
                    
                    # COLLECTING 상태에서 좋은 샷(중앙 정렬) 계속 수집
                    if current_state == STATE_COLLECTING and is_centered:
                        # 이미 수집된 것 중 현재 카메라의 최고 점수 확인
                        # (카메라당 너무 많이 수집되면 느려지니 제한을 둘 수도 있음)
                        
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
                        
                        file_path = os.path.join(temp_dir, f"cam{i}_f{frame_idx}_{int(time.time()*1000)}.jpg")
                        cv2.imwrite(file_path, roi_img)
                        
                        # 증거 추가
                        evidence_bucket.append({
                            'path': file_path,
                            'score': conf,
                            'cam_id': i
                        })
                        
                        cv2.putText(disp_frame, "COLLECTING", (px1, py1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            display_frames.append(resize_frame(disp_frame, scale=0.4))

        # --- 3. 화면 병합 및 상태 표시 ---
        if display_frames:
            combined_view = np.hstack(display_frames)
            
            # 상태 텍스트
            status_map = {0: "IDLE", 1: "COLLECTING", 2: "COOLDOWN"}
            status_color = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)} # G, Orange, R
            
            s_text = f"Status: {status_map[current_state]}"
            if state_timer > 0:
                s_text += f" ({state_timer})"
            
            cv2.putText(combined_view, s_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color[current_state], 2)
            cv2.putText(combined_view, f"Evidence: {len(evidence_bucket)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Multi-Camera Evidence Collector', combined_view)

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(']'):
            perspective_intensity = min(0.5, perspective_intensity + 0.01)
        elif key == ord('['):
            perspective_intensity = max(0.0, perspective_intensity - 0.01)

    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()
    
    if history:
        log_path = system_conf.get('log_file', 'outputs/gate_access_log.csv')
        pd.DataFrame(history).to_csv(log_path, index=False, encoding='utf-8-sig')
        logger.info(f"로그 저장 완료: {log_path}")

if __name__ == "__main__":
    main()
