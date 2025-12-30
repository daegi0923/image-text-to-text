import cv2
import time
import pandas as pd
from datetime import datetime
import os

from utils.config import load_config
from utils.logger import setup_logger
from utils.visualizer import Visualizer
from utils.image_utils import apply_perspective_correction, preprocess_for_ocr
from drivers.camera import Camera
from core.detector import ContainerDetector
from core.fusion_engine import FusionEngine
from services.ocr_worker import ContainerOCR

def main():
    # 1. 설정 및 로거 초기화
    config = load_config()
    system_conf = config.get('system', {})
    model_conf = config.get('model', {})
    params_conf = config.get('parameters', {})
    
    logger = setup_logger(log_file=system_conf.get('log_file', 'outputs/gate_log.csv'))
    logger.info("시스템 시작...")

    # 2. 모듈 초기화
    try:
        # 카메라
        video_path = system_conf.get('video_path', 'data/raw_videos/gate_side2.mp4')
        camera = Camera(video_path)
        
        # 탐지기
        detector = ContainerDetector(
            model_path=model_conf.get('yolo_path', 'outputs/yolo_container_ocr/weights/best.pt'),
            default_model=model_conf.get('yolo_default', 'yolo11n.pt'),
            conf_threshold=model_conf.get('conf_threshold', 0.5)
        )
        
        # OCR
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
        # 퓨전 엔진 (투표 로직)
        fusion_engine = FusionEngine()
        
    except Exception as e:
        logger.error(f"초기화 실패: {e}")
        return

    # 변수 설정
    cooldown_frames = params_conf.get('cooldown_frames', 150)
    cooldown_counter = 0
    perspective_intensity = params_conf.get('perspective_intensity', 0.0)
    
    history = []
    frame_idx = 0
    temp_roi_path = os.path.join(system_conf.get('temp_frame_dir', 'temp_frames'), 'roi_capture.jpg')
    os.makedirs(os.path.dirname(temp_roi_path), exist_ok=True)

    logger.info(">>> 모니터링 시작 (종료: 'q')")

    while True:
        frame = camera.get_frame()
        if frame is None:
            break
            
        frame_idx += 1
        if cooldown_counter > 0:
            cooldown_counter -= 1
            
        # 1. 탐지
        best_box = detector.detect(frame)
        
        # 2. 로직 처리
        roi_img = None
        is_centered = False
        
        if best_box is not None:
            # 중앙 정렬 확인
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
            fh, fw = frame.shape[:2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            is_centered = (fw * 0.4 < cx < fw * 0.6) and (fh * 0.25 < cy < fh * 0.75)
            
            # 박스 그리기
            Visualizer.draw_detection(frame, best_box, is_centered)
            
            # OCR 수행 조건: 쿨다운 아님 & 중앙 정렬됨
            if cooldown_counter == 0 and is_centered:
                # ROI 추출 (Padding)
                pw_pad = int((x2 - x1) * 0.1)
                ph_pad = int((y2 - y1) * 0.1)
                px1 = max(0, x1 - pw_pad)
                py1 = max(0, y1 - ph_pad)
                px2 = min(fw, x2 + pw_pad)
                py2 = min(fh, y2 + ph_pad)
                
                roi_raw = frame[py1:py2, px1:px2].copy()
                
                # 전처리 (Resize, Sharpen, Perspective)
                roi_pre = preprocess_for_ocr(roi_raw)
                roi_img = apply_perspective_correction(roi_pre, intensity=perspective_intensity)
                
                # 디스크에 저장 (OCR 모델 입력용)
                cv2.imwrite(temp_roi_path, roi_img)
                
                # OCR 실행
                ocr_result = ocr_worker.extract_container_number(temp_roi_path)
                
                if ocr_result['found']:
                    num = ocr_result['container_number']
                    Visualizer.draw_ocr_result(frame, num)
                    
                    if ocr_result.get('check_digit_valid', False):
                        fusion_engine.add_prediction(num)
                        
                        consensus, count = fusion_engine.get_consensus()
                        if consensus:
                            logger.info(f"★ 컨테이너 인식 확정: {consensus}")
                            history.append({
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'frame_id': frame_idx,
                                'container_number': consensus
                            })
                            # 쿨다운 및 초기화
                            cooldown_counter = cooldown_frames
                            fusion_engine.clear()

        # 3. 상태 표시
        Visualizer.draw_status(frame, cooldown_counter, perspective_intensity)
        Visualizer.draw_roi_preview(frame, roi_img, f"Int: {perspective_intensity:.2f}")

        cv2.imshow('Container Recognition System', frame)
        
        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(']'):
            perspective_intensity = min(0.5, perspective_intensity + 0.01)
        elif key == ord('['):
            perspective_intensity = max(0.0, perspective_intensity - 0.01)

    camera.release()
    cv2.destroyAllWindows()
    
    # 결과 저장
    if history:
        log_path = system_conf.get('log_file', 'outputs/gate_access_log.csv')
        pd.DataFrame(history).to_csv(log_path, index=False, encoding='utf-8-sig')
        logger.info(f"로그 저장 완료: {log_path}")

if __name__ == "__main__":
    main()
