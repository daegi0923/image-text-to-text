import cv2
import logging
import os
import pandas as pd
import numpy as np  # 행렬 연산을 위해 추가
from datetime import datetime
from collections import deque, Counter
from ultralytics import YOLO
from container_ocr import ContainerOCR

# =================================================================================================
# 설정
# =================================================================================================

VIDEO_PATH = 'videos/gate_side1.mp4'

OUTPUT_DIR = 'outputs'
TEMP_FRAME_DIR = 'temp_frames'
LOG_FILE = os.path.join(OUTPUT_DIR, 'gate_access_log.csv')
TEMP_ROI_IMG = os.path.join(TEMP_FRAME_DIR, 'roi_capture.jpg')

CONF_THRESHOLD = 0.5 
COOLDOWN_FRAMES = 150 

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def ensure_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(TEMP_FRAME_DIR):
        os.makedirs(TEMP_FRAME_DIR)

def apply_perspective_correction(image, intensity=0.15):
    """
    아래에서 위로 찍은 영상의 원근 왜곡을 보정합니다.
    이미지의 윗부분(원거리)이 좁게 보이는 현상을 펴줍니다.
    
    Args:
        image: 입력 ROI 이미지
        intensity (float): 보정 강도 (0.0 ~ 0.5). 값이 클수록 윗부분을 더 많이 늘림.
    """
    h, w = image.shape[:2]
    
    # 원본 이미지에서 '위가 좁은 사다리꼴' 영역을 정의
    # (실제로는 직사각형이지만 원근감 때문에 좁아 보이는 영역을 지정)
    dx = int(w * intensity)
    
    src_points = np.float32([
        [dx, 0],            # Top-Left (안쪽으로 들어감)
        [w - dx, 0],        # Top-Right (안쪽으로 들어감)
        [w, h],             # Bottom-Right (그대로)
        [0, h]              # Bottom-Left (그대로)
    ])
    
    # 이를 반듯한 직사각형으로 매핑 (결과적으로 윗부분이 늘어남)
    dst_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    
    # 변환 행렬 계산 및 적용
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected = cv2.warpPerspective(image, M, (w, h))
    
    return corrected

class GateSystem:
    def __init__(self):
        ensure_dirs()
        logging.info(">>> 시스템 초기화 중...")
        my_model_path = 'outputs/yolo_container_ocr/weights/best.pt' 
        if os.path.exists(my_model_path):
            logging.info(f"학습된 커스텀 모델 로드 중: {my_model_path}")
            self.yolo = YOLO(my_model_path)
        else:
            logging.warning("커스텀 모델을 찾을 수 없어 기본 모델(yolo11n.pt)을 사용합니다.")
            self.yolo = YOLO('yolo11n.pt')             
        logging.info("Qwen3-VL OCR 모델 로딩...")
        self.ocr_engine = ContainerOCR()
        
        self.history = []
        self.cooldown_counter = 0
        self.detection_buffer = deque(maxlen=5)
        
        # 투시 변환 초기 강도
        self.perspective_intensity = 0.0
        
    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            logging.error(f"영상을 열 수 없습니다: {VIDEO_PATH}")
            return

        frame_idx = 0
        
        logging.info(">>> 전체 화면 모니터링 시작")
        logging.info(">>> [조작] 'q': 종료, ']': 보정 강화, '[': 보정 약화")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            # 1. YOLO 감지
            results = self.yolo(frame, conf=CONF_THRESHOLD, verbose=False)
            
            # 가장 신뢰도 높은 박스 하나만 선택 (여러 개일 경우)
            best_box = None
            max_conf = 0
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box

            # 2. 박스가 있으면 OCR 처리
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                
                # 중앙 정렬 확인 (가로 1/3 ~ 2/3, 세로 1/4 ~ 3/4)
                fh, fw = frame.shape[:2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                is_centered = (fw * 0.33 < cx < fw * 0.66) and (fh * 0.25 < cy < fh * 0.75)

                # 시각화 (중앙이면 초록, 아니면 빨강)
                box_color = (0, 255, 0) if is_centered else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                if self.cooldown_counter == 0:
                    if not is_centered:
                        cv2.putText(frame, "MOVE TO CENTER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        roi_img = np.array([]) # 빈 이미지로 설정하여 아래 로직 스킵
                    else:
                        # YOLO 박스 영역 Crop (Padding 추가)
                        pw_pad = int((x2 - x1) * 0.1)  # 가로 10% 패딩
                        ph_pad = int((y2 - y1) * 0.1)  # 세로 10% 패딩
                        
                        px1 = max(0, x1 - pw_pad)
                        py1 = max(0, y1 - ph_pad)
                        px2 = min(fw, x2 + pw_pad)
                        py2 = min(fh, y2 + ph_pad)
                        
                        roi_img = frame[py1:py2, px1:px2]
                    
                    if roi_img.size > 0:
                        # 동적 강도 적용 (Crop된 이미지에 대해 보정)
                        corrected_img = apply_perspective_correction(roi_img, intensity=self.perspective_intensity)
                        
                        cv2.imwrite(TEMP_ROI_IMG, corrected_img)
                        
                        # 미리보기 (보정된 이미지)
                        try:
                            # 비율 유지하며 리사이즈
                            ph, pw = corrected_img.shape[:2]
                            target_w = 200
                            target_h = int(target_w * (ph / pw))
                            
                            if target_h > 200: # 높이가 너무 크면 높이 기준으로 맞춤
                                target_h = 200
                                target_w = int(target_h * (pw / ph))
                            
                            preview = cv2.resize(corrected_img, (target_w, target_h))
                            frame[0:target_h, 0:target_w] = preview
                            cv2.putText(frame, f"Intensity: {self.perspective_intensity:.2f}", (5, target_h - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        except:
                            pass # 이미지가 너무 작거나 오류 시 패스

                        # OCR 실행
                        ocr_result = self.ocr_engine.extract_container_number(TEMP_ROI_IMG)
                        
                        if ocr_result['found']:
                            num = ocr_result['container_number']
                            if ocr_result.get('check_digit_valid', False):
                                self.detection_buffer.append(num)
                                
                                if len(self.detection_buffer) >= 3:
                                    most_common, count = Counter(self.detection_buffer).most_common(1)[0]
                                    if count >= 3:
                                        self._save_log(most_common, frame_idx)
                                        self.cooldown_counter = COOLDOWN_FRAMES
                                        self.detection_buffer.clear()
                                        logging.info(f"★ SAVED: {most_common}")

                            cv2.putText(frame, f"Detecting: {num}", (160, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 상태 표시
            status = "COOLDOWN" if self.cooldown_counter > 0 else "READY"
            color = (0, 0, 255) if status == "COOLDOWN" else (0, 255, 0)
            cv2.putText(frame, status, (frame.shape[1]-150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 캘리브레이션 정보 표시
            cv2.putText(frame, f"Calib Mode: [ / ] (Val: {self.perspective_intensity:.2f})", 
                        (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('Container Gate Monitoring', frame)
            
            # 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(']'): # 강도 증가
                self.perspective_intensity = min(0.5, self.perspective_intensity + 0.01)
            elif key == ord('['): # 강도 감소
                self.perspective_intensity = max(0.0, self.perspective_intensity - 0.01)

        cap.release()
        cv2.destroyAllWindows()
        self._finalize_log()

    def _save_log(self, number, frame_id):
        self.history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'frame_id': frame_id,
            'container_number': number
        })

    def _finalize_log(self):
        if self.history:
            pd.DataFrame(self.history).to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
            logging.info(f"Done. Saved {len(self.history)} records.")

if __name__ == "__main__":
    system = GateSystem()
    system.run()