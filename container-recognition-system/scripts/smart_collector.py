import cv2
import time
import os
import sys
import yaml
import numpy as np
import shutil
from datetime import datetime
from ultralytics import YOLO

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drivers.camera import Camera

def load_config():
    path = "configs/settings.yaml"
    if not os.path.exists(path):
        path = "../configs/settings.yaml"
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resize_for_display(frame, width=480):
    if frame is None: return None
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)))

def main():
    print("=== 📸 [스마트] 트럭 자동 수집기 ===")
    print("ROI에 트럭(0)/컨테이너(1) 진입 시 자동 녹화")
    print("종료: Q")

    config = load_config()
    sys_conf = config.get('system', {})
    
    # 1. 카메라 설정
    cameras = []
    master_unit = None
    
    base_save_path = "data/dataset/smart_captures"
    os.makedirs(base_save_path, exist_ok=True)

    for conf in sys_conf.get('cameras', []):
        name = conf.get('name')
        role = conf.get('role', 'slave')
        src = conf.get('source')
        weights = conf.get('weights')
        zone = conf.get('detection_zone')
        
        try:
            cam = Camera(src)
            unit = {
                'name': name,
                'role': role,
                'cam': cam,
                'zone': zone
            }
            cameras.append(unit)
            print(f"✅ 카메라: {name} ({role})")
            
            # Master는 YOLO 모델 로드
            if role == 'master':
                print(f"⚖️ Master 모델 로딩 중: {weights}...")
                # 경로 보정
                if not os.path.exists(weights):
                     weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), weights)
                
                unit['model'] = YOLO(weights)
                master_unit = unit
                
        except Exception as e:
            print(f"❌ 초기화 실패 ({name}): {e}")

    if not master_unit:
        print("🚨 Master 카메라(Top View)가 없습니다! 설정 확인하세요.")
        return

    # 상태 변수
    is_recording = False
    cooldown_counter = 0 
    COOLDOWN_FRAMES = 15
    current_session_dir = None
    frame_count = 0
    save_idx = 0
    
    # [중복 방지] 마지막 저장 프레임 & 시간
    last_saved_master_frame = None
    last_save_time = 0
    FORCE_SAVE_INTERVAL = 1.0 # 1초 지나면 강제 저장 (정차 중이라도)
    MOTION_THRESHOLD = 500000 # 픽셀 차이 합계 (환경에 따라 조절 필요)

    print(">>> 감시 시작 (ROI 감지 대기 중) <<<")

    while True:
        # 1. 프레임 확보
        frames = {}
        for unit in cameras:
            f = unit['cam'].get_frame()
            if f is None:
                f = np.zeros((360, 640, 3), dtype=np.uint8)
            frames[unit['name']] = f

        # 2. Master 감시 (ROI 체크)
        master_frame = frames[master_unit['name']]
        mh, mw = master_frame.shape[:2]
        
        # 추론용 리사이즈 (속도)
        det_w = 640
        det_scale = mw / det_w
        det_frame = cv2.resize(master_frame, (det_w, int(mh / det_scale)))
        
        results = master_unit['model'](det_frame, verbose=False, conf=0.5, classes=[0, 1]) 
        
        detected_in_roi = False
        box_viz = [] 

        if results:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1, x2 = int(x1 * det_scale), int(x2 * det_scale)
                y1, y2 = int(y1 * det_scale), int(y2 * det_scale)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                
                z = master_unit['zone']
                zx1, zx2 = int(mw*z['x_min']), int(mw*z['x_max'])
                zy1, zy2 = int(mh*z['y_min']), int(mh*z['y_max'])
                
                if zx1 < cx < zx2 and zy1 < cy < zy2:
                    detected_in_roi = True
                    box_viz.append((x1, y1, x2, y2)) 

        # 3. 녹화 상태 관리
        if detected_in_roi:
            cooldown_counter = COOLDOWN_FRAMES 
            if not is_recording:
                # [세션 시작]
                is_recording = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_session_dir = os.path.join(base_save_path, f"TRUCK_{timestamp}")
                os.makedirs(current_session_dir, exist_ok=True)
                for c in cameras:
                    os.makedirs(os.path.join(current_session_dir, c['name']), exist_ok=True)
                print(f"🎬 녹화 시작! -> {current_session_dir}")
                save_idx = 0
                last_saved_master_frame = None # 초기화
        else:
            if is_recording:
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    is_recording = False
                    print(f"💾 녹화 종료 (총 {save_idx}세트 저장됨)")
                    current_session_dir = None

        # 4. 저장 (중복 방지 로직 적용)
        if is_recording and current_session_dir:
            current_time = time.time()
            should_save = False
            
            # (A) 첫 프레임이면 무조건 저장
            if last_saved_master_frame is None:
                should_save = True
            else:
                # (B) 움직임 감지 (간단한 차분)
                # 그레이스케일 변환 후 차이 계산이 빠름
                prev_gray = cv2.cvtColor(last_saved_master_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(master_frame, cv2.COLOR_BGR2GRAY)
                
                # 리사이즈해서 비교 (속도 향상)
                prev_small = cv2.resize(prev_gray, (320, 240))
                curr_small = cv2.resize(curr_gray, (320, 240))
                
                diff = cv2.absdiff(prev_small, curr_small)
                motion_score = np.sum(diff)
                
                # (C) 조건: 많이 움직였거나 OR 시간이 꽤 지났거나
                if motion_score > MOTION_THRESHOLD:
                    should_save = True
                elif (current_time - last_save_time) > FORCE_SAVE_INTERVAL:
                    should_save = True # 정차 중이라도 가끔 저장

            if should_save:
                for unit in cameras:
                    fname = f"{save_idx:04d}.jpg"
                    path = os.path.join(current_session_dir, unit['name'], fname)
                    cv2.imwrite(path, frames[unit['name']])
                
                save_idx += 1
                last_saved_master_frame = master_frame.copy()
                last_save_time = current_time
        
        frame_count += 1

        # 5. 화면 출력 (Master에 ROI 및 상태 표시)
        disp = master_frame.copy()
        
        # ROI 그리기
        z = master_unit['zone']
        zx1, zx2 = int(mw*z['x_min']), int(mw*z['x_max'])
        zy1, zy2 = int(mh*z['y_min']), int(mh*z['y_max'])
        
        color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), color, 3)
        
        # 감지된 객체 그리기
        for bx in box_viz:
            cv2.rectangle(disp, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 255), 2)

        # 상태 텍스트
        status = "REC" if is_recording else "WAIT"
        cv2.putText(disp, f"MODE: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(disp, f"Saved: {save_idx}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Smart Collector (Master View)", resize_for_display(disp, width=800))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 정리
    for c in cameras: c['cam'].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
