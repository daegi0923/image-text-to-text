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

def detect_in_roi(unit, frame, scale_width=640):
    """
    특정 유닛의 ROI 내 객체 감지 여부 반환
    """
    if 'model' not in unit:
        return False, []

    h, w = frame.shape[:2]
    scale = w / scale_width
    small_h = int(h / scale)
    small_frame = cv2.resize(frame, (scale_width, small_h))
    
    # 트럭(0), 컨테이너(1)만 감지
    results = unit['model'](small_frame, verbose=False, conf=0.5, classes=[0, 1])
    
    detected = False
    boxes = []

    if results:
        for box in results[0].boxes:
            # 좌표 복원
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, x2 = int(x1 * scale), int(x2 * scale)
            y1, y2 = int(y1 * scale), int(y2 * scale)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            
            # ROI 체크
            z = unit['zone']
            zx1, zx2 = int(w*z['x_min']), int(w*z['x_max'])
            zy1, zy2 = int(h*z['y_min']), int(h*z['y_max'])
            
            if zx1 < cx < zx2 and zy1 < cy < zy2:
                detected = True
                boxes.append((x1, y1, x2, y2))
                
    return detected, boxes

def main():
    print("=== 📸 [스마트] 트럭 자동 수집기 (Dual-Check) ===")
    print("Front가 잡으면 시작 -> Back이 놓아주면 종료")
    print("메모리 최적화: 모델 캐싱 + 조건부 추론")
    print("종료: Q")

    config = load_config()
    sys_conf = config.get('system', {})
    
    # 1. 카메라 및 모델 로드 (메모리 최적화)
    cameras = []
    model_cache = {} # 같은 가중치 파일은 한 번만 로드
    
    base_save_path = "data/dataset/raw_captures"
    os.makedirs(base_save_path, exist_ok=True)
    
    # Master 찾기 및 나머지 설정
    # 주의: 여기서 'role'이 master인 놈은 항상 감시, 
    # weights가 있는 다른 놈들(Back View)은 녹화 때만 감시
    
    for conf in sys_conf.get('cameras', []):
        name = conf.get('name')
        role = conf.get('role', 'slave')
        src = conf.get('source')
        weights = conf.get('weights')
        zone = conf.get('detection_zone')
        # [수정] config에서 명시적으로 has_detector 여부를 가져옴 (기본값 False)
        config_has_detector = conf.get('has_detector', False)
        
        try:
            cam = Camera(src)
            unit = {
                'name': name,
                'role': role,
                'cam': cam,
                'zone': zone,
                'has_detector': False
            }
            
            # 모델 로드: 명시적으로 has_detector가 True이고 weights가 있는 경우만
            if config_has_detector and weights:
                # 절대 경로 변환
                if not os.path.isabs(weights):
                     weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), weights)
                
                if weights not in model_cache:
                    print(f"⚖️ 모델 로딩 (캐시): {os.path.basename(weights)}...")
                    model_cache[weights] = YOLO(weights)
                
                unit['model'] = model_cache[weights]
                unit['has_detector'] = True
                print(f"✅ 카메라: {name} (Role: {role}) [Detector Active]")
            else:
                # Master인데 detector가 없으면 경고
                if role == 'master':
                    print(f"⚠️ 경고: Master({name})에 detector 설정이 없습니다!")
                print(f"✅ 카메라: {name} (Role: {role}) [Monitor Only]")
                
            cameras.append(unit)
                
        except Exception as e:
            print(f"❌ 초기화 실패 ({name}): {e}")

    # Master(Front) 찾기
    master_unit = next((c for c in cameras if c['role'] == 'master'), None)
    if not master_unit:
        print("🚨 Master 카메라가 없습니다! settings.yaml 확인.")
        return

    # Back View 찾기 (Master가 아닌데 Detector가 있는 놈)
    assist_units = [c for c in cameras if c['role'] != 'master' and c['has_detector']]
    if assist_units:
        print(f"🤝 보조 감시 카메라(Exit Monitor): {[u['name'] for u in assist_units]}")
    else:
        print("ℹ️ 보조 감시 카메라 없음. Master 혼자 북치고 장구침.")

    # 상태 변수
    is_recording = False
    cooldown_counter = 0 
    COOLDOWN_FRAMES = 15 # 약 1~2초 여유
    current_session_dir = None
    frame_count = 0
    save_idx = 0
    
    # 중복 방지 변수
    last_saved_master_frame = None
    last_save_time = 0
    MIN_SAVE_INTERVAL = 0.5 
    FORCE_SAVE_INTERVAL = 2.0 
    MOTION_THRESHOLD = 500000 

    print(">>> 시스템 가동 <<<")

    while True:
        # 1. 모든 프레임 읽기 (Threaded Camera라 빠름)
        frames = {}
        for unit in cameras:
            f = unit['cam'].get_frame()
            if f is None: f = np.zeros((360, 640, 3), dtype=np.uint8)
            frames[unit['name']] = f

        # 2. 감지 로직 (조건부 추론)
        active_detection = False
        master_viz_boxes = []
        
        # [A] Master는 항상 감시 (진입 체크)
        master_detected, m_boxes = detect_in_roi(master_unit, frames[master_unit['name']])
        if master_detected:
            active_detection = True
            master_viz_boxes = m_boxes

        # [B] 보조 카메라는 '녹화 중일 때만' 감시 (퇴장 체크 & 자원 절약)
        if is_recording:
            for assist in assist_units:
                assist_detected, _ = detect_in_roi(assist, frames[assist['name']])
                if assist_detected:
                    active_detection = True # 보조 카메라가 보고 있으면 계속 녹화
                    # (디버그용) print(f"Back View {assist['name']} 감지 중...")

        # 3. 세션 상태 관리 (State Machine)
        if active_detection:
            cooldown_counter = COOLDOWN_FRAMES
            if not is_recording:
                # START
                is_recording = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_session_dir = os.path.join(base_save_path, f"TRUCK_{timestamp}")
                os.makedirs(current_session_dir, exist_ok=True)
                for c in cameras: os.makedirs(os.path.join(current_session_dir, c['name']), exist_ok=True)
                print(f"🎬 진입 감지! 녹화 시작 -> {timestamp}")
                save_idx = 0
                last_saved_master_frame = None
        else:
            if is_recording:
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    # STOP
                    is_recording = False
                    print(f"💾 퇴장 확인! 녹화 종료. (Frames: {save_idx})")
                    current_session_dir = None

        # 4. 저장 로직 (이전과 동일)
        if is_recording and current_session_dir:
            current_time = time.time()
            if (current_time - last_save_time) >= MIN_SAVE_INTERVAL:
                should_save = False
                m_frame = frames[master_unit['name']]
                
                if last_saved_master_frame is None:
                    should_save = True
                else:
                    # 움직임 체크
                    prev_gray = cv2.cvtColor(last_saved_master_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
                    p_small = cv2.resize(prev_gray, (320, 240))
                    c_small = cv2.resize(curr_gray, (320, 240))
                    diff = cv2.absdiff(p_small, c_small)
                    if np.sum(diff) > MOTION_THRESHOLD: should_save = True
                    elif (current_time - last_save_time) > FORCE_SAVE_INTERVAL: should_save = True

                if should_save:
                    for unit in cameras:
                        fname = f"{timestamp}_{save_idx:04d}.jpg"
                        path = os.path.join(current_session_dir, unit['name'], fname)
                        cv2.imwrite(path, frames[unit['name']])
                    save_idx += 1
                    last_saved_master_frame = m_frame.copy()
                    last_save_time = current_time

        frame_count += 1

        # 5. Master 화면 출력
        disp = frames[master_unit['name']].copy()
        z = master_unit['zone']
        h, w = disp.shape[:2]
        zx1, zx2 = int(w*z['x_min']), int(w*z['x_max'])
        zy1, zy2 = int(h*z['y_min']), int(h*z['y_max'])
        
        color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), color, 3)
        for bx in master_viz_boxes:
            cv2.rectangle(disp, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 255), 2)
            
        txt = f"REC (Back: {len(assist_units)})" if is_recording else "WAIT"
        cv2.putText(disp, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(disp, f"Saved: {save_idx}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        cv2.imshow("Smart Collector", resize_for_display(disp, width=800))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    for c in cameras: c['cam'].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()