import cv2
import time
import os
import sys
import yaml
import numpy as np
from datetime import datetime

# 프로젝트 루트 경로 추가 (모듈 import용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivers.camera import Camera

# 설정 로드
def load_config():
    path = "configs/settings.yaml"
    if not os.path.exists(path):
        # 스크립트 실행 위치에 따라 경로 보정
        path = "../configs/settings.yaml"
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

def main():
    print("=== 📸 다중 카메라 데이터 수집기 ===")
    print(" [R] : 연속 촬영 On/Off (0.2초 간격)")
    print(" [Space] : 1회 스냅샷")
    print(" [Q] : 종료")

    config = load_config()
    sys_conf = config.get('system', {})
    
    # 저장 경로 설정
    base_save_dir = "data/dataset/collected_captures"
    
    cameras = []
    for conf in sys_conf.get('cameras', []):
        name = conf.get('name')
        src = conf.get('source')
        
        # 저장 폴더 생성
        save_dir = os.path.join(base_save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            cam = Camera(src)
            cameras.append({
                'name': name,
                'cam': cam,
                'save_dir': save_dir
            })
            print(f"✅ 카메라 로드: {name}")
        except Exception as e:
            print(f"❌ 카메라 실패 ({name}): {e}")

    if not cameras:
        print("사용 가능한 카메라가 없습니다.")
        return

    recording = False
    last_record_time = 0
    record_interval = 0.4 # 0.2초마다 저장 (너무 빠르면 중복 많음)
    total_saved = 0

    while True:
        current_time = time.time()
        frames_to_show = []
        captured_this_loop = False

        # 1. 프레임 읽기
        current_frames = {} # {name: frame}
        for unit in cameras:
            frame = unit['cam'].get_frame()
            if frame is None:
                # 프레임 없으면 검은 화면
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
            
            current_frames[unit['name']] = frame
            
            # 화면 표시용 리사이즈
            disp = resize_frame(frame, width=480)
            
            # 녹화 중 표시 (빨간 테두리)
            if recording:
                cv2.rectangle(disp, (0,0), (disp.shape[1], disp.shape[0]), (0,0,255), 3)
                cv2.circle(disp, (30, 30), 10, (0,0,255), -1)
            
            cv2.putText(disp, unit['name'], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            frames_to_show.append(disp)

        # 2. 저장 로직 (연속 or 스냅샷)
        key = cv2.waitKey(1) & 0xFF
        
        # [Trigger 조건]
        save_now = False
        if key == ord(' '): # 스페이스바 (단발)
            save_now = True
            print("📸 스냅샷 찰칵!")
        elif recording and (current_time - last_record_time > record_interval): # 연속 촬영
            save_now = True
            last_record_time = current_time

        # [Save Action]
        if save_now:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19] # 밀리초 포함
            for unit in cameras:
                frame = current_frames.get(unit['name'])
                if frame is not None and frame.shape[0] > 0:
                    filename = f"{timestamp}.jpg"
                    path = os.path.join(unit['save_dir'], filename)
                    cv2.imwrite(path, frame)
            total_saved += 4 # 4대 기준
            # print(f"💾 저장 완료 ({total_saved}장 누적)")

        # 3. 화면 출력 (Grid)
        # 4개면 2x2, 아니면 가로로 쭉
        if len(frames_to_show) == 4:
            top = np.hstack(frames_to_show[:2])
            bot = np.hstack(frames_to_show[2:])
            grid = np.vstack([top, bot])
        else:
            grid = np.hstack(frames_to_show)

        # 상태 메시지
        status = f"REC (Total: {total_saved})" if recording else f"IDLE (Total: {total_saved})"
        cv2.putText(grid, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255) if recording else (255,255,255), 2)
        
        cv2.imshow("Data Collector", grid)

        # 키 조작
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            if recording:
                print("🔴 연속 촬영 시작 (0.2s 간격)")
            else:
                print("⚪ 연속 촬영 중지")

    # 종료
    for unit in cameras:
        unit['cam'].release()
    cv2.destroyAllWindows()
    print(f"👋 종료. 총 {total_saved}장 저장됨.")

if __name__ == "__main__":
    main()