import cv2
import yaml
import os
import sys
from ultralytics import YOLO

# 경로 설정 (프로젝트 루트 기준)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, 'configs', 'settings.yaml')

def load_config():
    if not os.path.exists(SETTINGS_PATH):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {SETTINGS_PATH}")
        sys.exit(1)
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_debug():
    print("=== 🕵️‍♂️ 모델 생(Raw) 성능 테스트 모드 ===")
    print("ROI 무시, 필터 무시. 모델이 보는 그대로 다 보여줍니다.")
    print("명령어: [n] 다음 카메라 / [q] 종료 / [Space] 일시정지")
    
    config = load_config()
    cameras = config.get('system', {}).get('cameras', [])
    
    if not cameras:
        print("설정된 카메라가 없습니다.")
        return

    for cam_conf in cameras:
        name = cam_conf.get('name', 'Unknown')
        source = cam_conf.get('source')
        weights = cam_conf.get('weights')
        
        # 경로 보정 (상대 경로일 경우)
        if not os.path.isabs(source):
            # source = os.path.join(BASE_DIR, source)
            pass
        if not os.path.isabs(weights):
            weights = os.path.join(BASE_DIR, weights)

        print(f"\n🎥 [테스트 중] {name}")
        print(f" - 소스: {source}")
        print(f" - 모델: {weights}")

        # if not os.path.exists(source):
        #     pass
        #     print(f"⚠️ 소스 파일 없음, 건너뜀: {source}")
        #     continue
            
        try:
            model = YOLO(weights)
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            continue

        cap = cv2.VideoCapture(source)
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("영상 종료. 다음 카메라로...")
                    break
            
                # 보기 좋게 리사이즈 (너무 크면 줄임)
                if frame.shape[1] > 1280:
                    frame = cv2.resize(frame, (1280, 720))

                # --- 핵심: 쌩 추론 ---
                # conf=0.25 (기본값) -> 너무 낮으면 쓰레기까지 다 잡음
                results = model(frame, verbose=False, conf=0.25)
                
                # YOLO 내장 시각화 기능 (박스, 라벨, 점수 다 그려줌)
                annotated_frame = results[0].plot()

            cv2.imshow(f"DEBUG: {name}", annotated_frame)
            
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                print("종료합니다.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                print("다음 카메라로 넘깁니다.")
                break
            elif key == ord(' '): # 스페이스바
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_debug()
