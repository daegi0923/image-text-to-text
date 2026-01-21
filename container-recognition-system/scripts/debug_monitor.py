import cv2
import time
import os
import sys
import yaml
import numpy as np
import threading
from ultralytics import YOLO

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivers.camera import Camera

def load_config():
    path = "configs/settings.yaml"
    # 경로 보정
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

def draw_yolo_results(frame, model, target_classes=None):
    """
    YOLO 추론 결과를 프레임에 그림 (ROI 무시, 전체 탐지)
    """
    # 1. 추론용 리사이즈 (640px)
    DETECT_W = 640
    fh, fw = frame.shape[:2]
    scale = fw / DETECT_W
    detect_h = int(fh / scale)
    
    small = cv2.resize(frame, (DETECT_W, detect_h))
    
    # 2. 추론 (conf=0.4 정도)
    results = model(small, verbose=False, conf=0.4)
    
    annotated = frame.copy()
    
    # 3. 그리기 (좌표 복원)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            
            # 타겟 클래스 필터링 (설정된 것만 보고 싶으면 활성화)
            if target_classes and cls_id not in target_classes:
                continue

            # 좌표 복원
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, x2 = int(x1 * scale), int(x2 * scale)
            y1, y2 = int(y1 * scale), int(y2 * scale)
            
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            
            # 색상: 트럭(0)=빨강, 컨테이너(1)=파랑, 코드(2)=초록
            color = (0, 255, 0)
            if cls_id == 0: color = (0, 0, 255)
            elif cls_id == 1: color = (255, 0, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            
    return annotated

def main():
    print("=== 🕵️‍♂️ [종합] 실시간 탐지 모니터 ===")
    print("ROI/트리거 로직 없이, 모든 카메라의 순수 인식 성능을 보여줍니다.")
    print("종료: Q")

    config = load_config()
    sys_conf = config.get('system', {})
    
    # 카메라 및 모델 로드
    units = []
    
    # 공용 모델 캐싱 (같은 파일 쓰면 메모리 아끼기)
    model_cache = {}

    for conf in sys_conf.get('cameras', []):
        name = conf.get('name')
        src = conf.get('source')
        weights = conf.get('weights')
        targets = conf.get('target_classes')
        
        # 경로 보정
        if not os.path.exists(weights):
             weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), weights)
             
        if weights not in model_cache:
            try:
                print(f"⏳ 모델 로딩 중: {os.path.basename(weights)}...")
                model_cache[weights] = YOLO(weights)
            except Exception as e:
                print(f"❌ 모델 실패 ({weights}): {e}")
                continue
        
        try:
            cam = Camera(src)
            units.append({
                'name': name,
                'cam': cam,
                'model': model_cache[weights],
                'targets': targets,
                'last_frame': None # 화면 유지용
            })
            print(f"✅ 준비: {name}")
        except Exception as e:
            print(f"❌ 카메라 실패 ({name}): {e}")

    if not units:
        print("실행 가능한 유닛이 없습니다.")
        return

    print(">>> 모니터링 시작 <<<")

    while True:
        display_list = []
        
        for unit in units:
            frame = unit['cam'].get_frame()
            
            if frame is None:
                if unit['last_frame'] is not None:
                    frame = unit['last_frame']
                else:
                    # 빈 화면
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
            
            # 추론 & 그리기
            # 성능을 위해 2프레임마다 1번만 추론할 수도 있지만,
            # 여기선 디버깅용이니 매번 그린다. (대신 리사이즈 추론)
            out_frame = draw_yolo_results(frame, unit['model'], unit['targets'])
            
            # 화면용 축소
            disp = resize_frame(out_frame, width=640)
            unit['last_frame'] = frame # 원본 저장
            
            # 이름 표시
            cv2.putText(disp, f"[{unit['name']}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            display_list.append(disp)

        # 그리드 만들기
        if len(display_list) == 4:
            top = np.hstack(display_list[:2])
            bot = np.hstack(display_list[2:])
            grid = np.vstack([top, bot])
        elif len(display_list) > 1:
            grid = np.hstack(display_list)
        else:
            grid = display_list[0]

        # 축소 (한눈에 보기 위해)
        final_view = resize_frame(grid, width=1280)
        cv2.imshow("Debug Monitor (All Cams)", final_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 정리
    for u in units:
        u['cam'].release()
    cv2.destroyAllWindows()
    print("종료되었습니다.")

if __name__ == "__main__":
    main()
