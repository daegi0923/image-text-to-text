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
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resize_frame(frame, width=640):
    if frame is None: return None
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

def draw_results(frame, model, target_classes=None):
    """
    YOLO 추론 결과를 프레임에 그림 (OBB 및 AABB 모두 지원)
    """
    if frame is None: return frame

    # 1. 추론용 리사이즈 (640px)
    DETECT_W = 640
    fh, fw = frame.shape[:2]
    scale = fw / DETECT_W
    detect_h = int(fh / scale)
    
    small = cv2.resize(frame, (DETECT_W, detect_h))
    
    # 2. 추론 (conf=0.4 정도)
    results = model(small, verbose=False, conf=0.4)
    
    annotated = frame.copy()
    if not results:
        return annotated

    r = results[0]
    
    # === [A] OBB 결과 그리기 (xyxyxyxy) ===
    if hasattr(r, 'obb') and r.obb is not None:
        for obb in r.obb:
            cls_id = int(obb.cls[0])
            if target_classes and cls_id not in target_classes: continue

            # 좌표 복원 (xyxyxyxy -> 4 points)
            pts = obb.xyxyxyxy[0].cpu().numpy()
            pts[:, 0] *= scale
            pts[:, 1] *= scale
            pts = pts.astype(np.int32)
            
            conf = float(obb.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            
            # 색상 (OBB는 조금 더 진하게)
            color = (0, 255, 0) # 기본 초록
            if cls_id == 0: color = (0, 0, 255) # 트럭 빨강
            elif cls_id == 1: color = (255, 0, 0) # 컨테이너 파랑
            elif cls_id == 2: color = (0, 255, 0) # 코드 초록
            
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=3)
            cv2.putText(annotated, label, (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # === [B] 일반 Box 결과 그리기 (xyxy) ===
    if hasattr(r, 'boxes') and r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if target_classes and cls_id not in target_classes: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, x2 = int(x1 * scale), int(x2 * scale)
            y1, y2 = int(y1 * scale), int(y2 * scale)
            
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            
            # 일반 박스는 노란색 계열
            color = (0, 255, 255)
            if cls_id == 0: color = (0, 100, 255)
            elif cls_id == 1: color = (255, 100, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # OBB랑 겹치면 글자 안 보이니 약간 위로
            cv2.putText(annotated, label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    return annotated

def main():
    print("=== 🕵️‍♂️ [OBB+Box] 실시간 디버그 모니터 ===")
    print("OBB 모델과 일반 모델 모두 시각화합니다.")
    print("종료: Q")

    config = load_config()
    sys_conf = config.get('system', {})
    
    units = []
    model_cache = {}

    for conf in sys_conf.get('cameras', []):
        name = conf.get('name')
        src = conf.get('source')
        weights = conf.get('weights')
        targets = conf.get('target_classes')
        
        if not weights: continue
            
        if not os.path.exists(weights):
             weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), weights)
             
        if weights not in model_cache:
            try:
                print(f"⏳ 모델 로딩: {os.path.basename(weights)}...")
                model_cache[weights] = YOLO(weights)
            except Exception as e:
                print(f"❌ 모델 로드 실패 ({weights}): {e}")
                continue
        
        try:
            cam = Camera(src)
            units.append({
                'name': name,
                'cam': cam,
                'model': model_cache[weights],
                'targets': targets,
                'last_frame': None
            })
            print(f"✅ 카메라 연결: {name}")
        except Exception as e:
            print(f"❌ 카메라 연결 실패 ({name}): {e}")

    if not units:
        print("사용 가능한 카메라 유닛이 없습니다.")
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
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
            
            # 그리기
            out_frame = draw_results(frame, unit['model'], unit['targets'])
            
            disp = resize_frame(out_frame, width=640)
            unit['last_frame'] = frame 
            
            cv2.putText(disp, f"[{unit['name']}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            display_list.append(disp)

        # 화면 배치 (Grid)
        if len(display_list) == 4:
            top = np.hstack(display_list[:2])
            bot = np.hstack(display_list[2:])
            grid = np.vstack([top, bot])
        elif len(display_list) > 1:
            grid = np.hstack(display_list)
        elif len(display_list) == 1:
            grid = display_list[0]
        else:
            break

        final_view = resize_frame(grid, width=1280)
        cv2.imshow("Debug Monitor (OBB)", final_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for u in units:
        u['cam'].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
