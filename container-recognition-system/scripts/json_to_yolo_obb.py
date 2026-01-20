import json
import os
import shutil
import math
import cv2
import numpy as np

# ==========================================
# [설정] JSON 파일 경로
# ==========================================
JSON_PATH = 'container-recognition-system/data/project_export.json' 
DATA_ROOT = 'container-recognition-system/data/dataset' 
OUTPUT_DIR = 'container-recognition-system/yolo_dataset_final'

# [통합] 중복된 클래스 이름이나 변형된 이름을 하나로 매핑
CLASS_MAP = {
    'Truck': 0,
    'truck': 0,
    'Container': 1,
    'container': 1,
    'CodeArea': 2,
    'code_area': 2,
    'Code_Area': 2,
    'codearea': 2
}
# ==========================================

def convert_label_studio_to_yolo_obb(item, img_w, img_h):
    """
    Label Studio JSON -> YOLO OBB 변환
    """
    original_w = item.get('original_width', img_w)
    original_h = item.get('original_height', img_h)
    
    if original_w == 0 or original_h == 0:
        return 0, 0, 0, 0, 0

    # 1. % 좌표 -> 픽셀 좌표 변환
    x = (item['x'] / 100.0) * original_w
    y = (item['y'] / 100.0) * original_h
    w = (item['width'] / 100.0) * original_w
    h = (item['height'] / 100.0) * original_h
    r_deg = item.get('rotation', 0)
    
    # 2. 중심점 계산
    # Label Studio의 회전은 보통 박스 중심(Center)을 축으로 함.
    # 따라서 회전 전의 중심점이나 후의 중심점이나 같음.
    cx = x + w / 2
    cy = y + h / 2
    
    # 3. 정규화 (0~1) - YOLO 입력용
    norm_cx = cx / original_w
    norm_cy = cy / original_h
    norm_w = w / original_w
    norm_h = h / original_h
    
    # 4. 각도 변환 (Degree -> Radian)
    # YOLO v8 OBB는 라디안 사용
    r_rad = math.radians(r_deg)

    return norm_cx, norm_cy, norm_w, norm_h, r_rad

def main():
    if not os.path.exists(JSON_PATH):
        print(f"❌ JSON 파일을 찾을 수 없습니다: {JSON_PATH}")
        print("Label Studio -> Export -> JSON 으로 다운받은 파일 경로를 설정해주세요.")
        return

    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🚀 총 {len(data)}개의 작업(Task) 처리 시작...")
    
    success_cnt = 0
    
    for task in data:
        # 1. 이미지 경로 찾기
        img_data = task.get('data', {})
        # 여러 키 중 하나라도 있으면 됨
        raw_path = img_data.get('image') or img_data.get('img') or img_data.get('file_upload')
        
        if not raw_path:
            continue
            
        # 경로 정리
        rel_path = raw_path.replace('/data/local-files/?d=', '')
        
        # 실제 파일 확인
        src_path = os.path.join(DATA_ROOT, rel_path)
        if not os.path.exists(src_path):
            if 'raw_captures' in rel_path:
                part = rel_path.split('raw_captures')[-1].strip(os.sep)
                src_path = os.path.join(DATA_ROOT, 'raw_captures', part)
            
            if not os.path.exists(src_path):
                print(f"⚠️ 이미지 없음: {rel_path}")
                continue

        # 2. 유니크 파일명 생성
        path_obj = os.path.normpath(rel_path)
        parts = path_obj.split(os.sep)
        if len(parts) >= 2:
            new_fname = f"{parts[-2]}_{parts[-1]}"
        else:
            new_fname = os.path.basename(rel_path)
            
        dst_img = os.path.join(OUTPUT_DIR, 'images', new_fname)
        dst_lbl = os.path.join(OUTPUT_DIR, 'labels', os.path.splitext(new_fname)[0] + ".txt")

        # 3. 이미지 복사
        shutil.copy2(src_path, dst_img)
        
        # 메타데이터 읽기
        img_w, img_h = 0, 0
        temp_img = cv2.imread(src_path)
        if temp_img is not None:
            img_h, img_w = temp_img.shape[:2]
        
        # 4. 라벨 변환 및 저장
        yolo_lines = []
        annotations = task.get('annotations', [])
        
        for ann in annotations:
            result = ann.get('result', [])
            for res in result:
                if res['type'] != 'rectanglelabels':
                    continue
                
                # [수정된 부분] 라벨 이름 정규화
                raw_label = res['value']['rectanglelabels'][0]
                label_name = raw_label.strip() # 공백 제거
                
                cls_id = CLASS_MAP.get(label_name)
                
                # 못 찾으면 소문자로 다시 시도
                if cls_id is None:
                    cls_id = CLASS_MAP.get(label_name.lower())
                
                if cls_id is None:
                    print(f"⚠️ 알 수 없는 라벨: {label_name} (Task ID: {task.get('id')})")
                    continue
                
                cx, cy, w, h, r = convert_label_studio_to_yolo_obb(res['value'], img_w, img_h)
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {r:.6f}")
        
        with open(dst_lbl, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
            
        success_cnt += 1

    print(f"✨ 변환 완료: {success_cnt}건")
    print(f"📂 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()