import pandas as pd
import json
import os
import shutil
import math
import ast
import numpy as np

# ==========================================
# [설정] 경로 및 클래스 매핑
# ==========================================
CSV_FILE = 'data/bpt_gate_260120.csv'
DATA_ROOT = 'data/dataset/yolo_dataset_obb/images'
OUTPUT_DIR = 'data/dataset/yolo_dataset_obb_test'

CLASS_MAP = {
    'Truck': 0,
    'Container': 1,
    'code_area': 2
}
# ==========================================

def convert_to_yolo_obb(item, img_w, img_h):
    try:
        label_name = item['rectanglelabels'][0]
        cls_id = CLASS_MAP.get(label_name)
        if cls_id is None:
            return None 

        # Label Studio: x, y, width, height (0-100%)
        # x, y는 회전 축(Pivot)인 Top-Left 기준임
        x = item['x'] / 100.0
        y = item['y'] / 100.0
        w = item['width'] / 100.0
        h = item['height'] / 100.0
        r_deg = item.get('rotation', 0)
        r_rad = math.radians(r_deg)
        
        cos_a = math.cos(r_rad)
        sin_a = math.sin(r_rad)

        # 1. 회전 전 4개 꼭짓점의 상대 좌표 (Pivot 기준)
        # 순서: TL, TR, BR, BL
        raw_corners = [
            (x + (0 * cos_a - 0 * sin_a), y + (0 * sin_a + 0 * cos_a)), # TL
            (x + (w * cos_a - 0 * sin_a), y + (w * sin_a + 0 * cos_a)), # TR
            (x + (w * cos_a - h * sin_a), y + (w * sin_a + h * cos_a)), # BR
            (x + (0 * cos_a - h * sin_a), y + (0 * sin_a + h * cos_a))  # BL
        ]

        # 2. [중요] 점 정렬 (Sorting)
        # y값이 가장 작은 점(가장 위)을 찾거나, 중심점 기준으로 각도 정렬
        pts = np.array(raw_corners)
        
        # x+y가 가장 작은 점을 시작점으로 잡는 방식 (가장 좌상단)
        sum_pts = pts.sum(axis=1)
        start_idx = np.argmin(sum_pts)
        
        # 시작점부터 시계 방향으로 재배열
        ordered_pts = np.roll(pts, -start_idx, axis=0)
        
        # 3. 8개 좌표 문자열 생성
        formatted_coords = " ".join([f"{p:.6f}" for p in ordered_pts.flatten()])
        return f"{cls_id} {formatted_coords}"
    except Exception as e:
        print(f"변환 에러: {e}")
        return None

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV 파일 없음: {CSV_FILE}")
        return

    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    print(f"📄 데이터 로드: {len(df)}건")

    success_cnt = 0
    fail_cnt = 0

    for idx, row in df.iterrows():
        # 이미지 경로 처리
        raw_path = row.get('image', '')
        if not raw_path:
            raw_path = row.iloc[4]
        
        rel_path = raw_path.replace('/data/local-files/?d=', '')
        full_img_path = os.path.join(DATA_ROOT, rel_path).replace('\\', '/')
        cam = full_img_path.split('/')[-2]
        image_name = full_img_path.split('/')[-1]
        target_image_path = os.path.join(DATA_ROOT,f'{cam}_{image_name}')
        print(target_image_path)
        if not os.path.exists(target_image_path):
            fail_cnt += 1
            continue

        # 라벨 파싱
        label_str = row.get('label')
        if not isinstance(label_str, str):
            label_str = row.iloc[5]

        try:
            labels = json.loads(label_str)
        except:
            try:
                labels = ast.literal_eval(label_str)
            except:
                continue

        # 파일명 생성 및 복사
        path_parts = rel_path.split('/')
        new_fname = f'{cam}_{image_name}'
            
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', new_fname)
        dst_label_path = os.path.join(OUTPUT_DIR, 'labels', os.path.splitext(new_fname)[0] + ".txt")

        shutil.copy2(target_image_path, dst_img_path)

        # 라벨 작성
        yolo_lines = []
        for item in labels:
            img_w = item.get('original_width', 1920)
            img_h = item.get('original_height', 1080)
            line = convert_to_yolo_obb(item, img_w, img_h)
            if line:
                yolo_lines.append(line)
        
        with open(dst_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
            
        success_cnt += 1

    print(f"🎉 변환 완료! 성공: {success_cnt}, 실패: {fail_cnt}")

if __name__ == "__main__":
    main()