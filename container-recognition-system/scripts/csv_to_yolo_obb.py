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
DATA_ROOT = '../../data/dataset'
OUTPUT_DIR = '../../data/dataset/yolo_dataset_obb'

CLASS_MAP = {
    'Truck': 0,
    'Container': 1,
    'CodeArea': 2
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
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # 2. 회전 행렬 적용하여 절대 좌표 계산
        rotated_pts = []
        for dx, dy in corners:
            nx = x + (dx * cos_a - dy * sin_a)
            ny = y + (dx * sin_a + dy * cos_a)
            rotated_pts.append([nx, ny])
        
        pts = np.array(rotated_pts)

        # 3. [핵심] 점 정렬 (Sorting) 로직
        # y값이 가장 작은 점(제일 위)을 시작점으로 선택
        # 만약 y가 같다면 x가 작은 점을 우선함
        start_idx = np.lexsort((pts[:, 0], pts[:, 1]))[0]
        
        # 시작점부터 시계 방향으로 재배열 (Label Studio corners가 이미 시계방향임)
        ordered_pts = np.roll(pts, -start_idx, axis=0)
        
        # 4. 8개 좌표 문자열 생성 (class_id x1 y1 ... x4 y4)
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
        
        if not os.path.exists(full_img_path):
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
        new_fname = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else os.path.basename(rel_path)
            
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', new_fname)
        dst_label_path = os.path.join(OUTPUT_DIR, 'labels', os.path.splitext(new_fname)[0] + ".txt")

        shutil.copy2(full_img_path, dst_img_path)

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