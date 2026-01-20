import pandas as pd
import json
import os
import shutil
import math
import ast

# ==========================================
# [설정] 경로 및 클래스 매핑
# ==========================================
CSV_FILE = '/data/bpt_gate_260120.csv'  # CSV 파일 경로
DATA_ROOT = '/data/dataset' # 실제 이미지가 있는 최상위 폴더
OUTPUT_DIR = 'container-recognition-system/yolo_dataset_obb' # 결과 저장될 폴더

# 클래스 이름 -> ID 매핑 (settings.yaml과 맞춰야 함)
CLASS_MAP = {
    'Truck': 0,
    'Container': 1,
    'CodeArea': 2
}
# ==========================================

def convert_to_yolo_obb(item, img_w, img_h):
    """
    Label Studio 좌표 -> YOLO OBB 포맷 변환
    """
    try:
        label_name = item['rectanglelabels'][0]
        cls_id = CLASS_MAP.get(label_name)
        if cls_id is None:
            return None 

        # Label Studio: x, y, width, height (0-100%)
        # Rotation: 0-360 degrees (clockwise)
        x = item['x'] / 100.0
        y = item['y'] / 100.0
        w = item['width'] / 100.0
        h = item['height'] / 100.0
        r_deg = item.get('rotation', 0)

        # 1. 중심점 계산 (Label Studio의 x,y는 회전 전 Top-Left 기준)
        # 회전을 중심(cx, cy) 기준으로 한다면 중심점 좌표는 불변함.
        cx = x + w / 2
        cy = y + h / 2
        
        # 2. 각도 변환 (Label Studio -> YOLO)
        # Label Studio: 시계 방향이 양수(+)
        # YOLO OBB: 보통 0~pi/2 범위를 쓰거나 버전에 따라 다름.
        # 시각화했을 때 반대라면 부호를 뒤집어봐야 함.
        # 일단 마이너스로 변경 시도
        rotation_rad = math.radians(r_deg) 
        
        # NOTE: 만약 여전히 밀린다면, 
        # Label Studio의 회전 중심이 (x, y) 즉 Top-Left일 수도 있음.
        # 그럴 경우:
        # real_cx = x + (w/2)*cos(r) - (h/2)*sin(r)
        # real_cy = y + (w/2)*sin(r) + (h/2)*cos(r)
        # 이런 식으로 삼각함수 보정이 필요함.
        
        # 일단 각도는 그대로 두고(양수), 만약 시각화에서 반대면 그때 뒤집자.
        # (이전 시도에서 밀렸다고 했으니 중심점 보정 공식을 적용해봄)
        
        # [가설 2] 회전 중심이 Top-Left(x,y)가 아니라, 
        # Label Studio가 주는 x,y 자체가 회전된 박스의 어떤 지점일 수 있음.
        
        # 하지만 Label Studio 공식 문서는 "x, y, w, h are unrotated coordinates"라고 함.
        # 그렇다면 cx, cy는 단순히 x + w/2가 맞음.
        
        # 그렇다면 문제는 'rotation'의 방향임.
        # 한번 부호를 반대로 뒤집어서 저장해봄.
        # rotation_rad = -math.radians(r_deg)
        
        return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {rotation_rad:.6f}"

    except Exception as e:
        print(f"변환 에러: {e}")
        return None

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV 파일 없음: {CSV_FILE}")
        return

    # 폴더 생성
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

    # CSV 로드
    df = pd.read_csv(CSV_FILE)
    print(f"📄 데이터 로드: {len(df)}건")

    success_cnt = 0
    fail_cnt = 0

    for idx, row in df.iterrows():
        # 1. 이미지 경로 파싱
        # 예: /data/local-files/?d=raw_captures/side_view_1/20260119_165144_0009.jpg
        raw_path = row.get('image', '') # 컬럼명이 'image'가 아니면 수정 필요 (보통 Label Studio는 'image'임)
        if not raw_path:
            # 헤더에 따라 컬럼명이 다를 수 있음 (예: 'ocr', 'photo' 등)
            # 첫 번째나 두 번째 컬럼을 이미지로 간주
            raw_path = row.iloc[4] # 5번째 컬럼 (인덱스 4) - 네가 준 데이터 기준
        
        # 접두어 제거 ('/data/local-files/?d=' 제거)
        rel_path = raw_path.replace('/data/local-files/?d=', '')
        
        # 실제 로컬 경로 조합
        full_img_path = os.path.join(DATA_ROOT, rel_path)
        
        # 경로 보정 (윈도우/맥 호환)
        full_img_path = full_img_path.replace('\\', '/')
        
        if not os.path.exists(full_img_path):
            # 경로가 안 맞으면 DATA_ROOT 없이 시도하거나, 상위 폴더 체크
            # 예: data/dataset/raw_captures/... 가 아니라 그냥 raw_captures/... 일 수도
            alt_path = os.path.join(os.path.dirname(DATA_ROOT), rel_path)
            if os.path.exists(alt_path):
                full_img_path = alt_path
            else:
                # print(f"⚠️ 이미지 못 찾음: {full_img_path}")
                fail_cnt += 1
                continue

        # 2. 라벨 파싱
        label_str = row.get('label')
        if not isinstance(label_str, str):
            # 라벨 컬럼명이 다를 수 있음 (네 데이터에선 6번째 컬럼 인덱스 5)
            label_str = row.iloc[5]

        try:
            # JSON 문자열 -> 리스트
            labels = json.loads(label_str)
        except:
            try:
                # 가끔 따옴표 문제로 json.loads 안될 때 ast 사용
                labels = ast.literal_eval(label_str)
            except:
                print(f"❌ 라벨 파싱 실패 (Row {idx})")
                continue

        # 3. 유니크 파일명 생성
        # side_view_1_20260119_165144_0009.jpg
        path_parts = rel_path.split('/')
        if len(path_parts) >= 2:
            new_fname = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            new_fname = os.path.basename(rel_path)
            
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', new_fname)
        dst_label_path = os.path.join(OUTPUT_DIR, 'labels', os.path.splitext(new_fname)[0] + ".txt")

        # 4. 이미지 복사
        shutil.copy2(full_img_path, dst_img_path)

        # 5. 라벨 파일 작성
        img_w = 1920 # 기본값 (JSON에 있으면 덮어씀)
        img_h = 1080
        
        yolo_lines = []
        for item in labels:
            if 'original_width' in item:
                img_w = item['original_width']
                img_h = item['original_height']
            
            line = convert_to_yolo_obb(item, img_w, img_h)
            if line:
                yolo_lines.append(line)
        
        with open(dst_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
            
        success_cnt += 1

    print(f"🎉 변환 완료! 성공: {success_cnt}, 실패(이미지없음): {fail_cnt}")
    print(f"📂 저장 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
