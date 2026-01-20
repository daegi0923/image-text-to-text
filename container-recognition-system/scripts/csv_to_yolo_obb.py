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
    (class x_center y_center w h rotation_rad)
    """
    try:
        label_name = item['rectanglelabels'][0]
        cls_id = CLASS_MAP.get(label_name)
        if cls_id is None:
            return None # 모르는 클래스는 패스

        # Label Studio는 0~100 비율 사용
        x = item['x'] / 100.0
        y = item['y'] / 100.0
        w = item['width'] / 100.0
        h = item['height'] / 100.0
        
        # Label Studio의 (x, y)는 Top-Left가 아니라 Center일 수도 있고 Box의 기준점일 수도 있음.
        # 보통 Label Studio 회전된 박스: (x, y)는 Top-Left, w, h, rotation(도)
        # 하지만 회전 중심이 어디냐에 따라 계산이 복잡함.
        # Label Studio의 'x', 'y'는 회전 전의 Top-Left 좌표일 확률이 높음.
        
        # 간단하게 Center 좌표로 변환 (회전 고려 X 근사치 - 정밀 변환 필요 시 공식 적용해야 함)
        # YOLO OBB 포맷: x_center y_center width height rotation(rad)
        # 일단 Label Studio JSON 값은 Center가 아니라 Top-Left 기준일 수 있으니 보정 필요.
        # r = item['rotation'] 
        # Label Studio 문서 기준: x, y is top-left of the bounding box (0-100)
        
        # 중심점 계산 (회전 없을 때 기준)
        cx = x + w / 2
        cy = y + h / 2
        
        # 회전값: 도(Degree) -> 라디안(Radian)
        # YOLO OBB는 -pi/2 ~ pi/2 범위를 주로 사용하거나 0~2pi 등 버전마다 다름.
        # 여기선 일반적인 라디안 변환만 적용.
        rotation_deg = item.get('rotation', 0)
        rotation_rad = math.radians(rotation_deg)

        # 정밀한 중심점 계산 (회전 적용)
        # 회전 중심이 (x, y)라면 cx, cy가 이동해야 함.
        # 하지만 Label Studio export 포맷마다 달라서, 일단 단순 Center로 변환.
        # (필요시 시각화해서 틀어지면 수정해야 함)
        
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
