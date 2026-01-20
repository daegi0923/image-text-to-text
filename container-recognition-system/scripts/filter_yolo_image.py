import os
import shutil
import pandas as pd
from pathlib import Path

# ==========================================
# [설정] 경로를 네 환경에 맞게 꼭 수정해!
# ==========================================
CSV_PATH = 'container-recognition-system/data/bpt_gate_260120.csv'
LABEL_EXPORT_DIR = './label_studio_export/labels'  # 내보낸 .txt 파일들이 있는 폴더
IMAGE_ROOT_DIR = 'container-recognition-system/data/dataset' # 'raw_captures'가 들어있는 부모 폴더
TARGET_DIR = 'container-recognition-system/yolo_dataset_ready'
# ==========================================

def organize_data():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV를 찾을 수 없어: {CSV_PATH}")
        return

    # 폴더 생성
    os.makedirs(os.path.join(TARGET_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels'), exist_ok=True)

    # CSV 로드 (헤더 없는 경우 대비)
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ CSV 읽기 실패: {e}")
        return

    success_count = 0
    missing_image = 0
    missing_label = 0

    print("🚀 데이터 매칭 및 이동 시작...")

    for idx, row in df.iterrows():
        # 1. 정보 추출 (네가 준 샘플 기준)
        # ID(0번째), ImagePath(4번째)
        task_id = str(row.iloc[0]) 
        raw_img_path = str(row.iloc[4]).replace('/data/local-files/?d=', '')
        
        # 2. 원본 이미지 실제 경로 확인
        full_img_src = os.path.join(IMAGE_ROOT_DIR, raw_img_path)
        if not os.path.exists(full_img_src):
            # 경로가 한 단계 위일 경우 대비
            full_img_src = os.path.join(os.path.dirname(IMAGE_ROOT_DIR), raw_img_path)
            if not os.path.exists(full_img_src):
                missing_image += 1
                continue

        # 3. 라벨 파일 찾기
        # Label Studio는 보통 task-ID.txt 또는 그냥 ID.txt로 내보냄
        label_candidate_names = [
            f"{task_id}.txt",
            f"task-{task_id}.txt",
            os.path.splitext(os.path.basename(raw_img_path))[0] + ".txt" # 혹시 파일명 기준일까봐
        ]
        
        found_label_src = None
        for l_name in label_candidate_names:
            l_path = os.path.join(LABEL_EXPORT_DIR, l_name)
            if os.path.exists(l_path):
                found_label_src = l_path
                break
        
        if not found_label_src:
            missing_label += 1
            continue

        # 4. 새 이름 만들기 (중복 방지: 폴더명_파일명)
        # 예: side_view_1_20260119_165144_0009.jpg
        p = Path(raw_img_path)
        new_base_name = f"{p.parent.name}_{p.stem}"
        
        dst_img_path = os.path.join(TARGET_DIR, 'images', new_base_name + p.suffix)
        dst_lbl_path = os.path.join(TARGET_DIR, 'labels', new_base_name + ".txt")

        # 5. 복사
        try:
            shutil.copy2(full_img_src, dst_img_path)
            shutil.copy2(found_label_src, dst_lbl_path)
            success_count += 1
        except Exception as e:
            print(f"❌ 복사 에러 ({task_id}): {e}")

    print("\n=== ✨ 정리 완료 ===")
    print(f"✅ 성공: {success_count}쌍")
    print(f"⚠️ 이미지 없음: {missing_image}")
    print(f"⚠️ 라벨 못 찾음: {missing_label}")
    print(f"📂 결과물 위치: {TARGET_DIR}")

if __name__ == "__main__":
    organize_data()
