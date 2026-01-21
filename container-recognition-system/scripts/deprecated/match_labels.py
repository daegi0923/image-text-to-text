import os
import glob

IMG_DIR = 'data/dataset/yolo_dataset_obb/images'
LBL_DIR = 'data/dataset/yolo_dataset_obb/labels'

def rename_labels():
    img_files = glob.glob(os.path.join(IMG_DIR, '*.jpg'))
    lbl_files = os.listdir(LBL_DIR)
    lbl_set = set(lbl_files)
    
    print(f"이미지 파일: {len(img_files)}개")
    print(f"라벨 파일: {len(lbl_set)}개")
    
    match_count = 0
    
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        name_part = os.path.splitext(img_name)[0]
        
        # 이미지 이름: side_view_1_20260119_165144_0008
        # 예상되는 라벨 이름 후보들 확인
        
        parts = name_part.split('_')
        # 뒤에서부터 조합해가며 라벨 파일이 있는지 확인
        # 1. 0008.txt
        # 2. 165144_0008.txt
        # 3. 20260119_165144_0008.txt
        # ...
        
        found_lbl = None
        
        # 뒤에서부터 i개 요소를 합쳐서 라벨 파일명 추측
        for i in range(1, len(parts) + 1):
            candidate = "_".join(parts[-i:]) + ".txt"
            if candidate in lbl_set:
                found_lbl = candidate
                break
                
        if found_lbl:
            src = os.path.join(LBL_DIR, found_lbl)
            dst_name = name_part + ".txt" # 이미지 파일명과 동일하게 변경
            dst = os.path.join(LBL_DIR, dst_name)
            
            if src != dst:
                print(f"[변경] {found_lbl} -> {dst_name}")
                os.rename(src, dst)
                # 집합에서도 업데이트 (중복 방지 등)
                lbl_set.remove(found_lbl)
                lbl_set.add(dst_name)
                match_count += 1
            else:
                # 이미 이름이 맞음
                pass
        else:
            # 매칭되는 라벨 못 찾음
            pass

    print(f"총 {match_count}개의 라벨 파일 이름을 변경했습니다.")

if __name__ == "__main__":
    rename_labels()
