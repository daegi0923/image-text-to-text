import os
import shutil
import random
from pathlib import Path

def organize_dataset_with_exported_labels(image_dir, label_dir, dataset_root, split_ratio=0.8):
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    root_path = Path(dataset_root)
    
    if not image_path.exists() or not label_path.exists():
        print(f"오류: 경로를 찾을 수 없습니다.")
        return

    # 라벨 파일 리스트 가져오기
    label_files = list(label_path.glob('*.txt'))
    if not label_files:
        print("라벨 파일이 없습니다.")
        return

    print(f"총 {len(label_files)}개의 라벨을 발견했습니다.")
    
    # 매칭되는 데이터 쌍 만들기
    data_pairs = []
    for lp in label_files:
        # '001b946c-frame_1923.txt' -> 'frame_1923'
        # 하이픈(-) 뒤의 부분을 이미지 이름으로 간주
        filename_parts = lp.stem.split('-', 1)
        if len(filename_parts) > 1:
            base_name = filename_parts[1]
        else:
            base_name = lp.stem
            
        # 매칭되는 이미지 찾기 (jpg, png 등)
        img_found = None
        for ext in ['.jpg', '.jpeg', '.png']:
            ip = image_path / (filename_parts[0] + '-' + base_name + ext)
            print(ip)
            if ip.exists():
                img_found = ip
                break
        
        if img_found:
            data_pairs.append((img_found, lp, base_name))
        else:
            print(f"경고: 이미지 없음 - {base_name}")

    if not data_pairs:
        print("매칭되는 이미지-라벨 쌍이 없습니다.")
        return

    print(f"총 {len(data_pairs)}개의 쌍이 매칭되었습니다.")
    
    # [수정] 랜덤 셔플 제거하고 이름순(시간순)으로 정렬
    # 파일명에 타임스탬프가 있으므로 정렬하면 시퀀셜하게 정렬됨
    data_pairs.sort(key=lambda x: x[2]) 
    
    split_idx = int(len(data_pairs) * split_ratio)
    train_data = data_pairs[:split_idx]
    val_data = data_pairs[split_idx:]
    
    # 디렉토리 생성
    for split in ['train', 'val']:
        (root_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def copy_pairs(pairs, split):
        for img_p, lbl_p, base_name in pairs:
            # 이미지 복사 (이름은 base_name으로 통일)
            shutil.copy2(img_p, root_path / 'images' / split / (base_name + img_p.suffix))
            # 라벨 복사 (이름은 base_name으로 통일)
            shutil.copy2(lbl_p, root_path / 'labels' / split / (base_name + '.txt'))

    copy_pairs(train_data, 'train')
    copy_pairs(val_data, 'val')
    
    print(f"완료: Train {len(train_data)}쌍, Val {len(val_data)}쌍")
    print(f"데이터셋 정리 완료: {root_path.absolute()}")

if __name__ == '__main__':
    # 기본 경로 설정 (프로젝트 루트 기준)
    IMAGE_DIR = 'data/raw_data_multilabel/images' 
    LABEL_DIR = 'data/raw_data_multilabel/labels'
    DATASET_ROOT = 'data/dataset_multilabel'
    
    organize_dataset_with_exported_labels(IMAGE_DIR, LABEL_DIR, DATASET_ROOT)
