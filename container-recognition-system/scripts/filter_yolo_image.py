import os
import shutil

# 1. 경로 설정 (여기를 네 환경에 맞게 수정!)
ORIGINAL_IMAGES_DIR = './all_images'  # 원본 이미지들이 들어있는 폴더
EXPORTED_LABELS_DIR = './label_studio_export'  # Label Studio에서 다운받은 라벨 폴더
TARGET_DIR = './yolo_dataset'  # 새로 만들 데이터셋 폴더

def organize_yolo_data():
    # 폴더 생성
    os.makedirs(os.path.join(TARGET_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels'), exist_ok=True)

    # 라벨 폴더 안의 파일들 확인
    label_files = [f for f in os.listdir(EXPORTED_LABELS_DIR) if f.endswith('.txt')]
    
    count = 0
    for label_file in label_files:
        # 파일명만 추출 (확장자 제외)
        file_name = os.path.splitext(label_file)[0]
        
        # 라벨 파일 복사
        shutil.copy(
            os.path.join(EXPORTED_LABELS_DIR, label_file),
            os.path.join(TARGET_DIR, 'labels', label_file)
        )
        
        # 대응하는 이미지 찾기 (jpg, png, jpeg 등 체크)
        found_image = False
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            image_path = os.path.join(ORIGINAL_IMAGES_DIR, file_name + ext)
            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(TARGET_DIR, 'images', file_name + ext))
                found_image = True
                break
        
        if found_image:
            count += 1
        else:
            print(f"⚠️ 이미지가 없음: {file_name}")

    print(f"✅ 정리 완료! 총 {count}쌍의 데이터가 이동됨.")

if __name__ == "__main__":
    organize_yolo_data()