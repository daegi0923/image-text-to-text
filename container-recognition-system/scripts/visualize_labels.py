import cv2
import os
import numpy as np
from glob import glob

# ==========================================
# [설정] 데이터셋 경로
# ==========================================
DATASET_DIR = 'data/dataset/yolo_dataset_obb'
OUTPUT_DIR = 'data/output_viz'
# ==========================================

def visualize():
    img_dir = os.path.join(DATASET_DIR, 'images')
    lbl_dir = os.path.join(DATASET_DIR, 'labels')
    
    if not os.path.exists(img_dir):
        print(f"❌ 데이터셋 폴더 없음: {img_dir}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_files = glob(os.path.join(img_dir, '*.*'))
    
    print(f"🕵️‍♂️ 시각화 시작: {len(img_files)}장 확인 중...")

    for img_path in img_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, fname + ".txt")
        
        if not os.path.exists(lbl_path): continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            # format: class x1 y1 x2 y2 x3 y3 x4 y4 (8 points)
            if len(parts) < 9:
                continue
                
            cls_id = int(parts[0])
            coords = np.array(parts[1:]).reshape(-1, 2)
            
            # 정규화 좌표 -> 절대 좌표 변환
            abs_coords = coords.copy()
            abs_coords[:, 0] *= w_img
            abs_coords[:, 1] *= h_img
            box = abs_coords.astype(np.int32)
            
            # 그리기
            color = (0, 255, 0) # Green
            if cls_id == 0: color = (0, 0, 255) # Truck/Container (Red)
            elif cls_id == 1: color = (255, 0, 0) # Blue
            elif cls_id == 2: color = (0, 255, 255)
            cv2.polylines(img, [box], isClosed=True, color=color, thickness=3)
            
            # 시작점 표시 (방향 확인용)
            cv2.circle(img, tuple(box[0]), 5, (0, 255, 255), -1)
            
            # 클래스 ID 텍스트
            cv2.putText(img, f"ID:{cls_id}", tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 저장
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"viz_{fname}.jpg"), img)

    print(f"✅ 확인 완료! '{OUTPUT_DIR}' 폴더를 열어보세요.")

if __name__ == "__main__":
    visualize()