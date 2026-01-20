import cv2
import os
import math
import numpy as np
from glob import glob

# ==========================================
# [설정] 데이터셋 경로
# ==========================================
DATASET_DIR = 'container-recognition-system/yolo_dataset_obb'
OUTPUT_DIR = 'container-recognition-system/output_viz'
# ==========================================

def get_box_points(cx, cy, w, h, angle_rad):
    # 라디안 -> 도 (OpenCV용)
    angle_deg = math.degrees(angle_rad)
    # OpenCV: ((cx, cy), (w, h), angle_deg)
    rect = ((cx, cy), (w, h), angle_deg)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box

def visualize():
    img_dir = os.path.join(DATASET_DIR, 'images')
    lbl_dir = os.path.join(DATASET_DIR, 'labels')
    
    if not os.path.exists(img_dir):
        print("❌ 데이터셋 폴더 없음")
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
            # format: class cx cy w h rotation(rad)
            cls_id = int(parts[0])
            cx, cy, w, h, angle = parts[1], parts[2], parts[3], parts[4], parts[5]
            
            # 절대 좌표 변환
            abs_cx = cx * w_img
            abs_cy = cy * h_img
            abs_w = w * w_img
            abs_h = h * h_img
            
            # 박스 좌표 계산
            box = get_box_points(abs_cx, abs_cy, abs_w, abs_h, angle)
            
            # 그리기
            color = (0, 255, 0) # Green
            if cls_id == 0: color = (0, 0, 255) # Truck (Red)
            elif cls_id == 1: color = (255, 0, 0) # Container (Blue)
            
            cv2.drawContours(img, [box], 0, color, 3)
            
            # 중심점 표시
            cv2.circle(img, (int(abs_cx), int(abs_cy)), 5, (0, 255, 255), -1)
            
            # 방향 표시 (첫 번째 점이랑 중심점 연결) -> 각도 확인용
            cv2.line(img, (int(abs_cx), int(abs_cy)), tuple(box[0]), (255, 0, 255), 2)

        # 저장
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"viz_{fname}.jpg"), img)

    print(f"✅ 확인 완료! '{OUTPUT_DIR}' 폴더를 열어보세요.")

if __name__ == "__main__":
    visualize()
