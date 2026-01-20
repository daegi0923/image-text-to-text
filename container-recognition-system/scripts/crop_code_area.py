import cv2
import os
import math
import numpy as np
from glob import glob

# ==========================================
# [설정] 경로
# ==========================================
DATASET_DIR = 'container-recognition-system/yolo_dataset_obb'
OUTPUT_DIR = 'container-recognition-system/yolo_dataset_obb/crops_code_area'
TARGET_CLASS_ID = 2 # CodeArea
# ==========================================

def get_box_points(cx, cy, w, h, angle_rad):
    """
    OBB 중심점, 너비, 높이, 각도(라디안) -> 4개 꼭짓점 좌표 계산
    """
    # 회전 행렬 계산 없이 간단하게 기하학적으로 계산하거나
    # OpenCV RotatedRect 포맷을 이용해서 구할 수 있음.
    
    # 각도를 Degree로 변환 (OpenCV는 Degree 사용)
    angle_deg = math.degrees(angle_rad)
    
    # OpenCV RotatedRect 포맷: ((cx, cy), (w, h), angle)
    # 주의: OpenCV 버전에 따라 angle 정의가 다를 수 있음.
    # YOLO OBB angle은 보통 x축 기준 시계방향 or 반시계방향 라디안.
    # 여기서는 단순하게 4개 점을 구해서 감싸는 rect를 만듦.
    
    rect = ((cx, cy), (w, h), angle_deg)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box

def crop_objects():
    img_dir = os.path.join(DATASET_DIR, 'images')
    lbl_dir = os.path.join(DATASET_DIR, 'labels')
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print("❌ 데이터셋 폴더를 찾을 수 없습니다.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 이미지 파일 목록
    img_files = glob(os.path.join(img_dir, '*.*'))
    print(f"📂 총 {len(img_files)}개 이미지 스캔 중...")
    
    count = 0
    
    for img_path in img_files:
        # 확장자 제외 파일명
        fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, fname + ".txt")
        
        if not os.path.exists(lbl_path):
            continue
            
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        # 라벨 읽기
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            
            # CodeArea만 타겟
            if cls_id != TARGET_CLASS_ID:
                continue
                
            # OBB 좌표 (YOLO format: class cx cy w h angle)
            # 좌표는 0~1 정규화된 값이라 가정 (YOLO 표준)
            cx, cy, w, h, angle = parts[1], parts[2], parts[3], parts[4], parts[5]
            
            # 픽셀 좌표로 변환
            abs_cx = cx * w_img
            abs_cy = cy * h_img
            abs_w = w * w_img
            abs_h = h * h_img
            
            # 회전된 사각형의 4개 점 구하기 (순서: BL, TL, TR, BR 등 회전에 따라 다름)
            # cv2.boxPoints는 순서가 보장되지 않으므로 정렬 필요
            rect = ((abs_cx, abs_cy), (abs_w, abs_h), math.degrees(angle))
            box = cv2.boxPoints(rect)
            box = np.float32(box)

            # 4개 점 정렬 (Top-Left, Top-Right, Bottom-Right, Bottom-Left 순서)
            # x좌표 합, 차 등을 이용해 순서 찾기
            # 간단하게: 
            # 1. y가 가장 작은 두 점이 Top (그 중 x 작은게 TL, 큰게 TR)
            # 2. y가 가장 큰 두 점이 Bottom (그 중 x 작은게 BL, 큰게 BR)
            # 하지만 회전이 심하면 y만으로 판단 어려움.
            
            # 일반적인 정렬 방법:
            # 합(x+y)이 가장 작은게 TL, 가장 큰게 BR
            # 차(y-x)가 가장 작은게 TR, 가장 큰게 BL
            s = box.sum(axis=1)
            tl = box[np.argmin(s)]
            br = box[np.argmax(s)]

            diff = box[:, 1] - box[:, 0] # y - x
            tr = box[np.argmin(diff)]
            bl = box[np.argmax(diff)] # 여기가 잘못될 수 있음 (좌표계 확인 필요)
            
            # 더 안정적인 정렬 (x값 기준 sort -> 좌2/우2 나누고 -> y값 기준 sort)
            # 하지만 위 방법이 일반적임.
            
            # 원본 박스 좌표 (src)
            src_pts = np.array([tl, tr, br, bl], dtype="float32")
            
            # 변환 후 좌표 (dst) - 펴진 직사각형
            # 너비/높이: OBB의 w, h 사용
            # 가로/세로 비율에 따라 눕혀지거나 세워질 수 있음 -> 긴 쪽을 가로로?
            # 일단 라벨링 된 w, h 그대로 사용
            width = int(abs_w)
            height = int(abs_h)
            
            # 만약 세로로 긴(높이가 더 큰) 박스라면, 눕혀서 저장하고 싶을 수도 있음.
            # 여기서는 있는 그대로 저장.
            
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            # 투시 변환 행렬 계산 & 적용
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            crop = cv2.warpPerspective(img, M, (width, height))
            
            # 저장 (파일명_인덱스.jpg)
            save_name = f"{fname}_{idx}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, crop)
            count += 1
            
    print(f"✅ 완료! 총 {count}개의 CodeArea를 잘라냈습니다.")
    print(f"📂 저장 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    crop_objects()
