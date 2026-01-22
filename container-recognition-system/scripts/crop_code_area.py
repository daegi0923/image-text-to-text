import cv2
import os
import math
import numpy as np
from glob import glob

# ==========================================
# [설정] 경로
# ==========================================
DATASET_DIR = 'data/dataset/yolo_dataset_obb'
OUTPUT_DIR = 'data/yolo_dataset_obb/crops_code_area'
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
                
            # [Fix] YOLO OBB 포맷: class x1 y1 x2 y2 x3 y3 x4 y4 (총 9개 값)
            if len(parts) < 9:
                continue

            # 좌표 정규화 해제 (0~1 -> 픽셀)
            coords = np.array(parts[1:9], dtype=np.float32).reshape(4, 2)
            coords[:, 0] *= w_img
            coords[:, 1] *= h_img
            
            # 4개 점 정렬 (Top-Left부터 시계 방향 or 그에 준하는 순서)
            # 순서: TL, TR, BR, BL
            # x값 기준 sort -> 좌2, 우2
            # 좌2 중 y작은게 TL, 큰게 BL
            # 우2 중 y작은게 TR, 큰게 BR
            
            sorted_x = coords[np.argsort(coords[:, 0])]
            left_pts = sorted_x[:2]
            right_pts = sorted_x[2:]
            
            tl = left_pts[np.argmin(left_pts[:, 1])]
            bl = left_pts[np.argmax(left_pts[:, 1])]
            tr = right_pts[np.argmin(right_pts[:, 1])]
            br = right_pts[np.argmax(right_pts[:, 1])]
            
            src_pts = np.array([tl, tr, br, bl], dtype="float32")
            
            # 변환 후 크기 계산 (직사각형으로 펴기)
            # 상단/하단 너비 중 최대값
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            # 좌측/우측 높이 중 최대값
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # 목표 좌표 (Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left)
            dst_pts = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # 투시 변환 행렬 계산 & 적용
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            crop = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            
            # 저장 (파일명_인덱스.jpg)
            save_name = f"{fname}_{idx}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, crop)
            count += 1
            
    print(f"✅ 완료! 총 {count}개의 CodeArea를 잘라냈습니다.")
    print(f"📂 저장 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    crop_objects()
