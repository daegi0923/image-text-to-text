import cv2
import numpy as np

def apply_perspective_correction(image, intensity=0.15):
    """
    아래에서 위로 찍은 영상의 원근 왜곡을 보정합니다.
    이미지의 윗부분(원거리)이 좁게 보이는 현상을 펴줍니다.
    
    Args:
        image: 입력 ROI 이미지
        intensity (float): 보정 강도 (0.0 ~ 0.5). 값이 클수록 윗부분을 더 많이 늘림.
    """
    if image is None or image.size == 0:
        return image
        
    h, w = image.shape[:2]
    
    # 원본 이미지에서 '위가 좁은 사다리꼴' 영역을 정의
    dx = int(w * intensity)
    
    src_points = np.float32([
        [dx, 0],            # Top-Left (안쪽으로 들어감)
        [w - dx, 0],        # Top-Right (안쪽으로 들어감)
        [w, h],             # Bottom-Right (그대로)
        [0, h]              # Bottom-Left (그대로)
    ])
    
    dst_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected = cv2.warpPerspective(image, M, (w, h))
    
    return corrected

def preprocess_for_ocr(image):
    """
    OCR 성능 향상을 위한 전처리 (Resize & Sharpening)
    """
    if image is None or image.size == 0:
        return image
        
    rh, rw = image.shape[:2]
    if rh < 150: # 높이가 너무 작으면 확대
        scale = 150 / rh
        image = cv2.resize(image, (int(rw * scale), 150), interpolation=cv2.INTER_CUBIC)
    
    # 선명화 필터 적용 (부드러운 Unsharp Mask 방식)
    gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
    image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    return image
