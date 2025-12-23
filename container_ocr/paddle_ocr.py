import os
import re
import cv2
from paddleocr import PaddleOCR

# 맥북에선 보통 GPU 지원이 안 돼서 use_gpu=False가 안전함 (동수컴에선 True로 변경)
import torch
use_gpu = torch.cuda.is_available()

def extract_container_number(text):
    pattern = r'[A-Z]{4}[\s\d]{7,20}'
    match = re.search(pattern, text.upper())
    if match:
        clean = re.sub(r'[^A-Z0-9]', '', match.group())
        if len(clean) >= 10: return clean
    return "Not Found"

# PaddleOCR 인스턴스 (한국어/영어 지원)
# 동수컴 가면 자동으로 모델 다운로드됨
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)

img_dir = "images/"
imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_name in imgs:
    img_path = os.path.join(img_dir, img_name)
    
    # Paddle 실행
    result = ocr.ocr(img_path, cls=True)
    
    # Paddle은 결과가 3중 리스트라 텍스트만 쏙 빼야 함
    raw_texts = []
    if result[0]:
        for line in result[0]:
            raw_texts.append(line[1][0]) # 텍스트 부분만 추출
    
    raw_result = " ".join(raw_texts)
    clean_result = extract_container_number(raw_result)
    
    print(f"파일명: {img_name} -> 번호: {clean_result}")
    print(f"Raw Result : {raw_result}")
    print("-" * 20)