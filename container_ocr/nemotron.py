import torch
import os
import re
from nemotron_ocr.inference.pipeline import NemotronOCR

# 장치 설정
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def extract_container_number(text):
    pattern = r'[A-Z]{4}[\s\d]{7,20}'
    match = re.search(pattern, text.upper())
    if match:
        clean = re.sub(r'[^A-Z0-9]', '', match.group())
        if len(clean) >= 10: return clean
    return "Not Found"

# 모델 로드 (빌드 완료된 상태여야 함)
ocr = NemotronOCR()

img_dir = "images/"
imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_name in imgs:
    img_path = os.path.join(img_dir, img_name)
    
    # Nemotron 실행
    predictions = ocr(img_path)
    
    # 결과 합치기
    raw_result = " ".join([p['text'] for p in predictions])
    clean_result = extract_container_number(raw_result)
    
    print(f"파일명: {img_name} -> 번호: {clean_result}")
    print(f"Raw Result : {raw_result}")
    print("-" * 20)