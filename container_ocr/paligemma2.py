import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import os
import re

def extract_container_number(text):
    # 1. 알파벳 4글자 찾고, 그 뒤로 숫자와 공백이 섞인 덩어리를 넉넉하게 긁어옴
    # [A-Z]{4} : 알파벳 4개
    # [\s\d]{7,20} : 공백이나 숫자가 7~20개 섞인 구간
    pattern = r'[A-Z]{4}[\s\d]{7,20}'
    
    match = re.search(pattern, text.upper())
    if match:
        raw_match = match.group()
        # 2. 긁어온 덩어리에서 숫자와 알파벳만 남기고 공백/특수문자 싹 제거
        clean = re.sub(r'[^A-Z0-9]', '', raw_match)
        
        # 3. 컨테이너 규격(보통 11자리)에 근접하면 반환
        if len(clean) >= 10:
            return clean
            
    return "Not Found"
# 1. 모델이랑 프로세서 로드
model_id = "google/paligemma2-3b-ft-docci-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
processor = PaliGemmaProcessor.from_pretrained(model_id)
# 2. 이미지 폴더 경로
img_dir = "images/"
imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 3. 루프 돌면서 확인
prompt = "<image>ocr the container number in 'ABCD 123456 7' format. do not output anything else." # PaliGemma한테 시킬 명령
for img_name in imgs:
    img = Image.open(os.path.join(img_dir, img_name)).convert("RGB") #
    inputs = processor(text=prompt, images=img, return_tensors="pt").to("mps") #
    
    output = model.generate(**inputs, max_new_tokens=100)
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
    # 결과 출력 부분 수정
    clean_result = extract_container_number(result)
    print(f"파일명: {img_name} -> 번호: {clean_result}")
    print(f"Raw Result : {result}")
    print("------------------")
