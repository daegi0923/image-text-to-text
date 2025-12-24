import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from PIL import Image
import numpy as np

model_id = "google/t5gemma-2-270m-270m"
print("Loading model...")
processor = AutoProcessor.from_pretrained(model_id)
# mps 에러 회피 위해 cpu로 테스트 (가장 안전)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, 
    device_map=device, 
    torch_dtype=torch.float16 if device != "cpu" else torch.float32
)

# 테스트용 더미 이미지
img = Image.fromarray(np.uint8(np.random.rand(448, 448, 3) * 255))

print(f"Token ID for <image_soft_token>: {processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')}")

# Case 1: Processor 자동 (add_special_tokens=True/False 조절)
try:
    print("\n--- Test 1: Processor Automatic ---")
    prompt = "<image_soft_token>Describe this image."
    # add_special_tokens=False를 해야 토크나이저가 <image_soft_token>을 안 자를 수도 있음
    inputs = processor(text=prompt, images=img, return_tensors="pt", add_special_tokens=False).to(device)
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    # 이미지 토큰이 몇 개나 들어갔는지 확인
    img_token_count = (inputs.input_ids == 255999).sum().item()
    print(f"Image tokens count in inputs: {img_token_count}")
    
    model.generate(**inputs, max_new_tokens=10)
    print("Success!")
except Exception as e:
    print(f"Fail: {e}")

# Case 2: 수동 256개 주입
try:
    print("\n--- Test 2: Manual 256 tokens ---")
    token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    # 256개 이미지 토큰 + 텍스트 토큰
    img_tokens = torch.tensor([[token_id] * 256], device=device)
    text_tokens = processor.tokenizer("Describe", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([img_tokens, text_tokens], dim=1)
    
    pixel_values = processor.image_processor(img, return_tensors="pt").pixel_values.to(device)
    
    print(f"Input IDs shape: {input_ids.shape}")
    img_token_count = (input_ids == token_id).sum().item()
    print(f"Image tokens count in inputs: {img_token_count}")

    model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=10)
    print("Success!")
except Exception as e:
    print(f"Fail: {e}")

