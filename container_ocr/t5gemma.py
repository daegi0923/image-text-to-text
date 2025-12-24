import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from PIL import Image
import os
import re

class T5GemmaContainerOCR:
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        
        print(f"[{model_id}] 로딩 중... (Device: {self.device})")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            # MPS 안정성 위해 float32 사용
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32, 
                device_map=self.device
            )
            print("모델 로딩 완료.")
            
            # 이미지 토큰 ID 확인 및 설정
            self.img_token_id = self._get_image_token_id()
            
        except Exception as e:
            print(f"초기화 실패: {e}")
            exit()

    def _get_image_token_id(self):
        try:
            token_str = "<image_soft_token>"
            tid = self.processor.tokenizer.convert_tokens_to_ids(token_str)
            if tid == self.processor.tokenizer.unk_token_id:
                tid = 256001 # fallback
            print(f"이미지 토큰 ID: {tid}")
            return tid
        except:
            return 256001

    def _calculate_check_digit(self, owner_code, serial_number):
        # ISO 6346 체크 디지트 계산 로직
        char_values = {
            'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17,
            'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25,
            'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32,
            'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
        }
        
        container_id = owner_code + serial_number
        total = 0
        
        # serial_number가 7자리인 경우 앞 6자리만 계산에 사용 (마지막은 체크 디지트일 수 있음)
        # 하지만 함수 인자로 넘어올 때는 이미 분리된 serial(6자리)을 기대함
        # 만약 7자리가 넘어오면 앞 6자리만 쓴다.
        if len(serial_number) > 6:
            serial_number = serial_number[:6]

        for i, char in enumerate(container_id):
            # 숫자가 아니면 패스하거나 예외 처리
            if not char.isalnum(): continue
            
            val = char_values.get(char, int(char)) if char.isalpha() else int(char)
            # 가중치: 2^i
            total += val * (2 ** i)
        
        remainder = total % 11
        return 0 if remainder == 10 else remainder

    def parse_result(self, text):
        # 정제: 알파벳, 숫자 외 제거 -> 공백으로 치환
        cleaned = re.sub(r'[^A-Z0-9]', ' ', text.upper())
        
        # 패턴 1: 4글자 + 6자리 일련번호 + 1자리 체크디지트 (표준)
        # 패턴 2: 4글자 + 7자리 (마지막 자리가 체크디지트인 경우)
        patterns = [
            r'([A-Z]{4})\s*(\d{6})\s*(\d)',
            r'([A-Z]{4})\s*(\d{7})' 
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                owner = match.group(1)
                full_digits = match.group(2)
                
                if len(full_digits) == 6:
                    serial = full_digits
                    # 체크디지트가 그룹 3에 있다면
                    if match.lastindex >= 3:
                        check = match.group(3)
                    else:
                        continue # 체크디지트 없으면 패스
                else: # 7자리인 경우
                    serial = full_digits[:6]
                    check = full_digits[6]
                
                # 검증
                calc_check = self._calculate_check_digit(owner, serial)
                is_valid = (int(check) == calc_check)
                
                return {
                    'found': True,
                    'full_number': f"{owner} {serial} {check}",
                    'owner': owner,
                    'serial': serial,
                    'check': check,
                    'valid': is_valid,
                    'calc_check': calc_check
                }
        
        return {'found': False}

    def process_image(self, image_path):
        try:
            pil_img = Image.open(image_path).convert("RGB")
            
            # 상세 프롬프트 (설명충 방지, 단순화)
            prompt_text = """Find the ISO 6346 shipping container number in this image. Even if the text is split across multiple lines or has irregular spacing, extract it correctly.

The container number consists of:
- Owner code: exactly 4 uppercase letters
- Serial number: exactly 6 digits
- Check digit: exactly 1 digit

Examples: ABCU 123456 7 (THIS IS AN EXAMPLE)

Important: 
- Ignore line breaks and irregular spacing in the image
- The serial number is always 6 digits (not more, not less)
- Return ONLY the container number in format: XXXX NNNNNN N
- If no container number is found, respond with: NONE
- Do not include any other text in your response"""
            
            # 수동 토큰 조립 (256개 이미지 토큰 + 텍스트)
            text_tokens = self.processor.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            
            img_tokens = torch.tensor([[self.img_token_id] * 256], device=self.device)
            input_ids = torch.cat([img_tokens, text_tokens], dim=1)
            
            # 이미지 전처리 (float32)
            pixel_values = self.processor.image_processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
            pixel_values = pixel_values.to(dtype=torch.float32)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=50,
                    min_new_tokens=2,
                    do_sample=False
                )
            
            raw_text = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            result = self.parse_result(raw_text)
            
            print(f"[{os.path.basename(image_path)}]")
            print(f"  Raw Output: '{raw_text}'")
            
            if result['found']:
                valid_mark = "✅" if result['valid'] else "❌"
                print(f"  -> Found: {result['full_number']} (Check Digit: {valid_mark})")
                if not result['valid']:
                    print(f"     (Expected: {result['calc_check']}, Got: {result['check']})")
            else:
                print("  -> Not Found")
            print("-" * 40)
            
        except Exception as e:
            print(f"[{os.path.basename(image_path)}] Error: {e}")

if __name__ == "__main__":
    ocr_system = T5GemmaContainerOCR()
    
    img_dir = "images/"
    if os.path.exists(img_dir):
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not files:
            print("이미지가 없습니다.")
        else:
            print(f"총 {len(files)}개 이미지 분석 시작")
            for f in files:
                ocr_system.process_image(os.path.join(img_dir, f))
    else:
        print(f"폴더 없음: {img_dir}")
