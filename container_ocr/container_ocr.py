"""
컨테이너 일련번호 인식 시스템 (CUDA 최적화 버전)
Qwen3-VL을 사용하여 컨테이너 이미지에서 일련번호를 추출합니다.
GPU(CUDA)가 있으면 자동으로 활용하여 빠른 처리 속도를 제공합니다.
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import re
from typing import Union, List, Dict
from pathlib import Path
import iso6346


class ContainerOCR:
    """컨테이너 일련번호를 인식하는 클래스 (CUDA 자동 감지)"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        """
        Qwen3-VL 초기화 (GPU 자동 감지)
        
        Args:
            model_name: 사용할 Qwen3-VL 모델
        """
        print(f"Qwen3-VL 초기화 중: {model_name}")
        
        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA 감지됨: {gpu_name}")
            torch_dtype = torch.bfloat16  # GPU에서는 bfloat16 사용 (더 빠름)
        else:
            self.device = "cpu"
            print(f"⚠ CUDA를 사용할 수 없습니다. CPU 모드로 실행합니다.")
            torch_dtype = torch.float32
        
        print(f"사용 디바이스: {self.device}, dtype: {torch_dtype}")
        
        # 모델과 프로세서 로드
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Inference mode로 설정 (속도 향상)
        self.model.eval()
        
        print("Qwen3-VL 초기화 완료!")
    
    def extract_container_number(self, image_path: Union[str, Path]) -> Dict[str, any]:
        """
        이미지에서 컨테이너 일련번호를 추출
        
        Args:
            image_path: 컨테이너 이미지 경로
            
        Returns:
            추출된 컨테이너 정보 딕셔너리
        """
        # Path 객체를 문자열로 변환
        image_path_str = str(image_path)
        
        # 프롬프트 구성 - ISO 6346 규격 명시
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path_str},
                    {"type": "text", "text": """Find the ISO 6346 shipping container number in this image. Even if the text is split across multiple lines or has irregular spacing, extract it correctly.

The container number consists of:
- Owner code: 3 uppercase letters
- Category identifier: 1 uppercase letter (must be U, J, R, or Z)
- Serial number: 6 digits
- Check digit: 1 digit

Format example: MSKU 602345 2 (THIS IS AN EXAMPLE)

Important: 
- The 4th character MUST be U, J, R, or Z.
- Ignore line breaks and irregular spacing in the image.
- Return ONLY the container number in format: XXXX NNNNNN N
- If no container number is found, respond with: NONE
- Do not include any other text in your response"""}
                ]
            }
        ]
        
        # 입력 준비
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # 추론 - 짧은 출력으로 속도 향상
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=50  # 컨테이너 번호는 짧으므로 축소
            )
        
        # 입력 제거하고 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[DEBUG] 모델 출력: '{response}'")
        
        # 컨테이너 번호 파싱
        container_info = self._parse_container_number(response, [response])
        container_info['raw_output'] = response
        container_info['all_detected_text'] = [response]
        container_info['image_path'] = image_path_str
        
        return container_info
    
    def _parse_container_number(self, text: str, text_list: List[str] = None) -> Dict[str, str]:
        """
        모델 출력에서 ISO 6346 규격 컨테이너 번호를 파싱
        
        Args:
            text: 모델의 출력 텍스트
            text_list: 개별 인식된 텍스트 리스트
            
        Returns:
            파싱된 컨테이너 정보
        """
        # 공백 및 특수문자 제거 후 순수 글자/숫자만 추출
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # 11자리 (4문자 + 7숫자) 패턴 찾기
        match = re.search(r'([A-Z]{3}[UJRZ])(\d{6})(\d)', cleaned)
        
        if match:
            full_code = match.group(0)
            owner_code = match.group(1)
            serial_number = match.group(2)
            check_digit = match.group(3)
            
            # 모듈을 사용하여 검증
            is_valid, calculated_digit = iso6346.validate_container_number(full_code)
            
            return {
                'container_number': iso6346.format_container_number(full_code),
                'owner_code': owner_code,
                'serial_number': serial_number,
                'check_digit': check_digit,
                'check_digit_valid': is_valid,
                'calculated_check_digit': str(calculated_digit),
                'found': True
            }
        
        return {
            'container_number': None,
            'owner_code': None,
            'serial_number': None,
            'check_digit': None,
            'check_digit_valid': None,
            'calculated_check_digit': None,
            'found': False
        }
    
    def process_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, any]]:
        """
        여러 이미지를 배치로 처리
        
        Args:
            image_paths: 이미지 경로 리스트
            
        Returns:
            추출된 컨테이너 정보 리스트
        """
        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n처리 중 ({i}/{len(image_paths)}): {image_path}")
            try:
                result = self.extract_container_number(image_path)
                results.append(result)
                
                if result['found']:
                    valid_status = "✓" if result.get('check_digit_valid', False) else "✗"
                    print(f"✓ 발견: {result['container_number']} (체크 디지트: {valid_status})")
                    if not result.get('check_digit_valid', False):
                        print(f"  경고: 체크 디지트가 올바르지 않습니다 (계산값: {result.get('calculated_check_digit', 'N/A')})")
                else:
                    print(f"✗ 컨테이너 번호를 찾지 못했습니다")
                    print(f"  인식된 텍스트: {result['all_detected_text']}")
                    print(f"  결합 텍스트: {result['raw_output']}")
                    
            except Exception as e:
                print(f"✗ 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'image_path': str(image_path),
                    'found': False,
                    'error': str(e)
                })
        
        return results


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="컨테이너 이미지에서 일련번호를 인식합니다 (CUDA 자동 감지)"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="컨테이너 이미지 파일 경로"
    )
    
    args = parser.parse_args()
    
    # OCR 시스템 초기화
    ocr = ContainerOCR()
    
    # 컨테이너 번호 추출
    result = ocr.extract_container_number(args.image_path)
    
    # 결과 출력
    print("\n" + "="*60)
    print("컨테이너 번호 인식 결과")
    print("="*60)
    print(f"이미지: {result['image_path']}")
    
    if result['found']:
        valid_icon = "✓" if result.get('check_digit_valid', False) else "✗"
        print(f"\n✓ 컨테이너 번호 발견!")
        print(f"  - 전체 번호: {result['container_number']}")
        print(f"  - 소유자 코드: {result['owner_code']}")
        print(f"  - 일련번호: {result['serial_number']}")
        print(f"  - 체크 디지트: {result['check_digit']} ({valid_icon} {'유효' if result.get('check_digit_valid') else '무효'})")
        if not result.get('check_digit_valid', False):
            print(f"  - 올바른 체크 디지트: {result.get('calculated_check_digit', 'N/A')}")
    else:
        print(f"\n✗ 컨테이너 번호를 찾을 수 없습니다")
    
    print(f"\n원본 출력:\n{result['raw_output']}")
    print("="*60)


if __name__ == "__main__":
    main()