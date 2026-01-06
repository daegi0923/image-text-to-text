from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import re
from typing import Union, List, Dict
from pathlib import Path
from . import validator as iso6346


class ContainerOCR:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        print(f"Qwen3-VL 초기화 중: {model_name}")
        
        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.bfloat16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"Qwen3-VL 배취 추론 모드 준비 완료 ({self.device})")
    
    def _build_prompt(self, image_path: str) -> List[Dict]:
        """단일 이미지에 대한 프롬프트 구성"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Extract ONLY the container number in ISO 6346 format (XXXX NNNNNN N). If not clear, respond with NONE."}
                ]
            }
        ]

    def process_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, any]]:
        """
        이미지 리스트를 받아서 '진짜 배치(Batch) 추론' 수행
        """
        if not image_paths:
            return []

        # 1. 모든 이미지에 대한 메시지 리스트 생성
        batch_messages = [self._build_prompt(str(p)) for p in image_paths]
        
        # 2. 전처리 (Template 적용 및 Vision Info 처리)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        # 3. 텐서화 (Batch, Channel, Height, Width) - 자동 패딩 포함
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # 4. 일괄 추론 (Forward Pass 한 번에 N개 처리)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # 입력 토큰 제외하고 결과 토큰만 추출
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
            ]
            
            responses = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )

        # 5. 결과 파싱 및 리스트 반환
        results = []
        for i, response in enumerate(responses):
            res_text = response.strip()
            container_info = self._parse_container_number(res_text)
            container_info.update({
                'image_path': str(image_paths[i]),
                'raw_output': res_text,
                'found': container_info['found']
            })
            results.append(container_info)
            
        return results

    def _parse_container_number(self, text: str) -> Dict[str, any]:
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        match = re.search(r'([A-Z]{3}[UJRZ])(\d{6})(\d)', cleaned)
        
        if match:
            full_code = match.group(0)
            is_valid, calculated_digit = iso6346.validate_container_number(full_code)
            return {
                'container_number': iso6346.format_container_number(full_code),
                'owner_code': match.group(1),
                'serial_number': match.group(2),
                'check_digit': match.group(3),
                'check_digit_valid': is_valid,
                'calculated_check_digit': str(calculated_digit),
                'found': True
            }
        
        return {'container_number': None, 'found': False}

    def consolidate_results(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """투표 로직 (기존 유지)"""
        if not results: return {'found': False}
        
        candidates = {}
        for res in results:
            if not res.get('found') or not res.get('container_number'): continue
            num = res['container_number']
            if num not in candidates:
                candidates[num] = {'count': 0, 'valid': res.get('check_digit_valid', False), 'raw': res}
            candidates[num]['count'] += 1
            if res.get('check_digit_valid'): candidates[num]['valid'] = True

        if not candidates: return {'found': False}
        
        sorted_cand = sorted(candidates.items(), key=lambda x: (x[1]['valid'], x[1]['count']), reverse=True)
        winner_num, winner_data = sorted_cand[0]
        
        final_res = winner_data['raw'].copy()
        final_res['voting_meta'] = {'total_votes': len(results), 'winner_count': winner_data['count']}
        return final_res