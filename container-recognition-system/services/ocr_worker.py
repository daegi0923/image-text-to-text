from typing import Union, List, Dict
from pathlib import Path
import re
import time
import torch
import logging

# ISO 검증 모듈
from . import validator as iso6346

# 엔진별 임포트 (지연 로딩)
try:
    from paddleocr import PaddleOCR as PaddleEngine
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False


class ContainerOCR:
    def __init__(self, model_name: str = "paddle"):
        """
        OCR 엔진 초기화
        model_name: "paddle" 또는 "Qwen/..." (HuggingFace 경로)
        """
        self.logger = logging.getLogger(__name__)
        
        # 모델 이름에 따라 엔진 결정
        if "paddle" in model_name.lower():
            self.engine_type = "paddle"
            if not PADDLE_AVAILABLE:
                raise ImportError("PaddleOCR이 설치되지 않았습니다. pip install paddleocr")
            
            self.logger.info("🚀 PaddleOCR 초기화 중 (언어: en, GPU: 자동감지)...")
            # use_angle_cls=True: 뒤집힌 글자도 바로잡아서 읽음
            # use_gpu=True: GPU 있으면 씀 (없으면 자동 CPU)
            self.model = PaddleEngine(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available(), show_log=False)
            self.logger.info("✅ PaddleOCR 준비 완료!")
            
        else:
            self.engine_type = "qwen"
            if not QWEN_AVAILABLE:
                raise ImportError("Qwen 관련 라이브러리가 없습니다.")
                
            self.logger.info(f"🤖 Qwen3-VL 초기화 중: {model_name}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model.eval()
            self.logger.info(f"✅ Qwen3-VL 준비 완료 ({self.device})")

    def process_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, any]]:
        """
        이미지 리스트를 받아 일괄 처리 (엔진에 따라 분기)
        """
        if not image_paths: return []
        
        start_time = time.time()
        
        if self.engine_type == "paddle":
            results = self._process_batch_paddle(image_paths)
        else:
            results = self._process_batch_qwen(image_paths)
            
        elapsed = time.time() - start_time
        self.logger.info(f"⚡ Batch OCR 완료: {len(image_paths)}장 -> {elapsed:.2f}초")
        return results

    def _process_batch_paddle(self, image_paths) -> List[Dict]:
        """PaddleOCR 배치 처리"""
        results = []
        
        # PaddleOCR은 리스트로 던지면 내부적으로 루프를 돌지만 속도가 매우 빠름
        # (공식적으로 batch_size 파라미터가 없어서 한 장씩 호출해도 충분히 빠름)
        # 만약 진정한 배치를 원하면 PP-OCRv4 추론 모델을 직접 서빙해야 함
        
        for path in image_paths:
            img_path_str = str(path)
            try:
                # cls=True: 방향 보정
                ocr_result = self.model.ocr(img_path_str, cls=True)
                
                # 결과 파싱: 텍스트 덩어리들을 하나로 합침
                full_text = ""
                conf_sum = 0
                count = 0
                
                if ocr_result and ocr_result[0]:
                    # ocr_result[0] -> [ [ [[x,y],..], ("TEXT", 0.99) ], ... ]
                    texts = [line[1][0] for line in ocr_result[0]]
                    confs = [line[1][1] for line in ocr_result[0]]
                    full_text = " ".join(texts)
                    conf_sum = sum(confs)
                    count = len(confs)
                
                avg_conf = conf_sum / count if count > 0 else 0.0
                
                # 컨테이너 번호 추출 로직
                info = self._parse_container_number(full_text)
                info.update({
                    'image_path': img_path_str,
                    'raw_output': full_text,
                    'confidence': avg_conf
                })
                results.append(info)
                
            except Exception as e:
                self.logger.error(f"PaddleOCR Error ({path}): {e}")
                results.append({'found': False, 'image_path': img_path_str, 'error': str(e)})
                
        return results

    def _process_batch_qwen(self, image_paths) -> List[Dict]:
        """Qwen-VL 배치 처리 (기존 로직)"""
        # ... (아까 짠 배치 코드 그대로 유지) ...
        # 여기서는 지면 관계상 핵심만 남김, 실제로는 아까 작성한 코드 전체가 들어감
        # (네가 원하면 전체 다시 써줌)
        batch_messages = [self._build_prompt(str(p)) for p in image_paths]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=50)
            trimmed = [out[len(in_):] for in_, out in zip(inputs.input_ids, gen_ids)]
            responses = self.processor.batch_decode(trimmed, skip_special_tokens=True)
            
        results = []
        for i, text in enumerate(responses):
            info = self._parse_container_number(text)
            info['image_path'] = str(image_paths[i])
            results.append(info)
        return results

    def _build_prompt(self, image_path):
        return [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": "Extract container number (ISO 6346)"}]}]

    def _parse_container_number(self, text: str) -> Dict[str, any]:
        # 공백/특수문자 제거
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        # 패턴 매칭 (XXXX 123456 7)
        match = re.search(r'([A-Z]{3}[UJRZ])(\d{6})(\d)', cleaned)
        
        if match:
            full = match.group(0)
            valid, calc = iso6346.validate_container_number(full)
            return {
                'container_number': iso6346.format_container_number(full),
                'found': True,
                'check_digit_valid': valid
            }
        return {'container_number': None, 'found': False}

    def consolidate_results(self, results: List[Dict]) -> Dict:
        # 투표 로직 (기존 동일)
        if not results: return {'found': False}
        candidates = {}
        for r in results:
            if not r.get('found'): continue
            num = r['container_number']
            if num not in candidates: candidates[num] = {'count':0, 'valid': r.get('check_digit_valid')}
            candidates[num]['count'] += 1
            if r.get('check_digit_valid'): candidates[num]['valid'] = True
            
        if not candidates: return {'found': False}
        best = sorted(candidates.items(), key=lambda x: (x[1]['valid'], x[1]['count']), reverse=True)[0]
        return {'found': True, 'container_number': best[0], 'voting_meta': best[1]}
