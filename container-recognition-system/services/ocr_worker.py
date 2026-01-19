from typing import Union, List, Dict
from pathlib import Path
import re
import time
import torch
import logging
import os
import sys

# [Windows DLL 경로 강제 주입]
# PaddleOCR이 pip로 설치된 nvidia-cudnn-cu12의 DLL을 못 찾을 때 해결책
if os.name == 'nt':
    import site
    try:
        # site-packages 경로 찾기
        site_packages = site.getsitepackages()
        for sp in site_packages:
            cudnn_bin = os.path.join(sp, 'nvidia', 'cudnn', 'bin')
            cublas_bin = os.path.join(sp, 'nvidia', 'cublas', 'bin')
            
            if os.path.exists(cudnn_bin):
                os.add_dll_directory(cudnn_bin)
                # PATH에도 추가 (구형 호환)
                os.environ['PATH'] = cudnn_bin + os.pathsep + os.environ['PATH']
                print(f"DEBUG: Added DLL dir -> {cudnn_bin}")
                
            if os.path.exists(cublas_bin):
                os.add_dll_directory(cublas_bin)
                os.environ['PATH'] = cublas_bin + os.pathsep + os.environ['PATH']
                print(f"DEBUG: Added DLL dir -> {cublas_bin}")
    except Exception as e:
        print(f"Warning: Failed to add DLL directory: {e}")

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
            # use_gpu 옵션 제거 (자동 감지 위임)
            self.model = PaddleEngine(use_angle_cls=True, lang='en')
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
        """PaddleOCR 배치 처리 (세로 텍스트 대응 회전 로직 추가)"""
        results = []
        import cv2
        import numpy as np

        for path in image_paths:
            img_path_str = str(path)
            
            # 1. 원본 이미지 로드
            original_img = cv2.imread(img_path_str)
            if original_img is None:
                results.append({'found': False, 'image_path': img_path_str, 'error': 'Image load failed'})
                continue

            # 시도할 이미지 목록 (원본 -> 시계90 -> 반시계90)
            # (이미지 객체, 회전각도설명)
            attempts = [
                (original_img, "Original"),
                (cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE), "Rot90_CW"),
                (cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE), "Rot90_CCW")
            ]
            
            best_result = {'found': False, 'container_number': None, 'confidence': 0.0}
            
            for img, angle_desc in attempts:
                try:
                    # PaddleOCR에 numpy array 직접 전달 가능
                    ocr_result = self.model.ocr(img)
                    
                    full_text = ""
                    conf_sum = 0
                    count = 0
                    valid_lines = []
                    
                    if ocr_result:
                        if isinstance(ocr_result, list):
                            for item in ocr_result:
                                if not item: continue
                                if isinstance(item, list):
                                    for line in item:
                                        if isinstance(line, dict) and 'rec_texts' in line:
                                            valid_lines.extend(zip(line['rec_texts'], line.get('rec_scores', [0]*len(line['rec_texts']))))
                                        elif isinstance(line, list) and len(line) >= 2 and isinstance(line[1], tuple):
                                            valid_lines.append(line[1])
                                elif isinstance(item, dict) and 'rec_texts' in item:
                                    valid_lines.extend(zip(item['rec_texts'], item.get('rec_scores', [0]*len(item['rec_texts']))))

                    if valid_lines:
                        texts = [txt for txt, conf in valid_lines]
                        confs = [float(conf) for txt, conf in valid_lines]
                        full_text = " ".join(texts)
                        conf_sum = sum(confs)
                        count = len(confs)
                    
                    avg_conf = conf_sum / count if count > 0 else 0.0
                    
                    # 파싱 시도
                    info = self._parse_container_number(full_text)
                    
                    # 찾았으면 즉시 채택 (단, 체크 디지트 유효한 걸 우선)
                    if info['found']:
                        info.update({
                            'image_path': img_path_str,
                            'raw_output': full_text,
                            'confidence': avg_conf,
                            'rotation_used': angle_desc
                        })
                        
                        # 체크 디지트 맞으면 더 볼 것도 없이 확정
                        if info.get('check_digit_valid'):
                            best_result = info
                            break # 루프 탈출
                        
                        # 체크 디지트 틀려도 일단 후보로 등록 (다른 각도에서 더 좋은 게 나올 수 있으니 break 안 함)
                        if avg_conf > best_result.get('confidence', 0):
                            best_result = info
                    
                except Exception as e:
                    self.logger.warning(f"OCR Fail ({angle_desc}): {e}")
            
            # 3번 다 해봤는데도 없으면 실패 처리, 하나라도 건졌으면 성공
            if best_result.get('found'):
                if best_result.get('rotation_used') != "Original":
                    self.logger.info(f"🔄 회전 인식 성공 ({img_path_str}): {best_result['rotation_used']} -> {best_result['container_number']}")
                results.append(best_result)
            else:
                results.append({'found': False, 'image_path': img_path_str, 'raw_output': '', 'confidence': 0})
                
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