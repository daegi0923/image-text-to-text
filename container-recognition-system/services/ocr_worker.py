from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import re
from typing import Union, List, Dict
from pathlib import Path
# import iso6346 -> 변경됨
from . import validator as iso6346


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
                    {"type": "text", "text": """You are a precise OCR system for shipping container numbers.

                    **CRITICAL RULES:**
                    1. ONLY extract text that is ACTUALLY VISIBLE in this specific image
                    2. DO NOT make up, guess, or generate random container numbers
                    3. DO NOT use example numbers or placeholders
                    4. If you cannot clearly see a complete container number, respond with: NONE

                    **Container Number Format (ISO 6346):**
                    - Owner code: exactly 4 UPPERCASE letters
                    - Serial number: exactly 6 digits
                    - Check digit: exactly 1 digit
                    - Example format: ABCD 123456 7

                    **Instructions:**
                    - Look carefully at the image for text painted on the container
                    - The text may be split across multiple lines or have irregular spacing
                    - Extract ONLY what you can actually see in the image
                    - Combine the parts to form: XXXX NNNNNN N
                    - If the container number is incomplete, unclear, or not visible: respond with NONE
                    - Your response must be ONLY the container number or NONE, nothing else

                    **What you see in THIS image:**
                """}
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
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                return_dict_in_generate=True,  # 딕셔너리 형태로 결과 받기
                output_scores=True,           # 각 토큰별 점수(로짓) 포함하기
            )

            generated_ids = outputs.sequences  # 생성된 토큰 ID들
            scores = outputs.scores            # 각 단계의 로짓(Logits) 값들
                # 1. 생성된 토큰들만 골라내기 (입력 토큰 제외)
        gen_sequences = generated_ids[:, inputs.input_ids.shape[-1]:]

        # 2. 각 토큰별 확률 계산 및 Top-5 후보 분석
        probs = []
        logits = []
        print(f"\n[DEBUG] 상세 후보 분석 (Top 5):")
        
        for i in range(len(scores)):
            # 현재 스텝의 로짓 가져오기
            current_logits = scores[i][0]
            token_probs = torch.nn.functional.softmax(current_logits, dim=-1)
            
            # Top 5 후보 추출
            top_k_probs, top_k_ids = torch.topk(token_probs, k=5)
            
            step_candidates = []
            for j in range(5):
                cand_id = top_k_ids[j].item()
                cand_prob = top_k_probs[j].item()
                cand_token = self.processor.decode([cand_id])
                step_candidates.append(f"'{cand_token}'({cand_prob:.1%})")
            
            # 실제 선택된 토큰 정보 저장
            token_id = gen_sequences[0, i]
            prob = token_probs[token_id].item()
            logit = current_logits[token_id].item()
            chosen_token = self.processor.decode([token_id])
            
            probs.append(prob)
            logits.append(logit)
            
            print(f"  Step {i:02d} [{chosen_token}]: {', '.join(step_candidates)}")

        # 3. 토큰이랑 확률 매칭해서 확인
        decoded_tokens = [self.processor.decode([tid]) for tid in gen_sequences[0]]
        
        # 디버그 정보 포맷팅 (확률과 로짓 함께 표시)
        # 예: 'A'(99.9%, L:15.2)
        token_debug = [f"'{token}'({prob:.1%}, L:{logit:.1f})" for token, prob, logit in zip(decoded_tokens, probs, logits)]
        print(f"[DEBUG] 최종 선택 분석: {token_debug}")
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

    def consolidate_results(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """
        여러 카메라/프레임의 OCR 결과를 종합하여 투표(Voting)로 최종 결과 도출
        
        우선순위:
        1. ISO 6346 검증 통과 여부 (최우선)
        2. 다수결 (빈도수)
        3. 평균 신뢰도 (Confidence)
        """
        print(f"\n[Voting] 총 {len(results)}개의 결과로 투표를 진행합니다.")
        
        candidates = {}
        
        for res in results:
            if not res.get('found') or not res.get('container_number'):
                continue
                
            num = res['container_number']
            is_valid = res.get('check_digit_valid', False)
            
            # 신뢰도 점수 계산 (확률 정보가 없으면 기본값 0)
            # 여기서는 간단히 전체 텍스트 확률 평균을 사용하거나, 로짓 등을 활용할 수 있음
            # 현재 구조상 상세 확률은 로깅만 되고 있으므로, 우선순위 로직으로 처리
            
            if num not in candidates:
                candidates[num] = {
                    'count': 0,
                    'valid': is_valid,
                    'raw_results': []
                }
            
            candidates[num]['count'] += 1
            candidates[num]['raw_results'].append(res)
            
            # 하나라도 검증 통과했으면 해당 번호는 유효한 것으로 간주
            if is_valid:
                candidates[num]['valid'] = True

        if not candidates:
            print("[Voting] 유효한 컨테이너 번호 후보가 없습니다.")
            return {'found': False, 'reason': 'No candidates found'}

        # 후보 리스트 정렬
        # 1. 유효성(True > False)
        # 2. 빈도수(내림차순)
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: (x[1]['valid'], x[1]['count']),
            reverse=True
        )
        
        winner_number, winner_data = sorted_candidates[0]
        
        print(f"[Voting] 우승: {winner_number} (유효: {winner_data['valid']}, 득표: {winner_data['count']}/{len(results)})")
        
        # 2등이 있다면 로그 출력 (디버깅용)
        if len(sorted_candidates) > 1:
            runner_up_num, runner_up_data = sorted_candidates[1]
            print(f"  └ 2등: {runner_up_num} (유효: {runner_up_data['valid']}, 득표: {runner_up_data['count']})")

        # 최종 결과 반환 (대표 결과 하나를 리턴하되 메타데이터 추가)
        final_result = winner_data['raw_results'][0].copy()
        final_result['voting_meta'] = {
            'total_votes': len(results),
            'winner_count': winner_data['count'],
            'is_unanimous': winner_data['count'] == len(results),
            'candidates_count': len(candidates)
        }
        
        return final_result
