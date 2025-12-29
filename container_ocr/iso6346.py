import re

def get_iso6346_value(char):
    """문자를 ISO 6346 규격 숫자로 변환 (11의 배수 제외)"""
    # Mapping String: 0-9, A-Z (11의 배수인 11, 22, 33 위치는 ?로 처리)
    mapping = "0123456789A?BCDEFGHIJK?LMNOPQRSTU?VWXYZ"
    try:
        return mapping.index(char.upper())
    except ValueError:
        return 0

def validate_container_number(container_num):
    """
    ISO 6346 규격에 맞는지 검증 (체크 디지트 포함)
    
    Returns:
        (is_valid, calculated_check_digit)
    """
    # 공백 및 특수문자 제거
    cleaned = re.sub(r'[^A-Z0-9]', '', container_num.upper())
    
    # 기본 형식 체크: 알파벳 4자 + 숫자 7자 (총 11자)
    # 4번째 자리는 U(Freight), J(Equipment), R(Reefer), Z(Chassis) 중 하나여야 함
    if not re.match(r'^[A-Z]{3}[UJRZ][0-9]{7}$', cleaned):
        return False, None
    
    owner_serial = cleaned[:10]
    provided_check_digit = int(cleaned[10])
    
    # 가중치 합산 계산
    total_sum = 0
    for i, char in enumerate(owner_serial):
        val = get_iso6346_value(char)
        total_sum += val * (2 ** i)
    
    # ISO 6346 체크 디지트 공식
    # 1. Sum / 11 후 소수점 버림
    # 2. 결과에 11 곱함
    # 3. Sum에서 위 결과를 뺌
    calculated_check_digit = total_sum - (int(total_sum / 11) * 11)
    
    # 만약 나머지가 10이면 0으로 처리 (ISO 규격)
    if calculated_check_digit == 10:
        calculated_check_digit = 0
        
    is_valid = (provided_check_digit == calculated_check_digit)
    
    return is_valid, calculated_check_digit

def format_container_number(container_num):
    """인식된 번호를 XXXU 123456 7 형식으로 포맷팅"""
    cleaned = re.sub(r'[^A-Z0-9]', '', container_num.upper())
    if len(cleaned) == 11:
        return f"{cleaned[:4]} {cleaned[4:10]} {cleaned[10]}"
    return container_num
