import cv2
import os
import datetime
import shutil
from pathlib import Path
from container_ocr import ContainerOCR

def save_frame_image(video_name, frame, frame_count, suffix=""):
    save_dir = f"outputs/images/{os.path.splitext(video_name)[0]}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/frame_{frame_count:06d}{suffix}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def save_to_txt(video_name, data):
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.splitext(video_name)[0]
    filename = f"outputs/result_{base_name}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"영상 파일: {video_name}\n")
        f.write(f"분석 일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 110 + "\n")
        f.write(f"{ '시간':<10} {'컨테이너번호':<15} {'소유자':<6} {'일련번호':<8} {'체크':<4} {'유효':<4} {'이미지 경로':<40}\n")
        f.write("=" * 110 + "\n")
        if not data:
            f.write("결과 없음.\n")
        else:
            for item in data:
                valid_str = "O" if item['valid'] else "X"
                f.write(f"{item['time']:<10} {item['number']:<15} {item['owner']:<6} {item['serial']:<8} {item['check']:<4} {valid_str:<4} {item['image_path']:<40}\n")
    print(f"\n[저장 완료] {filename}")

def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"파일 없음: {video_path}")
        return

    # OCR 초기화 (container_ocr.py의 설정을 따름)
    print("OCR 모델 초기화 중...")
    try:
        ocr = ContainerOCR() 
    except Exception as e:
        print(f"OCR 초기화 실패: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0
    captured_data = []
    last_number = None
    
    # 임시 이미지 저장 경로 (OCR 모델이 파일 경로를 요구함)
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    temp_frame_path = temp_dir / "current_frame.jpg"

    print(f"[{os.path.basename(video_path)}] 분석 시작...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 15프레임마다 분석 (약 0.5초 단위)
        if frame_count % 15 == 0:
            # 임시 파일로 저장
            cv2.imwrite(str(temp_frame_path), frame)
            
            try:
                # OCR 수행
                # container_ocr.py의 extract_container_number 사용
                result = ocr.extract_container_number(temp_frame_path)
                
                if result['found']:
                    c_num = result['container_number']
                    print(f"[Frame {frame_count}] 발견: {c_num}")
                    
                    # 같은 번호가 연속으로 나오면 저장하지 않음 (단, 번호가 바뀌면 다시 저장)
                    # 여기서는 간단하게 바로 이전 인식된 번호와만 비교
                    if c_num != last_number:
                        current_time = str(datetime.timedelta(seconds=int(frame_count / fps)))
                        
                        # 결과 이미지 저장 (파일명에 번호 포함)
                        safe_num = c_num.replace(" ", "")
                        img_path = save_frame_image(os.path.basename(video_path), frame, frame_count, f"_{safe_num}")
                        
                        captured_data.append({
                            'time': current_time, 
                            'number': c_num, 
                            'owner': result['owner_code'],
                            'serial': result['serial_number'],
                            'check': result['check_digit'],
                            'valid': result.get('check_digit_valid', False),
                            'image_path': img_path
                        })
                        last_number = c_num
                else:
                    # 진행상황 표시 (너무 많이 출력되면 주석 처리)
                    # print(f"[Frame {frame_count}] .")
                    pass

            except Exception as e:
                print(f"[Frame {frame_count}] Error: {e}")

        frame_count += 1

    cap.release()
    
    # 임시 디렉토리 정리
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
    save_to_txt(os.path.basename(video_path), captured_data)

if __name__ == "__main__":
    target_video = "videos/gate_side1.mp4"
    process_video(target_video)