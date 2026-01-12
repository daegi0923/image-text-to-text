import cv2
import os
import time
from multiprocessing import Process

RTSP_CONFIG = [
    ("rtsp://admin:Hello@2023@10.100.3.111:554/0/onvif/profile2/media.smp", "cam_01"),
    ("rtsp://admin:Hello@2023@10.100.3.112:554/0/onvif/profile2/media.smp", "cam_02"),
    ("rtsp://admin:Hello@2023@10.100.3.113:554/0/onvif/profile2/media.smp", "cam_03"),
    ("rtsp://admin:Hello@2023@10.100.3.114:554/0/onvif/profile2/media.smp", "cam_04"),
]

def capture_cam(rtsp_url, cam_id):
    cap = cv2.VideoCapture(rtsp_url)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "data", "dataset", "raw_captures", cam_id)
    os.makedirs(save_path, exist_ok=True)
    
    # 움직임 감지를 위한 초기화
    ret, prev_frame = cap.read()
    if not ret: return
    
    # 연산 속도를 위해 흑백 + 가우시안 블러 적용
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    last_saved_time = 0
    save_interval = 0.5  # 최소 0.5초 간격으로 저장 (연사 방지)
    motion_threshold = 1000  # 픽셀 변화량 기준 (화면 크기에 따라 500~2000 조절)

    print(f"[{cam_id}] 스마트 캡처 시작...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 현재 프레임 전처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 차이 계산
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_count = cv2.countNonZero(thresh)

        curr_time = time.time()
        # 움직임이 있고, 저장 쿨타임이 지났을 때만 저장
        if motion_count > motion_threshold and (curr_time - last_saved_time) > save_interval:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # 파일명에 motion_count를 넣어보면 나중에 문계치 조절할 때 편함
            filename = f"{timestamp}_{cam_id}_{motion_count}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), frame)
            
            print(f"[{cam_id}] 차 들어옴! (Diff: {motion_count})")
            last_saved_time = curr_time
        
        prev_gray = gray  # 프레임 업데이트

    cap.release()

if __name__ == "__main__":
    # (메인 실행 로직은 동일)
    processes = [Process(target=capture_cam, args=(u, c)) for u, c in RTSP_CONFIG]
    for p in processes: p.start()
    for p in processes: p.join()