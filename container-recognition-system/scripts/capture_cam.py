import cv2
import os
import time
from multiprocessing import Process

# 여기에 RTSP 주소 4개 채워넣어라
RTSP_CONFIG = [
    ("rtsp://admin:Hello@2023@10.100.3.111:554/0/onvif/profile2/media.smp", "cam_01"),
    ("rtsp://admin:Hello@2023@10.100.3.112:554/0/onvif/profile2/media.smp", "cam_02"),
    ("rtsp://admin:Hello@2023@10.100.3.113:554/0/onvif/profile2/media.smp", "cam_03"),
    ("rtsp://admin:Hello@2023@10.100.3.114:554/0/onvif/profile2/media.smp", "cam_04"),
]

def capture_cam(rtsp_url, cam_id):
    cap = cv2.VideoCapture(rtsp_url)
    
    # 스크립트 위치 기준 상위 data 폴더로 잡는다. 엉뚱한데 저장하지 말고.
    # 결과 경로: container-recognition-system/data/dataset/raw_captures/{cam_id}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "data", "dataset", "raw_captures", cam_id)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"[{cam_id}] 캡처 시작: {rtsp_url} -> {save_path}")
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_id}] 영상 끊김. 재연결 시도 중... (또는 종료)")
            break
        
        # 5프레임마다 하나씩 저장 (필요하면 조절해)
        if count % 5 == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{cam_id}_{count:06d}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), frame)
            
        count += 1
    cap.release()
    print(f"[{cam_id}] 종료")

if __name__ == "__main__":
    processes = []
    
    print(f"총 {len(RTSP_CONFIG)}개의 카메라 캡처 프로세스를 시작한다.")
    
    for url, cam_id in RTSP_CONFIG:
        p = Process(target=capture_cam, args=(url, cam_id))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()