import cv2
import os

video_path = 'videos/gate_side1.mp4'
save_dir = 'dataset/raw_images'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
frame_interval = 3 # 30fps 영상 기준 0.5초마다 한 장씩

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 일정 간격으로 저장
    if count % frame_interval == 0:
        img_name = f"frame_{count}.jpg"
        cv2.imwrite(os.path.join(save_dir, img_name), frame)
        print(f"저장됨: {img_name}")
    
    count += 1

cap.release()
print("이미지 추출 완료.")