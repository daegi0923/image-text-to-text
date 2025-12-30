import cv2
import os

def capture_image(video_name):
    video_path = f'videos/{video_name}.mp4'
    save_dir = f'dataset/raw_images'
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_interval = 3 # 30fps 영상 기준 0.5초마다 한 장씩

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 일정 간격으로 저장
        if count % frame_interval == 0:
            img_name = f"{video_name}_frame_{count}.jpg"
            cv2.imwrite(os.path.join(save_dir, img_name), frame)
            print(f"저장됨: {img_name}")
        
        count += 1

    cap.release()
    print("이미지 추출 완료.")
    
if __name__ == '__main__':
    # 단일 비디오
    video_name = 'gate_side1'
    capture_image(video_name)
    # video list 이용
    video_names = [
        'gate_top1',
        'gate_top2',
        'gate_top3',
        'gate_top4',
        'gate_top5',
        'gate_top6',
        'gate_side1',
        'gate_side2',
        'gate_side3',
        'gate_side4',
        'gate_side5',
        'gate_side6'
    ]
    for video_name in video_names:
        capture_image(video_name)
