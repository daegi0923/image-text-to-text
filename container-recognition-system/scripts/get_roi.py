import cv2
import os

# 영상 경로 설정 (프로젝트 루트 실행 기준)
video_path = 'data/raw_videos/gate_side1.mp4'

if not os.path.exists(video_path):
    print(f"파일이 없습니다: {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

print("시작함. 영상 보다가 차 지나갈 때 'Space' 눌러서 멈추셈.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("영상 끝남.")
        break

    cv2.imshow('Video Player', frame)

    # 30ms 대기 (숫자 낮추면 빨라짐)
    key = cv2.waitKey(30) & 0xFF

    # 1. 스페이스바(32) 누르면 멈추고 ROI 선택 모드 진입
    if key == ord(' '):
        print("일시정지! 이제 마우스로 번호판 영역 긁으셈.")
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        
        if roi != (0, 0, 0, 0): # 제대로 선택했으면
            print(f"\n✅ 좌표 찾았다: {roi}")
            print(f"x, y, w, h = {roi}")
            break
    
    # 2. 'q' 누르면 그냥 종료
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
