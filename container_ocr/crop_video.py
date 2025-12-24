import cv2

# 위에서 뽑은 좌표 입력 (예시: 100, 200, 400, 150)
x, y, w, h = (338, 174, 214, 282) 

cap = cv2.VideoCapture('videos/gate_side1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. 고정 ROI 영역 잘라내기
    crop = frame[y:y+h, x:x+w]

    # 2. OCR 성능 향상을 위한 전처리 (흑백 전환)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 화면 확인용
    cv2.imshow('Cropped ROI', gray_crop)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()