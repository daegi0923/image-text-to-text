from ultralytics import YOLO

def train():
    # 모델 로드 (yolov8n.pt, yolov8s.pt, yolov8m.pt 등 선택 가능)
    # 처음 학습시에는 pretrained 모델을 사용하는 것이 좋습니다.
    print("모델 로드 중...")
    model = YOLO("yolov8s.pt")  

    # 모델 학습
    print("학습 시작...")
    results = model.train(
        data="data.yaml",  # 데이터 설정 파일
        epochs=100,        # 학습 반복 횟수
        imgsz=640,         # 이미지 크기
        batch=16,          # 배치 크기
        device='auto',     # GPU/MPS/CPU 자동 선택
        patience=20,       # 성능 향상이 없으면 조기 종료
        save=True,         # 체크포인트 저장
        project="outputs", # 저장 경로
        name="yolo_container_ocr" # 실험 이름
    )
    
    print("학습 완료!")

if __name__ == '__main__':
    train()
 