from ultralytics import YOLO
import os

def train():
    # 모델 로드
    print("모델 로드 중...")
    model = YOLO("yolo11n.pt")  

    # 데이터 설정 파일 경로 (프로젝트 루트 기준)
    data_config = "configs/data.yaml"
    if not os.path.exists(data_config):
        print(f"설정 파일을 찾을 수 없습니다: {data_config}")
        return
    print(data_config)
    # 모델 학습
    print("학습 시작...")
    results = model.train(
        data=data_config,  # 데이터 설정 파일
        epochs=100,        # 학습 반복 횟수
        imgsz=640,         # 이미지 크기
        batch=16,          # 배치 크기
        device='0',      # GPU/MPS/CPU 자동 선택 (필요시 변경)
        patience=20,       # 성능 향상이 없으면 조기 종료
        save=True,         # 체크포인트 저장
        project="outputs", # 저장 경로
        name="yolo_container_multilabel" # 실험 이름
    )
    
    print("학습 완료!")

if __name__ == '__main__':
    train()
