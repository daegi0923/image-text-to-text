from ultralytics import YOLO
import os

def train():
    # 모델 로드
    print("모델 로드 중...")
    model = YOLO("yolo11n-obb.pt")  

    # 데이터 설정 파일 경로
    data_config = "configs/data.yaml"
    if not os.path.exists(data_config):
        print(f"설정 파일을 찾을 수 없습니다: {data_config}")
        return
    
    # 모델 학습
    print("OBB 학습 시작...")
    results = model.train(
        data=data_config,
        epochs=100,
        imgsz=640,
        batch=16,
        device='0',
        patience=20,
        save=True,
        project="outputs",
        name="yolo_container_obb" # 이름도 OBB로 변경
    )
    
    print("학습 완료!")

if __name__ == '__main__':
    train()
