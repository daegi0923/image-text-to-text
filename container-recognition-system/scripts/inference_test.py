import cv2
import argparse
import random
import os
from ultralytics import YOLO

def get_color(cls_id):
    """클래스 ID별로 고정된 랜덤 색상을 반환 (눈에 잘 띄는 색 위주)"""
    random.seed(cls_id * 777)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def run_inference(model_path, source, conf_threshold=0.5, img_size=640):
    print(f"🔥 모델 로드 중: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    print(f"🎥 영상 소스 여는 중: {source}")
    
    # 입력이 숫자면 웹캠으로 간주
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 영상 시작 (FPS: {fps}) - 'q'를 눌러 종료, 'Space'로 일시정지")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("🎬 영상 종료")
                break
        else:
            # 일시정지 상태에서는 프레임만 계속 보여줌 (키 입력 대기)
            pass

        if not paused:
            # YOLO 추론 (Tracking 모드)
            results = model.track(frame, persist=True, conf=conf_threshold, verbose=False, imgsz=img_size)
            
            # 시각화
            if results:
                result = results[0]
                # 각 박스 순회
                if result.boxes:
                    for box in result.boxes:
                        # 좌표
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # 정보 추출
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id]
                        
                        # 트래킹 ID (있으면 표시)
                        track_id = int(box.id[0]) if box.id is not None else -1
                        
                        # 색상 및 라벨
                        color = get_color(cls_id)
                        label = f"{class_name} {conf:.2f}"
                        if track_id != -1:
                            label = f"ID:{track_id} {label}"

                        # 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 텍스트 배경 (가독성)
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 정보 표시
            cv2.putText(frame, f"Model: {os.path.basename(model_path)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 화면 출력
        cv2.imshow("Inference Test", frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 스페이스바
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"⏯ {status}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 모델 추론 테스트 스크립트")
    parser.add_argument("--model", type=str, required=True, help="학습된 .pt 모델 파일 경로")
    parser.add_argument("--source", type=str, required=True, help="테스트할 비디오 파일 경로 또는 웹캠 번호(0)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence Threshold (기본: 0.5)")
    
    args = parser.parse_args()
    
    run_inference(args.model, args.source, args.conf)
