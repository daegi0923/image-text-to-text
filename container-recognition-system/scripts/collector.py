import cv2
import os
import csv
import time
from datetime import datetime
from ultralytics import YOLO

def run_collector(model_path, source, output_dir="data/collected_samples", conf_threshold=0.5):
    # 1. 디렉토리 및 CSV 초기화
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "labels.csv")
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label', 'conf', 'collected_at', 'track_id'])

    print(f"🚀 상위 3장 수집기 가동! 저장 경로: {output_dir}")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    
    # 트랙별 버퍼: {tid: [{'conf': 0.9, 'img': frame}, ...]} - 최대 3개 유지
    best_shots_buffer = {} 
    finalized_ids = set()

    # 중앙 영역에서만 수집 (정확도 확보)
    ROI_X_MIN, ROI_X_MAX = 0.15, 0.85

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        fh, fw = frame.shape[:2]
        results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)

        current_time = time.time()

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box, track_id in zip(boxes, boxes.id):
                tid = int(track_id)
                if tid in finalized_ids: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cx = (x1 + x2) / 2 / fw

                if ROI_X_MIN < cx < ROI_X_MAX:
                    if tid not in best_shots_buffer:
                        best_shots_buffer[tid] = {'shots': [], 'last_seen': current_time}
                    
                    buffer = best_shots_buffer[tid]
                    buffer['last_seen'] = current_time
                    
                    pad = 15
                    crop = frame[max(0, y1-pad):min(fh, y2+pad), max(0, x1-pad):min(fw, x2+pad)].copy()
                    
                    # 상위 3개 관리 로직
                    shots = buffer['shots']
                    if len(shots) < 3:
                        shots.append({'conf': conf, 'img': crop})
                        shots.sort(key=lambda x: x['conf'], reverse=True)
                    else:
                        # 현재 3개 중 가장 낮은 점수보다 높으면 교체
                        if conf > shots[-1]['conf']:
                            shots[-1] = {'conf': conf, 'img': crop}
                            shots.sort(key=lambda x: x['conf'], reverse=True)

        # 화면에서 사라진 ID 처리
        for tid in list(best_shots_buffer.keys()):
            if current_time - best_shots_buffer[tid]['last_seen'] > 1.2: # 1.2초 대기
                data = best_shots_buffer[tid]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for i, shot in enumerate(data['shots']):
                    rank = i + 1
                    filename = f"crop_{timestamp}_ID{tid}_rank{rank}.jpg"
                    file_path = os.path.join(output_dir, filename)

                    # 이미지 저장
                    cv2.imwrite(file_path, shot['img'])
                    
                    # CSV 기록
                    with open(csv_path, 'a', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([filename, '', round(shot['conf'], 4), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tid])
                
                print(f"📸 ID {tid}: 선명도 상위 {len(data['shots'])}장 저장 완료")
                finalized_ids.add(tid)
                del best_shots_buffer[tid]

        # 디버그 화면
        display = cv2.resize(frame, (1280, 720))
        cv2.putText(display, f"Active Tracks: {len(best_shots_buffer)} | Total: {len(finalized_ids)}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Best 3 Collector", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 수집 종료. 총 {len(finalized_ids)}대의 데이터가 저장됨.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="YOLO 모델 경로")
    parser.add_argument("--source", type=str, required=True, help="영상 경로")
    args = parser.parse_args()
    run_collector(args.model, args.source)
