import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict
import queue
import threading

from utils.config import load_config
from utils.logger import setup_logger
from utils.visualizer import Visualizer
from utils.image_utils import apply_perspective_correction, preprocess_for_ocr
from drivers.camera import Camera
from core.detector import ContainerDetector
from services.ocr_worker import ContainerOCR

# --- 비동기 작업 처리를 위한 워커 스레드 함수 ---
def ocr_processing_thread(task_queue, ocr_worker, logger, history, system_conf):
    """
    백그라운드에서 OCR 작업을 처리하는 요리사 (Consumer)
    """
    logger.info("👨‍🍳 OCR 백그라운드 워커 시작됨")
    
    while True:
        try:
            # 큐에서 작업 가져오기 (블로킹 모드)
            task = task_queue.get()
            
            if task is None: # 종료 신호
                break
                
            unit_name = task['unit_name']
            track_id = task['track_id']
            images = task['images'] # [{'img':..., 'path':...}, ...]
            
            logger.info(f"🍳 [OCR] 처리 시작: {unit_name} ID:{track_id} (이미지 {len(images)}장)")
            
            # 디스크에 저장 (Qwen 입력용)
            img_paths = []
            for item in images:
                # 이미 경로가 있으면 쓰고, 없으면 저장 (메모리 버퍼인 경우)
                if 'path' in item and os.path.exists(item['path']):
                    img_paths.append(item['path'])
                else:
                    # 혹시 저장 안 된 이미지가 있다면 여기서 저장
                    pass 

            # OCR 수행 (Batch)
            if img_paths:
                res = ocr_worker.process_batch(img_paths)
                verdict = ocr_worker.consolidate_results(res)
                
                if verdict['found']:
                    num = verdict['container_number']
                    logger.info(f"🎉 [결과] {unit_name} ID:{track_id} -> 확정: {num}")
                    
                    # 결과 기록 (Thread-safe 하게 append)
                    record = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'unit': unit_name,
                        'track_id': track_id,
                        'number': num,
                        'confidence_votes': f"{verdict.get('voting_meta', {}).get('winner_count')}/{len(img_paths)}"
                    }
                    history.append(record)
                    
                    # CSV 즉시 저장 (옵션)
                    log_path = system_conf.get('log_file', 'outputs/gate_log.csv')
                    # 파일 I/O는 느리므로 실제 상용에선 DB나 별도 로거 사용 권장
                    # 여기선 편의상 덮어쓰기보다는 append 모드로 여는 게 좋으나, 기존 로직 유지
                    
                else:
                    logger.info(f"💨 [실패] {unit_name} ID:{track_id} -> 인식 불가")
            
            # 작업 완료 신호
            task_queue.task_done()
            
        except Exception as e:
            logger.error(f"🔥 OCR 워커 에러: {e}")
            task_queue.task_done()

def calculate_complex_score(image, conf, box_area, frame_area):
    """
    복합 점수 계산 (선명도 + 확신 + 크기)
    """
    # 1. 선명도 (0~수천) -> 정규화 필요하지만 상대 비교용으로 씀
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. 크기 비율 (0.0 ~ 1.0)
    size_ratio = box_area / (frame_area + 1e-5)
    
    # 가중치 합산 (heuristic)
    # 선명도가 제일 중요하지만, 너무 작거나 확신 낮은 건 거름
    score = (sharpness * 0.5) + (conf * 1000 * 0.3) + (size_ratio * 10000 * 0.2)
    return score, sharpness

def resize_frame(frame, scale=0.5):
    return cv2.resize(frame, None, fx=scale, fy=scale)

def main():
    config = load_config()
    system_conf = config.get('system', {})
    model_conf = config.get('model', {})
    params_conf = config.get('parameters', {})
    
    logger = setup_logger(log_file=system_conf.get('log_file', 'outputs/gate_log.csv'))
    logger.info("=== 비동기(Async) 스마트 트래킹 시스템 시작 ===")

    # 1. 초기화
    camera_units = [] 
    camera_configs = system_conf.get('cameras', [])
    global_target_classes = model_conf.get('target_classes', None)
    
    try:
        # OCR 모델 로드
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
        # 카메라 유닛 생성
        for conf in camera_configs:
            name = conf.get('name', 'unknown')
            src = conf.get('source')
            weights = conf.get('weights')
            zone = conf.get('detection_zone', {'x_min': 0.4, 'x_max': 0.6, 'y_min': 0.25, 'y_max': 0.75})
            target_classes = conf.get('target_classes', global_target_classes)
            
            if not src: continue
            try:
                cam = Camera(src)
                detector = ContainerDetector(
                    model_path=weights,
                    default_model=model_conf.get('yolo_default', 'yolo11n.pt'),
                    conf_threshold=model_conf.get('conf_threshold', 0.5)
                )
                camera_units.append({
                    'cam': cam, 'detector': detector, 'name': name,
                    'fps': cam.fps, 'acc': 0.0,
                    'zone': zone, 'target_classes': target_classes,
                    'track_buffer': {} 
                })
                logger.info(f"✅ 유닛: {name} | Targets: {target_classes}")
            except Exception as e:
                logger.error(f"❌ 유닛 실패 ({name}): {e}")

        if not camera_units: return
        min_fps = min(u['fps'] for u in camera_units)

    except Exception as e:
        logger.error(f"초기화 에러: {e}")
        return

    # 2. 비동기 큐 & 워커 설정
    task_queue = queue.Queue() # 무한 크기 큐 (메모리 주의)
    history = []
    
    # 워커 스레드 시작 (Daemon으로 실행하여 메인 종료 시 자동 종료)
    worker_thread = threading.Thread(
        target=ocr_processing_thread,
        args=(task_queue, ocr_worker, logger, history, system_conf),
        daemon=True
    )
    worker_thread.start()

    # 파라미터
    perspective_intensity = params_conf.get('perspective_intensity', 0.0)
    MAX_BUFFER_SIZE = 5      
    TRACK_PATIENCE = 1.0     # 초 단위
    
    temp_dir = system_conf.get('temp_frame_dir', 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)

    global_step = 0

    logger.info(">>> 메인 루프 시작 (카메라는 멈추지 않는다)")

    while True:
        global_step += 1
        active_frames = []
        all_closed = True
        
        # --- 프레임 읽기 ---
        for unit in camera_units:
            unit['acc'] += (unit['fps'] / min_fps)
            num_to_read = int(unit['acc'])
            unit['acc'] -= num_to_read
            
            frame = None
            for _ in range(num_to_read):
                f = unit['cam'].get_frame()
                if f is not None: frame = f
            
            if frame is not None:
                active_frames.append((frame, unit))
                all_closed = False
            else:
                active_frames.append((None, unit))

        if all_closed: 
            # 큐에 남은 작업 다 처리될 때까지 대기하고 싶으면 task_queue.join() 사용
            break

        # --- 메인 처리 루프 (Non-Blocking) ---
        display_frames = []
        current_time = time.time()

        for frame, unit in active_frames:
            if frame is None: continue
            
            disp = frame.copy()
            fh, fw = frame.shape[:2]
            frame_area = fh * fw
            zone = unit['zone']
            buffer = unit['track_buffer']
            
            # 존 표시
            zx1, zx2 = int(fw * zone['x_min']), int(fw * zone['x_max'])
            zy1, zy2 = int(fh * zone['y_min']), int(fh * zone['y_max'])
            cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), (255, 200, 0), 2)
            
            # --- Tracking ---
            results = unit['detector'].track(frame)
            
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes
                
                for box, track_id in zip(boxes, boxes.id):
                    tid = int(track_id)
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if unit['target_classes'] and cls_id not in unit['target_classes']:
                        continue
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    box_area = (x2-x1) * (y2-y1)
                    
                    is_centered = (zx1 < cx < zx2) and (zy1 < cy < zy2)
                    
                    color = (0, 255, 0) if is_centered else (0, 0, 255)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(disp, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if is_centered:
                        # 버퍼 관리
                        if tid not in buffer:
                            buffer[tid] = {'images': [], 'last_seen': current_time, 'enqueued': False}
                        
                        buffer[tid]['last_seen'] = current_time
                        
                        # 전처리 & 점수 계산
                        pad = 20
                        crop = frame[max(0, y1-pad):min(fh, y2+pad), max(0, x1-pad):min(fw, x2+pad)].copy()
                        pre = preprocess_for_ocr(crop)
                        final_img = apply_perspective_correction(pre, intensity=perspective_intensity)
                        
                        score, sharpness = calculate_complex_score(final_img, conf, box_area, frame_area)
                        
                        # 파일 경로 미리 생성 (나중에 쓰기 위해)
                        img_path = os.path.join(temp_dir, f"{unit['name']}_ID{tid}_{global_step}.jpg")
                        
                        img_entry = {'score': score, 'sharpness': sharpness, 'path': img_path, 'img': final_img}
                        
                        # A컷 경쟁 (Top-K 유지)
                        stored = buffer[tid]['images']
                        if len(stored) < MAX_BUFFER_SIZE:
                            # 디스크 쓰기 (여기서 쓰면 약간 느려질 수 있지만, 워커 부담 줄임)
                            cv2.imwrite(img_path, final_img)
                            stored.append(img_entry)
                        else:
                            # 꼴등 찾기 (점수 기준)
                            min_score_idx = min(range(len(stored)), key=lambda i: stored[i]['score'])
                            if score > stored[min_score_idx]['score']:
                                # 기존 파일 삭제 (선택사항)
                                try: os.remove(stored[min_score_idx]['path']) 
                                except: pass
                                
                                # 새 파일 쓰기 & 교체
                                cv2.imwrite(img_path, final_img)
                                stored[min_score_idx] = img_entry
                        
                        cv2.putText(disp, f"Sc:{int(score)}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- 퇴장 체크 & 큐 전송 ---
            ids_to_remove = []
            for tid, data in buffer.items():
                if data['enqueued']: continue
                
                # 사라진 지 오래됐으면
                if current_time - data['last_seen'] > TRACK_PATIENCE:
                    if data['images']:
                        # ★ 큐에 작업 던지기 (Non-blocking)
                        logger.info(f"🚀 [전송] {unit['name']} ID:{tid} -> OCR 큐 ({len(data['images'])}장)")
                        
                        task = {
                            'unit_name': unit['name'],
                            'track_id': tid,
                            'images': data['images'] # [{'path':...}, ...]
                        }
                        task_queue.put(task)
                    
                    data['enqueued'] = True
                    ids_to_remove.append(tid)

            # 메모리 정리
            for tid in list(buffer.keys()):
                if buffer[tid]['enqueued'] and (current_time - buffer[tid]['last_seen'] > TRACK_PATIENCE * 2):
                    del buffer[tid]

            # 큐 상태 표시
            q_size = task_queue.qsize()
            cv2.putText(disp, f"OCR Queue: {q_size}", (10, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            display_frames.append(resize_frame(disp, scale=0.4))

        if display_frames:
            combined = np.hstack(display_frames)
            cv2.imshow('Async System', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 종료 처리
    logger.info("🛑 시스템 종료 요청. 잔여 작업 처리 중...")
    task_queue.put(None) # 워커 종료 신호
    worker_thread.join() # 워커 끝날 때까지 대기
    
    for unit in camera_units: unit['cam'].release()
    cv2.destroyAllWindows()
    
    if history:
        pd.DataFrame(history).to_csv(system_conf.get('log_file', 'outputs/gate_log.csv'), index=False)
        logger.info("💾 로그 저장 완료")

if __name__ == "__main__":
    main()
