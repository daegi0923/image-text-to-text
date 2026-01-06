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

# --- 게이트 세션 관리자 (기존 동일) ---
class GateEventManager:
    def __init__(self, logger, log_file, timeout=10.0):
        self.logger = logger
        self.log_file = log_file
        self.timeout = timeout 
        self.current_session = [] 
        self.last_update_time = 0
        self.session_start_time = 0 # 세션 시작 시간
        self.is_active = False

    def add_event(self, result_data):
        # 세션의 첫 데이터라면 시작 시간 기록
        if not self.current_session:
            self.session_start_time = time.time()
            
        self.current_session.append(result_data)
        self.last_update_time = time.time()
        self.is_active = True
        self.logger.info(f"📥 [세션수집] {result_data['number']} (from {result_data['unit']} ID:{result_data['track_id']}) - 현재 누적 {len(self.current_session)}건")

    def update(self):
        if not self.is_active: return
        if time.time() - self.last_update_time > self.timeout:
            self.finalize_session()

    def finalize_session(self):
        if not self.current_session:
            self.is_active = False
            return
        
        # 소요 시간 계산
        duration = time.time() - self.session_start_time
        self.logger.info(f"🔒 세션 마감! 총 {len(self.current_session)}건 데이터로 최종 판결 (소요: {duration:.2f}초)")
        
        vote_box = {}
        for item in self.current_session:
            num = item['number']
            vote_box[num] = vote_box.get(num, 0) + 1 

        sorted_votes = sorted(vote_box.items(), key=lambda x: x[1], reverse=True)
        winner_num, votes = sorted_votes[0]
        units_involved = list(set([item['unit'] for item in self.current_session]))

        self.logger.info(f"🏆 [최종확정] {winner_num} (투표: {votes}/{len(self.current_session)}) | Cam: {units_involved}")

        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'final_number': winner_num,
            'vote_count': votes,
            'total_samples': len(self.current_session),
            'units': ",".join(units_involved),
            'duration_sec': round(duration, 2) # 실제 소요 시간 기록
        }
        
        df = pd.DataFrame([record])
        header = not os.path.exists(self.log_file)
        df.to_csv(self.log_file, mode='a', header=header, index=False, encoding='utf-8-sig')
        
        self.current_session = []
        self.is_active = False


# --- [NEW] 글로벌 배치 OCR 워커 ---
def ocr_global_batch_worker(task_queue, result_queue, ocr_worker, logger):
    """
    여러 카메라/트랙의 요청을 모아서(Accumulate) 한 번에 GPU 추론 수행
    """
    logger.info("👨‍🍳 [Global Batch] OCR 통합 워커 시작")
    
    # 배치 설정
    MAX_BATCH_SIZE = 16   # 한 번에 처리할 최대 이미지 수
    BATCH_TIMEOUT = 0.5   # 0.5초 동안 안 차면 그냥 출발
    
    pending_tasks = []      # 처리 대기 중인 Task들
    accumulated_images = [] # 실제 이미지 경로 리스트
    accumulated_meta = []   # 이미지별 주인이 누구인지 (매핑용)
    
    last_batch_time = time.time()

    while True:
        try:
            # 1. 큐에서 작업 가져오기 (Timeout을 줘서 주기적으로 배출 체크)
            # 타임아웃이 발생하면 'Empty' 예외가 발생함 -> except로 이동
            task = task_queue.get(timeout=0.1)
            
            if task is None: # 종료 신호
                break
                
            # 2. 작업 축적
            unit_name = task['unit_name']
            track_id = task['track_id']
            images = task['images'] # [{'path':...}, ...]
            
            # 유효한 이미지 경로만 추출
            valid_imgs = [img for img in images if os.path.exists(img['path'])]
            
            if valid_imgs:
                pending_tasks.append(task)
                for img in valid_imgs:
                    accumulated_images.append(img['path'])
                    # 나중에 결과 나왔을 때 누구 건지 알기 위해 메타데이터 저장
                    accumulated_meta.append({
                        'unit_name': unit_name,
                        'track_id': track_id,
                        'task_idx': len(pending_tasks) - 1 # 몇 번째 Task의 이미지인지
                    })
            
            task_queue.task_done()

        except queue.Empty:
            # 큐가 비었음 (0.1초 동안 새 작업 없음)
            pass

        # 3. 배치 처리 조건 확인 (꽉 찼거나, 시간이 지났거나)
        is_full = len(accumulated_images) >= MAX_BATCH_SIZE
        is_timeout = (len(accumulated_images) > 0) and (time.time() - last_batch_time > BATCH_TIMEOUT)
        
        if is_full or is_timeout:
            batch_size = len(accumulated_images)
            logger.info(f"🔥 [GPU 실행] 이미지 {batch_size}장 일괄 처리 시작 (Tasks: {len(pending_tasks)})")
            
            try:
                # --- 소요 시간 측정 시작 ---
                start_inference = time.time()
                
                all_results = ocr_worker.process_batch(accumulated_images)
                print(all_results)
                end_inference = time.time()
                duration = end_inference - start_inference
                avg_per_img = duration / batch_size if batch_size > 0 else 0
                
                logger.info(f"⏱️ [GPU 완료] 소요시간: {duration:.2f}s (평균: {avg_per_img:.3f}s/장)")
                # -------------------------
                
                # 4. 결과 재분배 (Result Redistribution)
                # Task별로 결과를 다시 묶어야 함
                task_results_map = {i: [] for i in range(len(pending_tasks))}
                
                for i, res in enumerate(all_results):
                    owner_task_idx = accumulated_meta[i]['task_idx']
                    task_results_map[owner_task_idx].append(res)

                # 5. 각 Task별로 투표(Consolidate) 후 결과 큐 전송
                for idx, task in enumerate(pending_tasks):
                    task_imgs_results = task_results_map[idx]
                    
                    # 각 Task(ID) 내부 투표
                    verdict = ocr_worker.consolidate_results(task_imgs_results)
                    
                    if verdict['found']:
                        num = verdict['container_number']
                        result_queue.put({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'unit': task['unit_name'],
                            'track_id': task['track_id'],
                            'number': num,
                            'raw_verdict': verdict
                        })
                    else:
                        logger.info(f"💨 {task['unit_name']} ID:{task['track_id']} -> 인식 실패")

            except Exception as e:
                logger.error(f"Global Batch Error: {e}")
                import traceback
                traceback.print_exc()
            
            # 초기화
            pending_tasks = []
            accumulated_images = []
            accumulated_meta = []
            last_batch_time = time.time()


# --- 유틸 ---
def calculate_complex_score(image, conf, box_area, frame_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    size_ratio = box_area / (frame_area + 1e-5)
    score = (sharpness * 0.5) + (conf * 1000 * 0.3) + (size_ratio * 10000 * 0.2)
    return score, sharpness

def resize_frame(frame, scale=0.5):
    return cv2.resize(frame, None, fx=scale, fy=scale)

def main():
    config = load_config()
    system_conf = config.get('system', {})
    model_conf = config.get('model', {})
    params_conf = config.get('parameters', {})
    
    log_file = system_conf.get('log_file', 'outputs/gate_log.csv')
    logger = setup_logger(log_file=log_file)
    logger.info("=== 글로벌 배치 & 세션 통합 시스템 시작 ===")

    # 1. 초기화
    camera_units = [] 
    camera_configs = system_conf.get('cameras', [])
    global_target_classes = model_conf.get('target_classes', None)
    
    try:
        ocr_worker = ContainerOCR(model_name=model_conf.get('ocr_model', 'Qwen/Qwen3-VL-2B-Instruct'))
        
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

    # 2. 큐 및 매니저 설정
    task_queue = queue.Queue()
    result_queue = queue.Queue() 
    event_manager = GateEventManager(logger, log_file, timeout=10.0)

    # ★ 글로벌 배치 워커 사용
    worker_thread = threading.Thread(
        target=ocr_global_batch_worker,
        args=(task_queue, result_queue, ocr_worker, logger),
        daemon=True
    )
    worker_thread.start()

    perspective_intensity = params_conf.get('perspective_intensity', 0.0)
    MAX_BUFFER_SIZE = 5      
    TRACK_PATIENCE = 1.0     
    
    temp_dir = system_conf.get('temp_frame_dir', 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    global_step = 0

    logger.info(">>> 통합 모니터링 시작")

    while True:
        global_step += 1
        
        # 세션 업데이트
        while not result_queue.empty():
            res = result_queue.get()
            event_manager.add_event(res)
        event_manager.update()

        # 프레임 읽기
        active_frames = []
        all_closed = True
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

        if all_closed: break

        # 메인 루프
        display_frames = []
        current_time = time.time()

        for frame, unit in active_frames:
            if frame is None: continue
            
            disp = frame.copy()
            fh, fw = frame.shape[:2]
            frame_area = fh * fw
            zone = unit['zone']
            buffer = unit['track_buffer']
            
            zx1, zx2 = int(fw * zone['x_min']), int(fw * zone['x_max'])
            zy1, zy2 = int(fh * zone['y_min']), int(fh * zone['y_max'])
            cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), (255, 200, 0), 2)
            
            # Tracking
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
                        if tid not in buffer:
                            buffer[tid] = {'images': [], 'last_seen': current_time, 'enqueued': False}
                        buffer[tid]['last_seen'] = current_time
                        
                        pad = 20
                        crop = frame[max(0, y1-pad):min(fh, y2+pad), max(0, x1-pad):min(fw, x2+pad)].copy()
                        pre = preprocess_for_ocr(crop)
                        final_img = apply_perspective_correction(pre, intensity=perspective_intensity)
                        score, sharpness = calculate_complex_score(final_img, conf, box_area, frame_area)
                        
                        img_path = os.path.join(temp_dir, f"{unit['name']}_ID{tid}_{global_step}.jpg")
                        img_entry = {'score': score, 'path': img_path, 'img': final_img}
                        
                        stored = buffer[tid]['images']
                        if len(stored) < MAX_BUFFER_SIZE:
                            cv2.imwrite(img_path, final_img)
                            stored.append(img_entry)
                        else:
                            min_score_idx = min(range(len(stored)), key=lambda i: stored[i]['score'])
                            if score > stored[min_score_idx]['score']:
                                try: os.remove(stored[min_score_idx]['path']) 
                                except: pass
                                cv2.imwrite(img_path, final_img)
                                stored[min_score_idx] = img_entry

            # 퇴장 체크 -> 큐 전송
            for tid, data in buffer.items():
                if data['enqueued']: continue
                if current_time - data['last_seen'] > TRACK_PATIENCE:
                    if data['images']:
                        task_queue.put({
                            'unit_name': unit['name'],
                            'track_id': tid,
                            'images': data['images']
                        })
                    data['enqueued'] = True

            # 정리
            for tid in list(buffer.keys()):
                if buffer[tid]['enqueued'] and (current_time - buffer[tid]['last_seen'] > TRACK_PATIENCE * 2):
                    del buffer[tid]

            q_cnt = task_queue.qsize()
            session_cnt = len(event_manager.current_session)
            cv2.putText(disp, f"Q:{q_cnt} S:{session_cnt}", (10, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            display_frames.append(resize_frame(disp, scale=0.4))

        if display_frames:
            combined = np.hstack(display_frames)
            status_text = "SESSION ON" if event_manager.is_active else "IDLE"
            cv2.putText(combined, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Global Batch System', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    logger.info("🛑 시스템 종료")
    event_manager.finalize_session()
    task_queue.put(None)
    worker_thread.join()
    for unit in camera_units: unit['cam'].release()
    cv2.destroyAllWindows()
    if event_manager.log_file: logger.info("💾 로그 저장 완료")

if __name__ == "__main__":
    main()
