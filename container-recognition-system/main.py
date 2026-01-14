import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict
import queue
import threading
from collections import deque

from utils.config import load_config
from utils.logger import setup_logger
from utils.image_utils import apply_perspective_correction, preprocess_for_ocr
from drivers.camera import Camera
from core.detector import ContainerDetector
from services.ocr_worker import ContainerOCR

# --- [시스템 상태 관리자] ---
class TriggerManager:
    def __init__(self, duration=5.0):
        self.active = False
        self.last_trigger_time = 0
        self.duration = duration
        self.trigger_source = None

    def activate(self, source_name="unknown"):
        self.active = True
        self.last_trigger_time = time.time()
        self.trigger_source = source_name

    def update(self):
        if self.active and (time.time() - self.last_trigger_time > self.duration):
            self.active = False
            self.trigger_source = None
            return False # Deactivated just now
        return self.active

    def is_active(self):
        return self.active

# --- [세션 관리자] 결과 집계 ---
class GateSessionManager:
    def __init__(self, logger, log_file, timeout=10.0):
        self.logger = logger
        self.log_file = log_file
        self.timeout = timeout 
        self.current_session = [] 
        self.last_update_time = 0
        self.session_start_time = 0
        self.is_session_active = False

    def add_result(self, result_data):
        if not self.current_session:
            self.session_start_time = time.time()
            self.is_session_active = True
            
        self.current_session.append(result_data)
        self.last_update_time = time.time()
        self.logger.info(f"📥 [수집] {result_data['number']} (Cam:{result_data['unit']}) - 누적 {len(self.current_session)}건")

    def update(self):
        # 데이터가 들어온 지 오래됐으면 세션 종료
        if self.is_session_active and (time.time() - self.last_update_time > self.timeout):
            self.finalize_session()

    def finalize_session(self):
        if not self.current_session:
            self.is_session_active = False
            return
        
        duration = time.time() - self.session_start_time
        
        # 1. 다수결 투표
        vote_box = {}
        for item in self.current_session:
            num = item['number']
            vote_box[num] = vote_box.get(num, 0) + 1 

        sorted_votes = sorted(vote_box.items(), key=lambda x: x[1], reverse=True)
        winner_num, votes = sorted_votes[0]
        units_involved = list(set([item['unit'] for item in self.current_session]))

        # 2. 증거 사진 선발대회 (Winner 번호를 지지한 이미지들 중 Top 3)
        all_evidence = []
        for item in self.current_session:
            if item['number'] == winner_num:
                # 각 결과에 포함된 원본 이미지 리스트를 가져옴
                if 'evidence_images' in item:
                    all_evidence.extend(item['evidence_images'])
        
        # 점수(score) 기준 내림차순 정렬
        all_evidence.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_3_evidence = all_evidence[:3]

        self.logger.info(f"🏆 [확정] {winner_num} (투표: {votes}/{len(self.current_session)}) | Cam: {units_involved}")

        # 3. 기록 저장
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'final_number': winner_num,
            'vote_count': votes,
            'total_samples': len(self.current_session),
            'units': ",".join(units_involved),
            'duration_sec': round(duration, 2),
            # 증거 사진 경로 추가
            'evidence_img_1': top_3_evidence[0]['path'] if len(top_3_evidence) > 0 else None,
            'evidence_img_2': top_3_evidence[1]['path'] if len(top_3_evidence) > 1 else None,
            'evidence_img_3': top_3_evidence[2]['path'] if len(top_3_evidence) > 2 else None,
        }
        
        df = pd.DataFrame([record])
        header = not os.path.exists(self.log_file)
        df.to_csv(self.log_file, mode='a', header=header, index=False, encoding='utf-8-sig')
        
        self.current_session = []
        self.is_session_active = False


# --- [OCR 워커] GPU 일괄 처리 ---
def ocr_global_batch_worker(task_queue, result_queue, ocr_worker, logger):
    logger.info("👨‍🍳 [OCR Worker] 대기 중...")
    
    pending_tasks = []
    accumulated_images = []
    accumulated_meta = []
    
    last_batch_time = time.time()
    MAX_BATCH_SIZE = 16
    BATCH_TIMEOUT = 0.5

    while True:
        try:
            task = task_queue.get(timeout=0.1)
            if task is None: break 
            
            valid_imgs = [img for img in task['images'] if os.path.exists(img['path'])]
            if valid_imgs:
                pending_tasks.append(task)
                for img in valid_imgs:
                    accumulated_images.append(img['path'])
                    accumulated_meta.append({
                        'task_idx': len(pending_tasks) - 1
                    })
            task_queue.task_done()
        except queue.Empty:
            pass

        # 배치 실행 조건
        is_full = len(accumulated_images) >= MAX_BATCH_SIZE
        is_timeout = (len(accumulated_images) > 0) and (time.time() - last_batch_time > BATCH_TIMEOUT)
        
        if is_full or is_timeout:
            try:
                # GPU Inference
                results = ocr_worker.process_batch(accumulated_images)
                
                # 결과 재분배
                task_results_map = {i: [] for i in range(len(pending_tasks))}
                for i, res in enumerate(results):
                    t_idx = accumulated_meta[i]['task_idx']
                    task_results_map[t_idx].append(res)

                # 각 Task별 결과 집계
                for idx, task in enumerate(pending_tasks):
                    verdict = ocr_worker.consolidate_results(task_results_map[idx])
                    if verdict['found']:
                        result_queue.put({
                            'unit': task['unit_name'],
                            'number': verdict['container_number'],
                            'track_id': task['track_id'],
                            'evidence_images': task['images'] # [중요] 원본 이미지 정보 전달
                        })
                    else:
                        pass 
            except Exception as e:
                logger.error(f"OCR Batch Error: {e}")
            
            # Reset
            pending_tasks = []
            accumulated_images = []
            accumulated_meta = []
            last_batch_time = time.time()


# --- [유틸] 점수 계산 ---
def calculate_score(image, conf, box_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 선명도 + 신뢰도 + 크기(가까울수록 큼)
    score = (sharpness * 0.4) + (conf * 1000 * 0.4) + (box_area * 0.2)
    return score

def resize_display(frame, scale=0.4):
    return cv2.resize(frame, None, fx=scale, fy=scale)

# --- [메인] ---
def main():
    config = load_config()
    sys_conf = config.get('system', {})
    param_conf = config.get('parameters', {})
    
    log_file = sys_conf.get('log_file', 'outputs/gate_log.csv')
    logger = setup_logger(log_file=log_file)
    logger.info("=== [Top-Triggered] 컨테이너 인식 시스템 시작 ===")

    # 1. 초기화
    trigger_manager = TriggerManager(duration=param_conf.get('trigger_duration', 5.0))
    session_manager = GateSessionManager(logger, log_file)
    
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    # 카메라 설정 로드
    cameras = []
    try:
        ocr_worker = ContainerOCR(model_name=config['model'].get('ocr_model', 'paddle'))
        
        for cam_conf in sys_conf.get('cameras', []):
            name = cam_conf.get('name')
            role = cam_conf.get('role', 'slave') 
            src = cam_conf.get('source')
            if not src: continue

            try:
                cam = Camera(src)
                detector = ContainerDetector(
                    model_path=cam_conf.get('weights'),
                    conf_threshold=config['model'].get('conf_threshold', 0.5)
                )
                cameras.append({
                    'name': name, 'role': role, 'cam': cam, 'detector': detector,
                    'zone': cam_conf.get('detection_zone'),
                    'targets': cam_conf.get('target_classes'), 
                    'buffer': {}, 
                    'fps': cam.fps, 'acc': 0.0
                })
                logger.info(f"🎥 [{role.upper()}] {name} 준비 완료")
            except Exception as e:
                logger.error(f"❌ 카메라 로드 실패 ({name}): {e}")

        if not cameras: 
            logger.error("카메라가 없습니다.")
            return

    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        return

    # OCR 워커 스레드 시작
    threading.Thread(target=ocr_global_batch_worker, 
                     args=(task_queue, result_queue, ocr_worker, logger), 
                     daemon=True).start()

    # 폴더 생성
    temp_dir = sys_conf.get('temp_frame_dir', 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)

    min_fps = min(c['fps'] for c in cameras)
    logger.info(">>> 시스템 루프 시작 (Press 'q' to exit)")

    # 2. Main Loop
    while True:
        current_time = time.time()
        
        # (1) 세션 및 트리거 상태 업데이트
        while not result_queue.empty():
            session_manager.add_result(result_queue.get())
        
        session_manager.update()
        trigger_active = trigger_manager.update()

        # (2) 프레임 읽기 (동기화)
        active_frames = []
        all_closed = True
        
        for c in cameras:
            c['acc'] += (c['fps'] / min_fps)
            n_read = int(c['acc'])
            c['acc'] -= n_read
            
            frame = None
            for _ in range(n_read):
                f = c['cam'].get_frame()
                if f is not None: frame = f
            
            if frame is not None:
                active_frames.append((frame, c))
                all_closed = False
            else:
                active_frames.append((None, c))
        
        if all_closed: break

        # (3) 감지 및 로직 수행
        display_frames = []
        
        for frame, unit in active_frames:
            if frame is None: continue
            
            disp = frame.copy()
            role = unit['role']
            fh, fw = frame.shape[:2]
            
            # ROI 표시
            z = unit['zone']
            zx1, zx2 = int(fw*z['x_min']), int(fw*z['x_max'])
            zy1, zy2 = int(fh*z['y_min']), int(fh*z['y_max'])
            
            zone_color = (0, 0, 255) if (role == 'master' and trigger_active) else \
                         (255, 0, 0) if (role == 'master') else \
                         (0, 255, 0) if (role == 'slave' and trigger_active) else (100, 100, 100)
                
            cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), zone_color, 2)
            cv2.putText(disp, f"{role.upper()}", (zx1, zy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

            # --- [CORE LOGIC] ---
            should_detect = (role == 'master') or (role == 'slave' and trigger_active)
            
            if should_detect:
                results = unit['detector'].track(frame)
                
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    for box, track_id in zip(boxes, boxes.id):
                        cls_id = int(box.cls[0])
                        
                        if unit['targets'] and cls_id not in unit['targets']:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        
                        # ROI 체크
                        if not (zx1 < cx < zx2 and zy1 < cy < zy2):
                            continue

                        cv2.rectangle(disp, (x1, y1), (x2, y2), zone_color, 2)
                        
                        # [Master]
                        if role == 'master':
                            if not trigger_active:
                                logger.info(f"🔔 [TRIGGER] {unit['name']} -> 시스템 가동")
                            trigger_manager.activate(source_name=unit['name'])

                        # [Slave]
                        elif role == 'slave':
                            tid = int(track_id)
                            conf = float(box.conf[0])
                            
                            if tid not in unit['buffer']:
                                unit['buffer'][tid] = {'images': [], 'last_seen': 0, 'sent': False}
                            
                            buf = unit['buffer'][tid]
                            buf['last_seen'] = current_time
                            
                            # 캡처 & 점수 계산
                            pad = 10
                            crop = frame[max(0, y1-pad):min(fh, y2+pad), max(0, x1-pad):min(fw, x2+pad)]
                            score = calculate_score(crop, conf, (x2-x1)*(y2-y1))
                            img_path = os.path.join(temp_dir, f"{unit['name']}_ID{tid}_{int(current_time*1000)}.jpg")
                            
                            # 버퍼 관리 (Top 5 유지)
                            if len(buf['images']) < 5: 
                                cv2.imwrite(img_path, crop)
                                buf['images'].append({'path': img_path, 'score': score})
                            else:
                                worst = min(buf['images'], key=lambda x: x['score'])
                                if score > worst['score']:
                                    try: os.remove(worst['path'])
                                    except: pass
                                    buf['images'].remove(worst)
                                    cv2.imwrite(img_path, crop)
                                    buf['images'].append({'path': img_path, 'score': score})

            # Slave 큐 전송
            if role == 'slave':
                to_remove = []
                for tid, data in unit['buffer'].items():
                    if not data['sent'] and (current_time - data['last_seen'] > 1.0):
                        if data['images']:
                            task_queue.put({
                                'unit_name': unit['name'],
                                'track_id': tid,
                                'images': data['images']
                            })
                        data['sent'] = True
                        to_remove.append(tid)
                    elif data['sent']:
                         to_remove.append(tid)
                
                for tid in to_remove:
                    del unit['buffer'][tid]

            display_frames.append(resize_display(disp))

        if display_frames:
            combined = np.hstack(display_frames)
            status_text = f"SYSTEM: {"RECORDING" if trigger_active else "IDLE"}"
            color = (0, 0, 255) if trigger_active else (0, 255, 0)
            cv2.putText(combined, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow('Container Recognition System', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    logger.info("🛑 시스템 종료 중...")
    task_queue.put(None)
    for c in cameras: c['cam'].release()
    cv2.destroyAllWindows()
    session_manager.finalize_session()

if __name__ == "__main__":
    main()
