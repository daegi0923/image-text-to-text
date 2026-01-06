import os
import logging
from ultralytics import YOLO

class ContainerDetector:
    def __init__(self, model_path, default_model='yolo11n.pt', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        
        if os.path.exists(model_path):
            logging.info(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            logging.warning(f"Custom model not found. Using default: {default_model}")
            self.model = YOLO(default_model)

    def detect(self, frame, target_classes=None):
        """
        [기존 호환용] 단일 객체 탐지
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        best_box = None
        max_conf = 0
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if target_classes is not None and cls_id not in target_classes:
                    continue
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box
        return best_box

    def track(self, frame, target_classes=None):
        """
        객체 추적 (Tracking) 수행
        
        Returns:
            results: YOLO 추적 결과 객체 (IDs 포함)
        """
        # persist=True는 이전 프레임의 정보를 기억해서 ID를 유지하게 함
        results = self.model.track(frame, conf=self.conf_threshold, persist=True, verbose=False)
        return results
