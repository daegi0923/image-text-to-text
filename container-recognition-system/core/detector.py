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
        프레임에서 특정 클래스의 객체만 탐지하고 가장 신뢰도 높은 박스를 반환합니다.
        
        Args:
            frame: 입력 이미지
            target_classes (list): 탐지할 클래스 ID 리스트 (예: [0, 1]). None이면 모든 클래스 탐지.
            
        Returns:
            best_box (ultralytics.engine.results.Boxes): 가장 높은 점수의 박스 객체 또는 None
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        best_box = None
        max_conf = 0
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                
                # 타겟 클래스 필터링
                if target_classes is not None and cls_id not in target_classes:
                    continue
                
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box
                    
        return best_box
