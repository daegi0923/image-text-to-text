import cv2

class Visualizer:
    @staticmethod
    def draw_detection(frame, box, is_centered=False):
        """
        탐지된 박스를 그립니다.
        """
        if box is None:
            return

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        color = (0, 255, 0) if is_centered else (0, 0, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if not is_centered:
            cv2.putText(frame, "MOVE TO CENTER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def draw_status(frame, cooldown_counter, perspective_intensity):
        status = "COOLDOWN" if cooldown_counter > 0 else "READY"
        color = (0, 0, 255) if status == "COOLDOWN" else (0, 255, 0)
        
        h, w = frame.shape[:2]
        cv2.putText(frame, status, (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Calib Mode: [ / ] (Val: {perspective_intensity:.2f})", 
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    @staticmethod
    def draw_ocr_result(frame, text):
        cv2.putText(frame, f"Detecting: {text}", (160, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    @staticmethod
    def draw_roi_preview(frame, roi_img, label="ROI"):
        if roi_img is None or roi_img.size == 0:
            return
            
        ph, pw = roi_img.shape[:2]
        target_w = 200
        target_h = int(target_w * (ph / pw))
        
        if target_h > 200:
            target_h = 200
            target_w = int(target_h * (pw / ph))
            
        try:
            preview = cv2.resize(roi_img, (target_w, target_h))
            frame[0:target_h, 0:target_w] = preview
            cv2.putText(frame, label, (5, target_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        except Exception:
            pass
