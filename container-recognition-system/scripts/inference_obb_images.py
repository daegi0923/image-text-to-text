import cv2
import os
import sys
import argparse
import numpy as np
from glob import glob
from ultralytics import YOLO

def get_color(cls_id):
    import random
    random.seed(cls_id * 777)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def run_obb_inference(model_path, source_dir, output_dir, conf_threshold=0.4):
    print(f"🔥 OBB 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 목록
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob(os.path.join(source_dir, ext)))
    
    if not img_files:
        print(f"❌ '{source_dir}' 폴더에 이미지가 없습니다.")
        return

    img_files.sort()
    print(f"📂 총 {len(img_files)}개 이미지 발견. 추론 시작...")
    
    # 크롭 저장 경로
    crops_dir = os.path.join(output_dir, "crops_code")
    os.makedirs(crops_dir, exist_ok=True)

    for i, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 추론
        results = model(img, conf=conf_threshold, verbose=False)
        r = results[0]
        
        annotated = img.copy()
        
        # OBB 그리기
        if hasattr(r, 'obb') and r.obb is not None:
            for idx, obb in enumerate(r.obb):
                cls_id = int(obb.cls[0])
                conf = float(obb.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                color = get_color(cls_id)
                
                # xyxyxyxy (4, 2)
                pts = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
                
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=3)
                cv2.putText(annotated, label, (pts[0][0], pts[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # [Add] CodeArea(2) 크롭 & 저장
                if cls_id == 2:
                    try:
                        # 4개 점 정렬 (TL, TR, BR, BL)
                        # x순 정렬 -> 좌2/우2 -> y순 정렬
                        sorted_x = pts[np.argsort(pts[:, 0])]
                        left = sorted_x[:2]
                        right = sorted_x[2:]
                        
                        tl = left[np.argmin(left[:, 1])]
                        bl = left[np.argmax(left[:, 1])]
                        tr = right[np.argmin(right[:, 1])]
                        br = right[np.argmax(right[:, 1])]
                        
                        src_pts = np.array([tl, tr, br, bl], dtype="float32")
                        
                        # 너비/높이 계산
                        widthA = np.linalg.norm(br - bl)
                        widthB = np.linalg.norm(tr - tl)
                        maxWidth = max(int(widthA), int(widthB))
                        
                        heightA = np.linalg.norm(tr - br)
                        heightB = np.linalg.norm(tl - bl)
                        maxHeight = max(int(heightA), int(heightB))
                        
                        dst_pts = np.array([
                            [0, 0],
                            [maxWidth - 1, 0],
                            [maxWidth - 1, maxHeight - 1],
                            [0, maxHeight - 1]
                        ], dtype="float32")
                        
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        crop = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
                        
                        # 저장
                        crop_fname = f"crop_{os.path.basename(img_path).split('.')[0]}_{idx}.jpg"
                        cv2.imwrite(os.path.join(crops_dir, crop_fname), crop)
                        
                    except Exception as e:
                        print(f"⚠️ 크롭 실패 ({os.path.basename(img_path)}): {e}")

        # 일반 Box가 있다면 그것도 (보조)
        elif hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {float(box.conf[0]):.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 저장
        fname = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"res_{fname}")
        cv2.imwrite(save_path, annotated)
        
        # 화면 출력 (선택 사항, 너무 많으면 주석 처리)
        # 1280px 정도로 리사이즈해서 보여줌
        disp = annotated.copy()
        h, w = disp.shape[:2]
        if w > 1280:
            scale = 1280 / w
            disp = cv2.resize(disp, (1280, int(h * scale)))
        
        cv2.imshow("OBB Inference Preview", disp)
        print(f"[{i+1}/{len(img_files)}] {fname} 완료")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"✨ 완료! 결과가 '{output_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="YOLO OBB 모델 (.pt)")
    parser.add_argument("--source", type=str, required=True, help="이미지 폴더 경로")
    parser.add_argument("--output", type=str, default="outputs/inference_obb_results", help="결과 저장 폴더")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    run_obb_inference(args.model, args.source, args.output, args.conf)
