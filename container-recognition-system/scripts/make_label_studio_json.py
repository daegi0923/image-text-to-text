import os
import json
from datetime import datetime
from glob import glob

# 설정 (기존과 동일)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset", "raw_captures")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "dataset", "label_studio_import.json")
TIME_TOLERANCE = 10 

def parse_timestamp(filename):
    try:
        basename = os.path.basename(filename)
        parts = basename.split("_")
        timestamp_str = parts[0] + parts[1]
        return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    except Exception as e:
        return None

def find_closest_image(target_time, image_list):
    if not image_list: return None
    closest_img, min_diff = None, float('inf')
    for img in image_list:
        img_time = parse_timestamp(img)
        if img_time is None: continue
        diff = abs((img_time - target_time).total_seconds())
        if diff <= TIME_TOLERANCE and diff < min_diff:
            min_diff, closest_img = diff, img
    return closest_img

def main():
    cam_configs = {
        "cam_01": "Camera 1 (Side-R)",
        "cam_02": "Camera 2 (Top-Front)",
        "cam_03": "Camera 3 (Side-L)",
        "cam_04": "Camera 4 (Top-Back)"
    }
    
    cam_images = {}
    for cam_id in cam_configs.keys():
        path = os.path.join(DATASET_DIR, cam_id, "*.jpg")
        cam_images[cam_id] = sorted(glob(path))

    if not cam_images["cam_01"]: return

    tasks = []
    for img1 in cam_images["cam_01"]:
        t1 = parse_timestamp(img1)
        if t1 is None: continue
        
        # 핵심 변경 부분: images 리스트 생성
        image_list = []
        
        # 기준이 되는 cam_01 추가
        image_list.append({
            "url": f"/data/local-files/?d=data/dataset/raw_captures/cam_01/{os.path.basename(img1)}",
            "name": "cam_01",
            "title": cam_configs["cam_01"]
        })
        
        # 나머지 카메라 짝 찾기
        for cam_id in ["cam_02", "cam_03", "cam_04"]:
            match = find_closest_image(t1, cam_images[cam_id])
            if match:
                image_list.append({
                    "url": f"/data/local-files/?d=data/dataset/raw_captures/{cam_id}/{os.path.basename(match)}",
                    "name": cam_id,
                    "title": cam_configs[cam_id]
                })
        
        if len(image_list) >= 2:
            tasks.append({"data": {"images": image_list}})

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4, ensure_ascii=False)
    
    print(f"완료! 총 {len(tasks)}개 세트 생성됨.")

if __name__ == "__main__":
    main()