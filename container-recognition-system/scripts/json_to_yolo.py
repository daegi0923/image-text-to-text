import json
import os

# 1. 환경 설정
json_file = 'bpt_gate.json'  # 내보낸 JSON 파일명
output_dir = './yolo_dataset'    # 결과 저장 폴더
classes = ['truck', 'container', 'code_area'] # 클래스 순서 (중요!)

os.makedirs(f"{output_dir}/labels", exist_ok=True)

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for task in data:
    # 각 카메라별(cam1~cam4) 결과 순회
    for i in range(1, 5):
        cam_url_key = f'cam_0{i}_url'
        cam_to_name = f'cam{i}'
        
        # 1. 카메라 데이터가 없는 경우(예: cam_03) 안전하게 건너뛰기
        img_url = task['data'].get(cam_url_key)
        if not img_url:
            continue
            
        # 2. 파일명 깔끔하게 추출
        # '/data/local-files/?d=path/to/image.jpg'에서 'image.jpg'만 추출
        if '?d=' in img_url:
            actual_path = img_url.split('?d=')[-1]
            filename = os.path.basename(actual_path)
        else:
            filename = os.path.basename(img_url)
            
        label_filename = os.path.splitext(filename)[0] + '.txt'
        
        yolo_labels = []
        
        # 해당 카메라의 라벨 찾기
        for result in task['annotations'][0]['result']:
            # Label Studio의 to_name(cam1, cam2 등)과 현재 루프의 카메라 매칭
            if result.get('to_name') == cam_to_name and result['type'] == 'rectanglelabels':
                label_name = result['value']['rectanglelabels'][0]
                if label_name not in classes: 
                    continue
                class_id = classes.index(label_name)
                
                # 좌표 변환 (Label Studio % -> YOLO 0~1)
                x = result['value']['x'] / 100
                y = result['value']['y'] / 100
                w = result['value']['width'] / 100
                h = result['value']['height'] / 100
                
                cx = x + (w / 2)
                cy = y + (h / 2)
                
                yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 텍스트 파일 저장 (라벨이 있는 경우만)
        if yolo_labels:
            with open(f"{output_dir}/labels/{label_filename}", 'w') as lf:
                lf.write('\n'.join(yolo_labels))

print(f"변환 완료! {output_dir}/labels 폴더에 {len(os.listdir(output_dir + '/labels'))}개의 파일이 생성되었습니다.")