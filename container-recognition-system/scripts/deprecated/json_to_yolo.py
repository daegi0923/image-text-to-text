import json
import os
import shutil

# 1. 환경 설정
json_file = 'bpt_gate.json'      # 내보낸 JSON 파일명
output_dir = './yolo_dataset'    # 결과 저장 폴더
classes = ['truck', 'container', 'code_area']

# 폴더 구조 생성
os.makedirs(f"{output_dir}/labels", exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("데이터 변환 및 이미지 복사 시작함...")

for task in data:
    for i in range(1, 5):
        cam_url_key = f'cam_0{i}_url'
        cam_to_name = f'cam{i}'
        
        img_url = task['data'].get(cam_url_key)
        if not img_url:
            continue
            
        # 1. 경로 파싱 (Label Studio URL -> 로컬 상대 경로)
        # '/data/local-files/?d=data/dataset/...' -> 'data/dataset/...'
        if '?d=' in img_url:
            local_rel_path = img_url.split('?d=')[-1]
        else:
            local_rel_path = img_url.replace('/data/local-files/', '')

        filename = os.path.basename(local_rel_path)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        
        yolo_labels = []
        
        # 2. 해당 카메라의 라벨링 데이터 추출
        for result in task['annotations'][0]['result']:
            if result.get('to_name') == cam_to_name and result['type'] == 'rectanglelabels':
                label_name = result['value']['rectanglelabels'][0]
                if label_name not in classes: continue
                class_id = classes.index(label_name)
                
                # 좌표 변환
                x, y, w, h = result['value']['x'], result['value']['y'], result['value']['width'], result['value']['height']
                cx = (x + w / 2) / 100
                cy = (y + h / 2) / 100
                nw = w / 100
                nh = h / 100
                
                yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 3. 라벨이 있는 경우에만 처리
        if yolo_labels:
            # 라벨 텍스트 저장
            with open(f"{output_dir}/labels/{label_filename}", 'w') as lf:
                lf.write('\n'.join(yolo_labels))
            
            # 이미지 파일 복사 (원본 경로에 파일이 있는지 확인 후 복사)
            if os.path.exists(local_rel_path):
                shutil.copy(local_rel_path, f"{output_dir}/images/{filename}")
            else:
                print(f"경고: 이미지를 찾을 수 없음 -> {local_rel_path}")

print(f"작업 완료! 생성된 라벨: {len(os.listdir(output_dir + '/labels'))}개, 복사된 이미지: {len(os.listdir(output_dir + '/images'))}개")