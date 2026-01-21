import pandas as pd
import json
import os
import uuid

INPUT_FILE = 'data/bpt_gate_260120_final.csv'
OUTPUT_FILE = 'data/bpt_gate_260120.json'

def csv_to_label_studio_json():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 입력 파일 없음: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📄 데이터 로드: {len(df)}건")

    json_data = []

    for idx, row in df.iterrows():
        # 1. Label 파싱
        try:
            results = json.loads(row['label'])
        except:
            results = []

        # 2. Result 구조 조정 (id 추가 및 메타데이터 위치 조정)
        formatted_results = []
        for res in results:
            val = res.get('value', {})
            
            # 예시처럼 original_width, height 등을 상위 레벨로 추출
            orig_w = val.pop('original_width', 1920)
            orig_h = val.pop('original_height', 1080)
            
            new_res = {
                "id": str(uuid.uuid4())[:10], # 고유 ID 생성
                "type": res.get('type', 'rectanglelabels'),
                "from_name": res.get('from_name', 'label'),
                "to_name": res.get('to_name', 'image'),
                "original_width": orig_w,
                "original_height": orig_h,
                "image_rotation": 0,
                "value": val
            }
            formatted_results.append(new_res)

        # 3. 전체 구조 생성 (data + predictions)
        task = {
            "data": {
                "image": row['image']
            },
            "predictions": [{
                "model_version": "1.0",
                "score": 1.0,
                "result": formatted_results
            }]
        }
        json_data.append(task)

    # 4. JSON 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 JSON 변환 완료: {OUTPUT_FILE}")
    print(f"샘플 첫 번째 데이터 확인 완료.")

if __name__ == "__main__":
    csv_to_label_studio_json()
