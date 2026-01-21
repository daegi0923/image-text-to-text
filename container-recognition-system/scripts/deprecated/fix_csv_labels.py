import pandas as pd
import json
import os
import ast

INPUT_FILE = 'data/bpt_gate_260120_fixed.csv'
OUTPUT_FILE = 'data/bpt_gate_260120_final.csv'

def transform_labels():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 입력 파일 없음: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📄 데이터 로드: {len(df)}건")

    def convert_row(label_str):
        if pd.isna(label_str):
            return "[]"
            
        try:
            # 문자열 -> 리스트 변환
            # JSON 형식이 아닐 수도 있으니 ast.literal_eval 시도 후 json.loads
            try:
                data = json.loads(label_str)
            except:
                data = ast.literal_eval(label_str)
                
            if not isinstance(data, list):
                return label_str

            new_data = []
            for item in data:
                # Label Studio 포맷으로 래핑
                new_item = {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": item
                }
                new_data.append(new_item)
            
            # 다시 JSON 문자열로 변환 (공백 최소화)
            return json.dumps(new_data, separators=(',', ':'))
            
        except Exception as e:
            print(f"⚠️ 변환 실패: {label_str[:30]}... ({e})")
            return label_str

    if 'label' in df.columns:
        df['label'] = df['label'].apply(convert_row)
        print("✅ 라벨 포맷 변환 완료")
    else:
        print("❌ 'label' 컬럼이 없습니다.")
        return

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 저장 완료: {OUTPUT_FILE}")
    
    # 확인용 출력
    print("\n[샘플 출력]")
    sample = df['label'].iloc[0]
    print(sample[:150] + "..." if len(sample) > 150 else sample)

if __name__ == "__main__":
    transform_labels()
