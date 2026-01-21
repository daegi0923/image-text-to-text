import pandas as pd
import os

CSV_FILE = 'data/bpt_gate_260120.csv'
OUTPUT_FILE = 'data/bpt_gate_260120_fixed.csv'

def fix_image_paths():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 파일 없음: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"로드된 데이터: {len(df)}행")
    
    # 변환 로직
    # Old: /data/local-files/?d=raw_captures/VIEW/FILENAME.jpg
    # New: /data/local-files/?d=bpt_gate_1/VIEW_FILENAME.jpg
    
    old_prefix = "/data/local-files/?d=raw_captures/"
    new_prefix = "/data/local-files/?d=bpt_gate_1/"
    
    def transform(path):
        if isinstance(path, str) and path.startswith(old_prefix):
            # 1. prefix 제거
            temp = path.replace(old_prefix, "")
            # 2. 슬래시(/)를 언더바(_)로 치환 (폴더 구조 -> 파일명 플랫)
            temp = temp.replace("/", "_")
            # 3. 새 prefix 붙이기
            return new_prefix + temp
        return path

    # image 컬럼 변환
    if 'image' in df.columns:
        df['image'] = df['image'].apply(transform)
        print("✅ 경로 변환 완료")
    else:
        print("❌ 'image' 컬럼을 찾을 수 없습니다.")
        return

    # 저장
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 저장 완료: {OUTPUT_FILE}")
    
    # 샘플 출력
    print("\n[변환 예시]")
    print(df['image'].head().to_string(index=False))

if __name__ == "__main__":
    fix_image_paths()
