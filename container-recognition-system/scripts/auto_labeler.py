import sys
import os
import pandas as pd
from tqdm import tqdm
import logging

# 프로젝트 루트 경로 추가 (services, utils 불러오기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ocr_worker import ContainerOCR

def auto_label_paddle(data_dir="data/collected_samples"):
    csv_path = os.path.join(data_dir, "labels.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일이 없습니다: {csv_path}")
        return

    # CSV 로드
    df = pd.read_csv(csv_path)
    
    # 작업 대상: label이 비어있거나 NaN인 것들
    targets = df[df['label'].isna() | (df['label'] == '')]
    
    if len(targets) == 0:
        print("✅ 모든 데이터가 이미 라벨링 되어 있습니다.")
        return

    print(f"🔍 총 {len(targets)}개의 이미지에 대해 PaddleOCR 자동 라벨링을 시작합니다...")

    # PaddleOCR 강제 지정
    try:
        # 로그가 너무 많이 찍히는 걸 방지하기 위해 로깅 레벨 조절
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        ocr_worker = ContainerOCR(model_name="paddle")
    except Exception as e:
        print(f"❌ PaddleOCR 로드 실패: {e}")
        return

    success_count = 0
    
    for idx, row in tqdm(targets.iterrows(), total=len(targets)):
        img_name = row['filename']
        img_path = os.path.join(data_dir, img_name)
        
        if not os.path.exists(img_path):
            continue

        try:
            # ContainerOCR의 process_batch 사용 (내부적으로 회전 3번 시도함)
            results = ocr_worker.process_batch([img_path])
            
            if results and results[0].get('found'):
                prediction = results[0]['container_number']
                
                # DataFrame 업데이트
                df.at[idx, 'label'] = prediction
                success_count += 1
            else:
                # 못 찾았으면 빈칸 유지 (나중에 수동으로 채우기 위해)
                pass

        except Exception as e:
            print(f"❌ 처리 에러 ({img_name}): {e}")

    # 최종 결과 저장
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 완료! {success_count}개의 라벨을 PaddleOCR로 채웠습니다.")
    print(f"📂 파일 위치: {csv_path}")
    print("👉 이제 엑셀이나 메모장으로 labels.csv 열어서 검토만 하세요.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/collected_samples", help="이미지와 CSV가 있는 폴더")
    args = parser.parse_args()
    
    auto_label_paddle(args.dir)
