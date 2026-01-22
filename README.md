# image-text-to-text

```
container-recognition-system/
├── configs/                # 설정 파일
│   ├── data.yaml           # 데이터셋 경로 및 클래스 설정
│   └── settings.yaml       # 카메라 오프셋, YOLO/OCR 설정
├── core/                   # 핵심 엔진
│   ├── detector.py         # YOLO 기반 객체 탐지
│   └── fusion_engine.py    # 결과 통합 및 보팅 로직
├── data/                   # 데이터 저장소 (Git 관리 제외)
│   ├── dataset/            # 가공된 데이터셋
├── drivers/                # 하드웨어 제어
│   └── camera.py           # RTSP/Video 파일 스트림 통합 처리
├── models/                 # 학습된 모델 가중치 (.pt)
├── outputs/                # 추론 결과 및 로그
├── scripts/                # 데이터 구축 및 분석 도구
│   ├── crop_code_area.py   # 컨테이너 코드 영역 크롭
│   ├── dataset_split.py    # Train/Val 데이터셋 분할
│   ├── debug_monitor.py    # 실시간 디버깅 모니터
│   ├── inference_test.py   # 단일/배치 추론 테스트
│   ├── smart_collector.py  # 지능형 데이터 수집기
│   ├── train_yolo.py       # YOLO 모델 학습 스크립트
│   └── visualize_labels.py # 라벨링 결과 시각화 확인
├── services/               # 외부 연동 및 유틸 서비스
│   ├── ocr_worker.py       # OCR 엔진 (PaddleOCR 등)
│   └── validator.py        # ISO 6346 체크 디지트 검증
├── utils/                  # 공통 유틸리티
│   ├── config.py           # 설정 로드 유틸
│   ├── image_utils.py      # 이미지 처리 유틸
│   ├── logger.py           # 시스템 로그 관리
│   └── visualizer.py       # 결과 시각화 도구
├── main.py                 # 시스템 통합 실행 메인 스크립트
└── requirements.txt        # 의존성 패키지 목록
```
