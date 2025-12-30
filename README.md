# image-text-to-text
```
container-recognition-system/
├── configs/                # 설정 (카메라 오프셋, YOLO/OCR 경로, ISO 규칙)
│   └── settings.yaml
├── core/                   # 핵심 엔진
│   ├── detector.py         # YOLO 추론
│   ├── tracker.py          # Local ID 추적
│   ├── session_manager.py  # 세션 생명주기 (데이터 바구니)
│   └── fusion_engine.py    # Rule 1, 2, 3 보팅 로직
├── data/                   # 데이터 저장소 (Git 관리 제외 추천)
│   ├── raw_videos/         # 원본 영상 파일
│   ├── frames/             # 추출된 학습용 이미지 (.jpg)
│   └── labels/             # 라벨 파일 (.txt)
├── drivers/                # 하드웨어 제어
│   └── camera.py           # RTSP/Video 파일 스트림 통합 처리
├── scripts/                # 데이터 구축 및 배치 작업 (추가됨)
│   ├── extract_frames.py   # 비디오에서 베스트 프레임 자동 추출
│   ├── auto_label.py       # (선택) 가학습 모델로 초기 박스 자동 생성
│   └── dataset_split.py    # Train/Val 데이터셋 분할 및 포맷 변환
├── services/               # 외부 연동 서비스
│   ├── ocr_worker.py       # OCR 엔진 (Paddle, EasyOCR 등)
│   └── validator.py        # ISO 6346 체크 디지트 유틸
├── utils/                  # 공통 유틸리티
│   ├── logger.py
│   └── visualizer.py
├── main.py                 # 실시간 가동 메인 스크립트
└── requirements.txt
```