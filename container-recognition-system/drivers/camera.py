import cv2
import logging
import threading
import time

class Camera:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 스레드 제어 변수
        self.stopped = False
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        
        # 초기 프레임 한 번 읽기 (성공 여부 확인)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            logging.warning(f"Camera start failed: {source}")
        
        # 백그라운드 스레드 시작
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        
        logging.info(f"Camera initialized (Threaded): {source} ({self.width}x{self.height} @ {self.fps:.2f}fps)")

    def update(self):
        """
        백그라운드에서 계속 프레임을 읽어서 최신 프레임만 유지한다.
        (버퍼링 방지 핵심 로직)
        """
        while not self.stopped:
            if not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            
            with self.lock:
                self.ret = ret
                self.frame = frame
                
            if not ret:
                # 영상이 끝났거나(파일인 경우) 에러
                # 파일인 경우 루프를 위해 되감기 로직을 넣을 수도 있지만,
                # RTSP라면 보통 네트워크 끊김.
                time.sleep(0.1) 

    def get_frame(self):
        """
        가장 최근에 읽힌 프레임을 반환한다.
        """
        with self.lock:
            if not self.ret or self.frame is None:
                return None
            return self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()