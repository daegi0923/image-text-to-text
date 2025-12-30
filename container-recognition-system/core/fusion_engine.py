from collections import deque, Counter
import logging

class FusionEngine:
    def __init__(self, buffer_size=5, required_votes=3):
        self.buffer = deque(maxlen=buffer_size)
        self.required_votes = required_votes
        
    def add_prediction(self, prediction):
        """
        예측값을 버퍼에 추가합니다.
        """
        self.buffer.append(prediction)
        
    def get_consensus(self):
        """
        버퍼에 있는 예측값들 중 다수결로 합의된 결과를 반환합니다.
        합의되지 않았거나 데이터가 부족하면 None을 반환합니다.
        
        Returns:
            (consensus_value, count) or (None, 0)
        """
        if len(self.buffer) < self.required_votes:
            return None, 0
            
        most_common, count = Counter(self.buffer).most_common(1)[0]
        
        if count >= self.required_votes:
            return most_common, count
            
        return None, 0

    def clear(self):
        self.buffer.clear()
