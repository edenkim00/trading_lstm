"""전략 베이스 클래스"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class Strategy(ABC):
    """트레이딩 전략 베이스 클래스
    
    모든 전략은 이 클래스를 상속받아 구현해야 합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 전략 설정 딕셔너리
        """
        self.config = config or {}
        self.is_fitted = False
        
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        시장 데이터를 받아 트레이딩 신호를 생성합니다.
        
        Args:
            market_data: OHLCV + feature 데이터프레임
                - 필수 컬럼: timestamp, open, high, low, close, volume
                - 추가 피처들 (indicators 등)
                
        Returns:
            신호 데이터프레임:
                - timestamp: 시간
                - side: 포지션 ('long', 'short', 'flat')
                - size: 포지션 크기 (0~1, 자본 대비 비율)
                - confidence: 신호 신뢰도 (0~1)
        """
        pass
    
    @property
    def warmup_period(self) -> int:
        """
        신호 생성 전에 필요한 데이터 포인트 수
        
        Returns:
            최소 필요 데이터 개수
        """
        return self.config.get('warmup_period', 0)
    
    @property
    def name(self) -> str:
        """전략 이름"""
        return self.__class__.__name__
    
    def fit(self, *args, **kwargs):
        """
        전략 학습 (ML 기반 전략의 경우)
        
        룰 기반 전략은 구현하지 않아도 됩니다.
        """
        self.is_fitted = True
        return self
    
    def save(self, filepath: str):
        """전략 저장 (모델 가중치 등)"""
        raise NotImplementedError(f"{self.name} does not implement save()")
    
    @classmethod
    def load(cls, filepath: str):
        """저장된 전략 로드"""
        raise NotImplementedError(f"{cls.__name__} does not implement load()")
    
    def __repr__(self):
        return f"{self.name}(warmup={self.warmup_period})"
