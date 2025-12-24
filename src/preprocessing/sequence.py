"""LSTM용 시퀀스 생성"""
import logging
from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SequenceGenerator:
    """LSTM용 시퀀스 데이터 생성"""
    
    def __init__(
        self,
        sequence_length: int = 100,
        prediction_horizon: int = 1,
        stride: int = 1
    ):
        """
        Args:
            sequence_length: 입력 시퀀스 길이 (과거 몇 개 바를 볼지)
            prediction_horizon: 예측 시점 (다음 몇 번째 바를 예측할지)
            stride: 시퀀스 생성 간격
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'target',
        include_timestamp: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        시퀀스 생성
        
        Args:
            df: 데이터프레임
            feature_columns: 입력 피처 컬럼 리스트
            target_column: 타겟 컬럼 (없으면 추론 모드)
            include_timestamp: 타임스탬프 배열 포함 여부
            
        Returns:
            (X, y, timestamps) 또는 (X, None, timestamps)
            X shape: (num_sequences, sequence_length, num_features)
            y shape: (num_sequences,)
        """
        X_list = []
        y_list = []
        ts_list = []
        
        # 피처 데이터
        feature_data = df[feature_columns].values
        
        # 타겟 존재 여부 확인
        has_target = target_column in df.columns
        if has_target:
            target_data = df[target_column].values
        
        # 타임스탬프
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
        
        # 시퀀스 생성
        for i in range(0, len(df) - self.sequence_length - self.prediction_horizon + 1, self.stride):
            # 입력 시퀀스
            X_seq = feature_data[i:i + self.sequence_length]
            X_list.append(X_seq)
            
            # 타겟 (예측 시점의 값)
            if has_target:
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                y_list.append(target_data[target_idx])
            
            # 타임스탬프 (예측 시점)
            if include_timestamp and 'timestamp' in df.columns:
                ts_idx = i + self.sequence_length + self.prediction_horizon - 1
                ts_list.append(timestamps[ts_idx])
        
        X = np.array(X_list)
        y = np.array(y_list) if has_target else None
        ts = np.array(ts_list) if include_timestamp else None
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y, ts
    
    @staticmethod
    def create_target(
        df: pd.DataFrame,
        target_type: str = 'direction',
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        타겟 변수 생성
        
        Args:
            df: 데이터프레임
            target_type: 타겟 타입
                - 'direction': 방향 예측 (상승=1, 하락=0)
                - 'return': 수익률 예측
                - 'binary_return': 수익률 부호 (양수=1, 음수=0)
            horizon: 예측 시점 (다음 몇 번째 바)
            
        Returns:
            타겟 컬럼이 추가된 데이터프레임
        """
        df = df.copy()
        
        if target_type == 'direction':
            # 다음 종가가 현재 종가보다 높으면 1
            df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
            
        elif target_type == 'return':
            # 수익률 예측
            df['target'] = df['close'].pct_change(horizon).shift(-horizon)
            
        elif target_type == 'binary_return':
            # 수익률 부호
            returns = df['close'].pct_change(horizon).shift(-horizon)
            df['target'] = (returns > 0).astype(int)
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        logger.info(f"Created target '{target_type}' with horizon={horizon}")
        
        return df


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for LSTM"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            X: 입력 시퀀스 (num_sequences, sequence_length, num_features)
            y: 타겟 (num_sequences,)
            device: 디바이스 (cpu, cuda)
        """
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
