"""Feature Scaling"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

logger = logging.getLogger(__name__)


class FeatureScaler:
    """피처 스케일링"""
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        feature_range: Tuple[float, float] = (-1, 1)
    ):
        """
        Args:
            scaler_type: 스케일러 타입 (standard, minmax, robust)
            feature_range: MinMaxScaler의 범위
        """
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        self.scalers = {}  # {column_name: scaler}
        
    def fit(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> 'FeatureScaler':
        """
        스케일러 학습
        
        Args:
            df: 학습 데이터프레임
            exclude_columns: 스케일링에서 제외할 컬럼
            
        Returns:
            self
        """
        if exclude_columns is None:
            exclude_columns = ['timestamp']
        
        # 스케일링할 컬럼 선택
        scale_columns = [col for col in df.columns if col not in exclude_columns]
        
        for col in scale_columns:
            # 스케일러 생성
            scaler = self._create_scaler()
            
            # 학습
            scaler.fit(df[[col]])
            self.scalers[col] = scaler
        
        logger.info(f"Fitted {len(self.scalers)} scalers ({self.scaler_type})")
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        데이터 스케일링
        
        Args:
            df: 변환할 데이터프레임
            exclude_columns: 스케일링에서 제외할 컬럼
            
        Returns:
            스케일링된 데이터프레임
        """
        if not self.scalers:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        if exclude_columns is None:
            exclude_columns = ['timestamp']
        
        df_scaled = df.copy()
        
        for col, scaler in self.scalers.items():
            if col in df_scaled.columns and col not in exclude_columns:
                df_scaled[col] = scaler.transform(df_scaled[[col]])
        
        return df_scaled
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        스케일링 역변환
        
        Args:
            df: 역변환할 데이터프레임
            columns: 역변환할 컬럼 (None이면 모든 컬럼)
            
        Returns:
            원래 스케일로 복원된 데이터프레임
        """
        if not self.scalers:
            raise ValueError("Scaler not fitted.")
        
        df_original = df.copy()
        
        if columns is None:
            columns = list(self.scalers.keys())
        
        for col in columns:
            if col in self.scalers and col in df_original.columns:
                scaler = self.scalers[col]
                df_original[col] = scaler.inverse_transform(df_original[[col]])
        
        return df_original
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """학습 및 변환을 동시에 수행"""
        self.fit(df, exclude_columns)
        return self.transform(df, exclude_columns)
    
    def _create_scaler(self):
        """스케일러 타입에 따라 스케일러 객체 생성"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def save(self, filepath: str) -> None:
        """스케일러 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'scaler_type': self.scaler_type,
            'feature_range': self.feature_range,
            'scalers': self.scalers
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Saved scaler to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureScaler':
        """저장된 스케일러 로드"""
        save_dict = joblib.load(filepath)
        
        scaler = cls(
            scaler_type=save_dict['scaler_type'],
            feature_range=save_dict['feature_range']
        )
        scaler.scalers = save_dict['scalers']
        
        logger.info(f"Loaded scaler from {filepath}")
        return scaler


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_based: bool = True,
    train_end_date: str = None,
    val_end_date: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터를 train/val/test로 분할
    
    Args:
        df: 데이터프레임
        train_ratio: 학습 데이터 비율 (날짜 지정 안 할 경우)
        val_ratio: 검증 데이터 비율 (날짜 지정 안 할 경우)
        test_ratio: 테스트 데이터 비율 (날짜 지정 안 할 경우)
        time_based: 시간 기반 분할 (True) vs 랜덤 분할 (False)
        train_end_date: 학습 데이터 종료 날짜 (YYYY-MM-DD)
        val_end_date: 검증 데이터 종료 날짜 (YYYY-MM-DD)
        
    Returns:
        (train_df, val_df, test_df)
    """
    # 날짜 기반 분할
    if train_end_date is not None:
        if 'timestamp' not in df.columns:
            raise ValueError("timestamp column required for date-based split")
        
        train_df = df[df['timestamp'] < train_end_date].copy()
        
        if val_end_date is not None:
            val_df = df[(df['timestamp'] >= train_end_date) & 
                       (df['timestamp'] < val_end_date)].copy()
            test_df = df[df['timestamp'] >= val_end_date].copy()
        else:
            # val_end_date 없으면 나머지를 val로
            val_df = df[df['timestamp'] >= train_end_date].copy()
            test_df = pd.DataFrame()
        
        return train_df, val_df, test_df
    
    # 비율 기반 분할
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(df)
    
    if time_based:
        # 시간순 분할
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
    else:
        # 랜덤 분할
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = shuffled.iloc[:train_end].copy()
        val_df = shuffled.iloc[train_end:val_end].copy()
        test_df = shuffled.iloc[val_end:].copy()
    
    logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df
