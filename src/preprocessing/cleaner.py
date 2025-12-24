"""데이터 정제 모듈"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataCleaner:
    """OHLCV 데이터 정제"""
    
    @staticmethod
    def clean(
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        fill_missing: bool = True,
        validate_ohlc: bool = True,
        remove_outliers: bool = False,
        outlier_std: float = 5.0
    ) -> pd.DataFrame:
        """
        데이터 정제
        
        Args:
            df: 원본 데이터프레임
            drop_duplicates: 중복 제거 여부
            fill_missing: 결측치 처리 여부
            validate_ohlc: OHLC 유효성 검증
            remove_outliers: 이상치 제거 여부
            outlier_std: 이상치 판정 표준편차 기준
            
        Returns:
            정제된 데이터프레임
        """
        df = df.copy()
        original_len = len(df)
        
        # 1. 중복 제거
        if drop_duplicates:
            df = DataCleaner._drop_duplicates(df)
            logger.info(f"Removed {original_len - len(df)} duplicate rows")
        
        # 2. 타임스탬프 정렬
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 3. OHLC 유효성 검증
        if validate_ohlc:
            df = DataCleaner._validate_ohlc(df)
        
        # 4. 결측치 처리
        if fill_missing:
            df = DataCleaner._fill_missing(df)
        
        # 5. 이상치 제거
        if remove_outliers:
            df = DataCleaner._remove_outliers(df, outlier_std)
        
        # 6. 음수/0 값 처리
        df = DataCleaner._fix_invalid_values(df)
        
        logger.info(f"Cleaning complete: {original_len} -> {len(df)} rows")
        
        return df
    
    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """중복 행 제거"""
        return df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    
    @staticmethod
    def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """OHLC 관계 유효성 검증
        
        High >= Open, Close, Low
        Low <= Open, Close, High
        """
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_mask.sum() > 0:
            logger.warning(f"Found {invalid_mask.sum()} rows with invalid OHLC relationships")
            
            # 잘못된 행의 high/low 수정
            df.loc[invalid_mask, 'high'] = df.loc[invalid_mask, [
                'open', 'high', 'low', 'close'
            ]].max(axis=1)
            
            df.loc[invalid_mask, 'low'] = df.loc[invalid_mask, [
                'open', 'high', 'low', 'close'
            ]].min(axis=1)
        
        return df
    
    @staticmethod
    def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
            
            # 가격 데이터: forward fill
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].fillna(method='ffill')
            
            # 볼륨: 0으로 채우기
            volume_cols = ['volume', 'quote_volume', 'trades']
            df[volume_cols] = df[volume_cols].fillna(0)
            
            # 남은 결측치 제거
            df = df.dropna()
        
        return df
    
    @staticmethod
    def _remove_outliers(df: pd.DataFrame, std_threshold: float = 5.0) -> pd.DataFrame:
        """이상치 제거 (Z-score 기반)"""
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            # Z-score 계산
            z_scores = np.abs((df[col] - mean) / std)
            
            # 이상치 마스크
            outlier_mask = z_scores > std_threshold
            
            if outlier_mask.sum() > 0:
                logger.warning(f"Removing {outlier_mask.sum()} outliers in {col}")
                df = df[~outlier_mask]
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def _fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
        """음수/0 값 수정"""
        price_cols = ['open', 'high', 'low', 'close']
        
        # 가격이 0 이하인 행 제거
        invalid_mask = (df[price_cols] <= 0).any(axis=1)
        
        if invalid_mask.sum() > 0:
            logger.warning(f"Removing {invalid_mask.sum()} rows with invalid prices (<= 0)")
            df = df[~invalid_mask]
        
        # 볼륨이 음수인 경우 0으로 변경
        volume_cols = ['volume', 'quote_volume']
        for col in volume_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                if negative_mask.sum() > 0:
                    logger.warning(f"Setting {negative_mask.sum()} negative {col} values to 0")
                    df.loc[negative_mask, col] = 0
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def resample(
        df: pd.DataFrame,
        target_interval: str,
        agg_dict: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        시간 간격 리샘플링
        
        Args:
            df: 원본 데이터프레임
            target_interval: 목표 시간 간격 (5T, 15T, 1H, 1D 등)
            agg_dict: 집계 규칙 딕셔너리
            
        Returns:
            리샘플링된 데이터프레임
        """
        if agg_dict is None:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum',
            }
        
        df = df.set_index('timestamp')
        resampled = df.resample(target_interval).agg(agg_dict)
        resampled = resampled.dropna().reset_index()
        
        logger.info(f"Resampled from {len(df)} to {len(resampled)} rows ({target_interval})")
        
        return resampled
