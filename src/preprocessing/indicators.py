"""기술적 지표 계산"""
import logging
from typing import List, Optional

import pandas as pd
import numpy as np
import ta

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """기술적 지표 계산기"""
    
    @staticmethod
    def add_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 15]) -> pd.DataFrame:
        """
        수익률 계산
        
        Args:
            df: 데이터프레임
            periods: 계산할 기간 리스트
            
        Returns:
            수익률 컬럼이 추가된 데이터프레임
        """
        df = df.copy()
        
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        logger.info(f"Added return features for periods: {periods}")
        return df
    
    @staticmethod
    def add_volatility(df: pd.DataFrame, windows: List[int] = [10, 30]) -> pd.DataFrame:
        """
        변동성 계산
        
        Args:
            df: 데이터프레임
            windows: 롤링 윈도우 리스트
            
        Returns:
            변동성 컬럼이 추가된 데이터프레임
        """
        df = df.copy()
        
        # 로그 수익률 (1-period)
        if 'log_return_1' not in df.columns:
            df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        
        for window in windows:
            # 표준편차 기반 변동성
            df[f'volatility_{window}'] = df['log_return_1'].rolling(window).std()
            
            # Parkinson's volatility (High-Low)
            df[f'parkinson_vol_{window}'] = np.sqrt(
                1 / (4 * np.log(2)) * 
                (np.log(df['high'] / df['low']) ** 2).rolling(window).mean()
            )
        
        logger.info(f"Added volatility features for windows: {windows}")
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
        """RSI (Relative Strength Index) 추가"""
        df = df.copy()
        
        for period in periods:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(
                close=df['close'], window=period
            ).rsi()
        
        logger.info(f"Added RSI for periods: {periods}")
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence) 추가"""
        df = df.copy()
        
        macd = ta.trend.MACD(
            close=df['close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        logger.info(f"Added MACD ({fast}/{slow}/{signal})")
        return df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: int = 2
    ) -> pd.DataFrame:
        """볼린저 밴드 추가"""
        df = df.copy()
        
        bb = ta.volatility.BollingerBands(
            close=df['close'],
            window=period,
            window_dev=std
        )
        
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        logger.info(f"Added Bollinger Bands ({period}, {std}σ)")
        return df
    
    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """OBV (On-Balance Volume) 추가"""
        df = df.copy()
        
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()
        
        logger.info("Added OBV")
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR (Average True Range) 추가"""
        df = df.copy()
        
        df[f'atr_{period}'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        ).average_true_range()
        
        logger.info(f"Added ATR ({period})")
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [9, 21, 50]) -> pd.DataFrame:
        """EMA (Exponential Moving Average) 추가"""
        df = df.copy()
        
        for period in periods:
            df[f'ema_{period}'] = ta.trend.EMAIndicator(
                close=df['close'],
                window=period
            ).ema_indicator()
        
        logger.info(f"Added EMA for periods: {periods}")
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """볼륨 관련 피처 추가"""
        df = df.copy()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume ratio (현재 / 20일 평균)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Buy/Sell pressure (추정)
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        logger.info("Added volume features")
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        설정에 따라 모든 지표 추가
        
        Args:
            df: 데이터프레임
            config: 지표 설정 딕셔너리
            
        Returns:
            모든 지표가 추가된 데이터프레임
        """
        df = df.copy()
        
        # Returns
        if 'returns' in config:
            df = IndicatorCalculator.add_returns(df, config['returns'].get('periods', [1, 5, 15]))
        
        # Volatility
        if 'volatility' in config:
            df = IndicatorCalculator.add_volatility(df, config['volatility'].get('windows', [10, 30]))
        
        # Technical indicators
        if 'indicators' in config:
            for indicator in config['indicators']:
                name = indicator.get('name')
                
                if name == 'rsi':
                    df = IndicatorCalculator.add_rsi(df, indicator.get('periods', [14]))
                elif name == 'macd':
                    df = IndicatorCalculator.add_macd(
                        df,
                        indicator.get('fast', 12),
                        indicator.get('slow', 26),
                        indicator.get('signal', 9)
                    )
                elif name == 'bbands':
                    df = IndicatorCalculator.add_bollinger_bands(
                        df,
                        indicator.get('period', 20),
                        indicator.get('std', 2)
                    )
                elif name == 'obv':
                    df = IndicatorCalculator.add_obv(df)
                elif name == 'atr':
                    df = IndicatorCalculator.add_atr(df, indicator.get('period', 14))
                elif name == 'ema':
                    df = IndicatorCalculator.add_ema(df, indicator.get('periods', [9, 21, 50]))
        
        # Volume features
        if config.get('volume_features'):
            df = IndicatorCalculator.add_volume_features(df)
        
        # NaN 값 처리 (지표 계산 초기 부분)
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Total features: {len(df.columns)}")
        
        return df
