"""데이터 다운로드 및 저장"""
from pathlib import Path
from typing import List, Optional
import logging

import pandas as pd

from .client import BinanceClient
from .cache import ParquetCache

logger = logging.getLogger(__name__)


class DataDownloader:
    """데이터 다운로드 및 캐싱 관리"""
    
    def __init__(
        self,
        storage_path: str = "data/raw",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Args:
            storage_path: 데이터 저장 경로
            api_key: Binance API 키
            api_secret: Binance API 시크릿
        """
        self.client = BinanceClient(api_key, api_secret)
        self.cache = ParquetCache(storage_path)
        
    def download(
        self,
        symbols: List[str],
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        force_update: bool = False
    ) -> dict:
        """
        여러 심볼의 데이터 다운로드
        
        Args:
            symbols: 심볼 리스트
            interval: 시간 간격
            start_date: 시작 날짜
            end_date: 종료 날짜
            force_update: 기존 데이터 덮어쓰기 여부
            
        Returns:
            {symbol: 데이터프레임} 딕셔너리
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            
            # 캐시 확인
            if not force_update:
                cached_df = self.cache.load(symbol, interval)
                if cached_df is not None and not cached_df.empty:
                    logger.info(f"Found cached data for {symbol}, checking if update needed")
                    last_cached_time = cached_df['timestamp'].max()
                    
                    # 최신 데이터만 가져오기
                    if end_date:
                        end_dt = pd.to_datetime(end_date)
                        if last_cached_time >= end_dt:
                            logger.info(f"Cache is up to date for {symbol}")
                            results[symbol] = cached_df
                            continue
            
            # 데이터 다운로드
            try:
                df = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df.empty:
                    logger.warning(f"No data downloaded for {symbol}")
                    continue
                
                # 캐시 저장
                self.cache.save(df, symbol, interval)
                results[symbol] = df
                
                logger.info(f"Successfully downloaded and cached {len(df)} rows for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                continue
        
        return results
    
    def download_single(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        단일 심볼 데이터 다운로드
        
        Args:
            symbol: 심볼
            interval: 시간 간격
            start_date: 시작 날짜
            end_date: 종료 날짜
            force_update: 기존 데이터 덮어쓰기 여부
            
        Returns:
            데이터프레임
        """
        results = self.download(
            symbols=[symbol],
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            force_update=force_update
        )
        
        return results.get(symbol, pd.DataFrame())
    
    def load_cached(
        self,
        symbol: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        캐시된 데이터 로드
        
        Args:
            symbol: 심볼
            interval: 시간 간격
            
        Returns:
            데이터프레임 또는 None
        """
        return self.cache.load(symbol, interval)
