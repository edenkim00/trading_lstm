"""Binance API 클라이언트"""
from typing import List, Optional
from datetime import datetime, timedelta
import time
import logging

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class BinanceClient:
    """Binance API 래퍼 클래스"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Args:
            api_key: Binance API 키 (읽기 전용 가능)
            api_secret: Binance API 시크릿
            testnet: 테스트넷 사용 여부
        """
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.rate_limit_delay = 0.1  # 100ms delay between requests
        
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 조회
        
        Args:
            symbol: 거래 심볼 (예: BTCUSDT)
            interval: 시간 간격 (1m, 5m, 1h 등)
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 조회 개수 (기본 1000, 최대 1000)
            
        Returns:
            OHLCV 데이터프레임
        """
        try:
            # Binance API 호출
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=int(start_time.timestamp() * 1000) if start_time else None,
                endTime=int(end_time.timestamp() * 1000) if end_time else None,
                limit=limit
            )
            
            # DataFrame 변환
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # 데이터 타입 변환
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['quote_volume'] = df['quote_volume'].astype(float)
            df['trades'] = df['trades'].astype(int)
            
            # 필요한 컬럼만 선택
            df = df[[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'quote_volume', 'trades'
            ]]
            
            # Rate limit 준수
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise
            
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        과거 데이터 대량 조회 (페이지네이션)
        
        Args:
            symbol: 거래 심볼
            interval: 시간 간격
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD), None이면 현재까지
            
        Returns:
            OHLCV 데이터프레임
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        
        all_data = []
        current_start = start_dt
        
        # 간격별 최대 조회 기간 (1000개 제한)
        interval_timedeltas = {
            '1m': timedelta(minutes=1000),
            '5m': timedelta(minutes=5000),
            '15m': timedelta(minutes=15000),
            '1h': timedelta(hours=1000),
            '4h': timedelta(hours=4000),
            '1d': timedelta(days=1000),
        }
        
        chunk_size = interval_timedeltas.get(interval, timedelta(days=100))
        
        logger.info(f"Downloading {symbol} {interval} from {start_date} to {end_date or 'now'}")
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            
            try:
                chunk_df = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=current_end,
                    limit=1000
                )
                
                if chunk_df.empty:
                    break
                    
                all_data.append(chunk_df)
                logger.info(f"Downloaded {len(chunk_df)} rows up to {chunk_df['timestamp'].iloc[-1]}")
                
                # 다음 청크의 시작점
                current_start = chunk_df['timestamp'].iloc[-1] + timedelta(milliseconds=1)
                
            except Exception as e:
                logger.error(f"Error downloading chunk: {e}")
                break
        
        if not all_data:
            logger.warning("No data downloaded")
            return pd.DataFrame()
            
        # 모든 청크 합치기
        df = pd.concat(all_data, ignore_index=True)
        
        # 중복 제거
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Total {len(df)} rows downloaded")
        
        return df
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> dict:
        """
        거래소 정보 조회
        
        Args:
            symbol: 특정 심볼 (None이면 전체)
            
        Returns:
            거래소 정보
        """
        info = self.client.get_exchange_info()
        
        if symbol:
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    return s
            return None
            
        return info
    
    def get_server_time(self) -> datetime:
        """서버 시간 조회"""
        server_time = self.client.get_server_time()
        return datetime.fromtimestamp(server_time['serverTime'] / 1000)
