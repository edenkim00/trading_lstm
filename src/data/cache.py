"""Parquet 기반 데이터 캐시"""
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)


class ParquetCache:
    """Parquet 파일 기반 데이터 캐시"""
    
    def __init__(self, base_path: str = "data/raw"):
        """
        Args:
            base_path: 캐시 루트 디렉토리
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def _get_file_path(self, symbol: str, interval: str) -> Path:
        """
        캐시 파일 경로 생성
        
        Args:
            symbol: 심볼
            interval: 시간 간격
            
        Returns:
            파일 경로
        """
        file_dir = self.base_path / symbol / interval
        file_dir.mkdir(parents=True, exist_ok=True)
        return file_dir / "data.parquet"
    
    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        compression: str = 'snappy'
    ) -> None:
        """
        데이터프레임을 Parquet 파일로 저장
        
        Args:
            df: 저장할 데이터프레임
            symbol: 심볼
            interval: 시간 간격
            compression: 압축 방식 (snappy, gzip, brotli)
        """
        file_path = self._get_file_path(symbol, interval)
        
        try:
            # 기존 데이터 로드 (있는 경우)
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                
                # 새 데이터와 병합
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # 중복 제거 및 정렬
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                df = combined_df
            
            # Parquet 저장
            df.to_parquet(
                file_path,
                engine='pyarrow',
                compression=compression,
                index=False
            )
            
            logger.info(f"Saved {len(df)} rows to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save parquet file: {e}")
            raise
    
    def load(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Parquet 파일에서 데이터 로드
        
        Args:
            symbol: 심볼
            interval: 시간 간격
            start_date: 시작 날짜 필터 (YYYY-MM-DD)
            end_date: 종료 날짜 필터 (YYYY-MM-DD)
            
        Returns:
            데이터프레임 또는 None
        """
        file_path = self._get_file_path(symbol, interval)
        
        if not file_path.exists():
            logger.warning(f"Cache file not found: {file_path}")
            return None
        
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            # 날짜 필터링
            if start_date:
                df = df[df['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load parquet file: {e}")
            return None
    
    def exists(self, symbol: str, interval: str) -> bool:
        """
        캐시 파일 존재 여부 확인
        
        Args:
            symbol: 심볼
            interval: 시간 간격
            
        Returns:
            존재 여부
        """
        file_path = self._get_file_path(symbol, interval)
        return file_path.exists()
    
    def delete(self, symbol: str, interval: str) -> None:
        """
        캐시 파일 삭제
        
        Args:
            symbol: 심볼
            interval: 시간 간격
        """
        file_path = self._get_file_path(symbol, interval)
        
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted cache file: {file_path}")
        else:
            logger.warning(f"Cache file not found: {file_path}")
    
    def get_info(self, symbol: str, interval: str) -> Optional[dict]:
        """
        캐시 파일 정보 조회
        
        Args:
            symbol: 심볼
            interval: 시간 간격
            
        Returns:
            파일 정보 딕셔너리
        """
        file_path = self._get_file_path(symbol, interval)
        
        if not file_path.exists():
            return None
        
        try:
            # Parquet 메타데이터 읽기
            parquet_file = pq.ParquetFile(file_path)
            
            # 데이터 로드 (통계용)
            df = pd.read_parquet(file_path)
            
            return {
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': parquet_file.metadata.num_columns,
                'start_date': df['timestamp'].min(),
                'end_date': df['timestamp'].max(),
                'compression': parquet_file.metadata.row_group(0).column(0).compression,
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return None
