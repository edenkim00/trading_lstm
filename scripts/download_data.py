"""데이터 다운로드 스크립트"""
import argparse
import logging
from pathlib import Path
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataDownloader
from src.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='바이낸스에서 OHLCV 데이터 다운로드')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--interval', type=str, default='1h', help='시간 간격 (1m, 5m, 1h, 1d 등)')
    parser.add_argument('--start', type=str, required=True, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/raw', help='출력 디렉토리')
    parser.add_argument('--force', action='store_true', help='기존 데이터 덮어쓰기')
    
    args = parser.parse_args()
    
    # 로거 설정
    logger = setup_logger(level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("데이터 다운로드 시작")
    logger.info("=" * 60)
    logger.info(f"심볼: {args.symbol}")
    logger.info(f"간격: {args.interval}")
    logger.info(f"기간: {args.start} ~ {args.end or '현재'}")
    logger.info(f"출력: {args.output}")
    
    # 다운로더 생성
    downloader = DataDownloader(storage_path=args.output)
    
    # 다운로드
    try:
        df = downloader.download_single(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
            force_update=args.force
        )
        
        if df.empty:
            logger.error("데이터를 다운로드하지 못했습니다.")
            return 1
        
        logger.info("=" * 60)
        logger.info(f"다운로드 완료: {len(df)} rows")
        logger.info(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        logger.info("=" * 60)
        
        # 간단한 통계
        logger.info(f"시작가: ${df['open'].iloc[0]:.2f}")
        logger.info(f"종가: ${df['close'].iloc[-1]:.2f}")
        logger.info(f"최고가: ${df['high'].max():.2f}")
        logger.info(f"최저가: ${df['low'].min():.2f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"다운로드 실패: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
