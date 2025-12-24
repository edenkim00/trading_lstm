"""리포트 생성 스크립트"""
import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics import PerformanceMetrics, BacktestVisualizer
from src.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='백테스트 리포트 생성')
    parser.add_argument('--result-dir', type=str, required=True, help='백테스트 결과 디렉토리')
    parser.add_argument('--output', type=str, default=None, help='출력 파일 (없으면 콘솔 출력)')
    
    args = parser.parse_args()
    
    # 로거 설정
    logger = setup_logger(level=logging.INFO)
    
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        logger.error(f"결과 디렉토리를 찾을 수 없습니다: {result_dir}")
        return 1
    
    # 데이터 로드
    logger.info(f"결과 로드: {result_dir}")
    
    equity_curve = pd.read_csv(result_dir / 'equity_curve.csv')
    equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
    
    trades = pd.read_csv(result_dir / 'trades.csv')
    if not trades.empty:
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        if 'exit_time' in trades.columns:
            trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    metrics_df = pd.read_csv(result_dir / 'metrics.csv')
    metrics = metrics_df.iloc[0].to_dict()
    
    # 리포트 생성
    report = PerformanceMetrics.format_metrics(metrics)
    
    # 출력
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"리포트 저장: {output_path}")
    else:
        print("\n")
        print(report)
    
    # 차트 재생성
    logger.info("차트 재생성...")
    results = {
        'equity_curve': equity_curve,
        'trades': trades
    }
    BacktestVisualizer.plot_summary(results, save_dir=str(result_dir), show=False)
    
    logger.info("리포트 생성 완료!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
