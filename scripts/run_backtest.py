"""백테스트 실행 스크립트"""
import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

import yaml
import pandas as pd

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import ParquetCache
from src.preprocessing import (
    DataCleaner,
    IndicatorCalculator,
    FeatureScaler,
    SequenceGenerator
)
from src.strategy import LSTMStrategy
from src.backtest import BacktestEngine, FeeModel, SlippageModel
from src.metrics import PerformanceMetrics, BacktestVisualizer
from src.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='백테스트 실행')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--interval', type=str, default='1h', help='시간 간격')
    parser.add_argument('--strategy-path', type=str, required=True, help='전략 파일 경로')
    parser.add_argument('--scaler-path', type=str, required=True, help='스케일러 파일 경로')
    parser.add_argument('--features-config', type=str, default='configs/features.yaml', help='피처 설정')
    parser.add_argument('--backtest-config', type=str, default='configs/backtest.yaml', help='백테스트 설정')
    parser.add_argument('--output-dir', type=str, default='results', help='결과 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 로거 설정
    logger = setup_logger(level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("백테스트 실행")
    logger.info("=" * 60)
    
    # 설정 로드
    with open(args.backtest_config, 'r') as f:
        backtest_config = yaml.safe_load(f)['backtest']
    
    with open(args.features_config, 'r') as f:
        features_config = yaml.safe_load(f)
    
    # 데이터 로드
    logger.info(f"데이터 로드: {args.symbol} {args.interval}")
    cache = ParquetCache('data/raw')
    df = cache.load(
        args.symbol,
        args.interval,
        start_date=backtest_config.get('start_date'),
        end_date=backtest_config.get('end_date')
    )
    
    if df is None or df.empty:
        logger.error("데이터를 찾을 수 없습니다.")
        return 1
    
    logger.info(f"로드된 데이터: {len(df)} rows")
    
    # 데이터 정제
    logger.info("데이터 전처리...")
    df = DataCleaner.clean(df)
    df = IndicatorCalculator.add_all_indicators(df, features_config)
    df = df.dropna().reset_index(drop=True)
    
    # Feature Scaling
    logger.info("Feature scaling...")
    scaler = FeatureScaler.load(args.scaler_path)
    exclude_cols = ['timestamp']
    df_scaled = scaler.transform(df, exclude_columns=exclude_cols)
    
    # 전략 로드
    logger.info(f"전략 로드: {args.strategy_path}")
    strategy = LSTMStrategy.load(args.strategy_path)
    
    # 백테스트 엔진 설정
    logger.info("백테스트 엔진 설정...")
    fee_model = FeeModel(
        maker_fee=backtest_config['fees']['maker'],
        taker_fee=backtest_config['fees']['taker']
    )
    
    slippage_model = SlippageModel(
        type=backtest_config['slippage']['type'],
        bps=backtest_config['slippage']['bps']
    )
    
    # 백테스트 실행
    logger.info("백테스트 실행...")
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=backtest_config['initial_capital'],
        fee_model=fee_model,
        slippage_model=slippage_model,
        position_sizing=backtest_config['position']['sizing_method'],
        risk_per_trade=backtest_config['position']['risk_per_trade'],
        max_position_size=backtest_config['position']['max_position_size'],
        stop_loss_pct=backtest_config['risk_management']['stop_loss'].get('value') 
            if backtest_config['risk_management']['stop_loss']['enabled'] else None,
        take_profit_pct=backtest_config['risk_management']['take_profit'].get('value')
            if backtest_config['risk_management']['take_profit']['enabled'] else None
    )
    
    results = engine.run(
        market_data=df_scaled,
        warmup_bars=backtest_config.get('warmup_bars', 0)
    )
    
    # 성과 지표 계산
    logger.info("성과 지표 계산...")
    metrics = PerformanceMetrics.calculate_all(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        initial_capital=backtest_config['initial_capital']
    )
    
    # 결과 출력
    print("\n")
    print(PerformanceMetrics.format_metrics(metrics))
    
    # 결과 저장
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"결과 저장: {output_dir}")
    
    # CSV 저장
    results['equity_curve'].to_csv(output_dir / 'equity_curve.csv', index=False)
    results['trades'].to_csv(output_dir / 'trades.csv', index=False)
    
    # 메트릭 저장
    pd.DataFrame([metrics]).to_csv(output_dir / 'metrics.csv', index=False)
    
    # 차트 생성
    logger.info("차트 생성...")
    BacktestVisualizer.plot_summary(results, save_dir=str(output_dir), show=False)
    
    logger.info("=" * 60)
    logger.info(f"백테스트 완료! 결과: {output_dir}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
