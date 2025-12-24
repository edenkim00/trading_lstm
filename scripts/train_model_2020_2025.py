"""2020-2025 데이터로 모델 학습"""
import argparse
import logging
from pathlib import Path
import sys

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
    SequenceGenerator,
    split_data
)
from src.strategy import LSTMStrategy
from src.utils import setup_logger


def main():
    # 로거 설정
    logger = setup_logger(level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("LSTM 모델 학습 (2020-2025)")
    logger.info("=" * 60)
    
    # 설정 로드
    with open('configs/strategy.yaml', 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    with open('configs/features.yaml', 'r') as f:
        features_config = yaml.safe_load(f)
    
    # 데이터 로드
    logger.info("데이터 로드: BTCUSDT 1h")
    cache = ParquetCache('data/raw')
    df = cache.load('BTCUSDT', '1h')
    
    if df is None or df.empty:
        logger.error("데이터를 찾을 수 없습니다.")
        return 1
    
    logger.info(f"로드된 데이터: {len(df)} rows")
    logger.info(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 데이터 정제
    logger.info("데이터 정제...")
    df = DataCleaner.clean(df)
    
    # 기술적 지표 추가
    logger.info("기술적 지표 계산...")
    df = IndicatorCalculator.add_all_indicators(df, features_config)
    
    # 타겟 생성
    logger.info("타겟 생성...")
    prediction_config = strategy_config['strategy']['prediction']
    df = SequenceGenerator.create_target(
        df,
        target_type=prediction_config['target'],
        horizon=prediction_config['horizon']
    )
    
    # NaN 제거
    df = df.dropna().reset_index(drop=True)
    logger.info(f"정제 후 데이터: {len(df)} rows")
    
    # Train/Val/Test 날짜 기반 분할
    # 2020-01-01 ~ 2024-06-30: train (약 4.5년)
    # 2024-07-01 ~ 2024-12-31: val (6개월)
    # 2025-01-01 ~ : test (2025년 데이터)
    logger.info("데이터 분할 (날짜 기반)...")
    train_df, val_df, test_df = split_data(
        df,
        train_end_date='2024-07-01',
        val_end_date='2025-01-01'
    )
    
    logger.info(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} ~ {train_df['timestamp'].max()})")
    logger.info(f"Val: {len(val_df)} rows ({val_df['timestamp'].min()} ~ {val_df['timestamp'].max()})")
    logger.info(f"Test: {len(test_df)} rows ({test_df['timestamp'].min()} ~ {test_df['timestamp'].max()})")
    
    # 피처 컬럼 선택
    exclude_cols = ['timestamp', 'target']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"피처 개수: {len(feature_columns)}")
    logger.info(f"피처 목록: {feature_columns}")
    
    # Feature Scaling
    logger.info("Feature scaling...")
    scaler_config = features_config['features']['scaler']
    scaler = FeatureScaler(scaler_type=scaler_config['type'])
    
    train_df_scaled = scaler.fit_transform(train_df, exclude_columns=exclude_cols)
    val_df_scaled = scaler.transform(val_df, exclude_columns=exclude_cols)
    test_df_scaled = scaler.transform(test_df, exclude_columns=exclude_cols)
    
    # 스케일러 저장
    scaler.save('models/lstm/scaler.pkl')
    logger.info("스케일러 저장: models/lstm/scaler.pkl")
    
    # 전략 생성
    logger.info("LSTM 전략 생성...")
    strategy_cfg = strategy_config['strategy']
    strategy_cfg['feature_columns'] = feature_columns
    strategy = LSTMStrategy(config=strategy_cfg)
    
    # 모델 학습
    logger.info("모델 학습 시작...")
    training_config = strategy_cfg['training']
    training_config['model_save_path'] = 'models/lstm/model.pth'
    
    strategy.fit(
        train_df=train_df_scaled,
        val_df=val_df_scaled,
        feature_columns=feature_columns,
        target_column='target',
        training_config=training_config
    )
    
    # 전략 저장
    strategy.save('models/lstm/model_strategy.pth')
    
    logger.info("=" * 60)
    logger.info("학습 완료!")
    logger.info("모델: models/lstm/model.pth")
    logger.info("전략: models/lstm/model_strategy.pth")
    logger.info("스케일러: models/lstm/scaler.pkl")
    logger.info("=" * 60)
    
    # 백테스트 설정도 업데이트
    logger.info("백테스트 설정 업데이트...")
    with open('configs/backtest.yaml', 'r') as f:
        backtest_config = yaml.safe_load(f)
    
    backtest_config['backtest']['execution']['start_date'] = '2025-01-01'
    backtest_config['backtest']['execution']['end_date'] = '2025-01-31'
    
    with open('configs/backtest.yaml', 'w') as f:
        yaml.dump(backtest_config, f, default_flow_style=False)
    
    logger.info("백테스트 설정 업데이트 완료 (2025년 데이터)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
