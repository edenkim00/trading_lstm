"""모델 학습 스크립트"""
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
    parser = argparse.ArgumentParser(description='LSTM 모델 학습')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--interval', type=str, default='1h', help='시간 간격')
    parser.add_argument('--config', type=str, default='configs/strategy.yaml', help='전략 설정 파일')
    parser.add_argument('--features-config', type=str, default='configs/features.yaml', help='피처 설정 파일')
    parser.add_argument('--model-path', type=str, default='models/lstm/model.pth', help='모델 저장 경로')
    parser.add_argument('--scaler-path', type=str, default='models/lstm/scaler.pkl', help='스케일러 저장 경로')
    
    args = parser.parse_args()
    
    # 로거 설정
    logger = setup_logger(level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("LSTM 모델 학습")
    logger.info("=" * 60)
    
    # 설정 로드
    with open(args.config, 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    with open(args.features_config, 'r') as f:
        features_config = yaml.safe_load(f)
    
    # 데이터 로드
    logger.info(f"데이터 로드: {args.symbol} {args.interval}")
    cache = ParquetCache('data/raw')
    df = cache.load(args.symbol, args.interval)
    
    if df is None or df.empty:
        logger.error("데이터를 찾을 수 없습니다. 먼저 download_data.py를 실행하세요.")
        return 1
    
    logger.info(f"로드된 데이터: {len(df)} rows")
    
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
    
    # Train/Val/Test 분할
    logger.info("데이터 분할...")
    split_config = features_config['features']['split']
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=split_config['train'],
        val_ratio=split_config['val'],
        test_ratio=split_config['test']
    )
    
    # 피처 컬럼 선택
    exclude_cols = ['timestamp', 'target']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"피처 개수: {len(feature_columns)}")
    
    # Feature Scaling
    logger.info("Feature scaling...")
    scaler_config = features_config['features']['scaler']
    scaler = FeatureScaler(scaler_type=scaler_config['type'])
    
    train_df_scaled = scaler.fit_transform(train_df, exclude_columns=exclude_cols)
    val_df_scaled = scaler.transform(val_df, exclude_columns=exclude_cols)
    test_df_scaled = scaler.transform(test_df, exclude_columns=exclude_cols)
    
    # 스케일러 저장
    scaler.save(args.scaler_path)
    logger.info(f"스케일러 저장: {args.scaler_path}")
    
    # 전략 생성
    logger.info("LSTM 전략 생성...")
    strategy_cfg = strategy_config['strategy']
    strategy_cfg['feature_columns'] = feature_columns
    strategy = LSTMStrategy(config=strategy_cfg)
    
    # 모델 학습
    logger.info("모델 학습 시작...")
    training_config = strategy_cfg['training']
    training_config['model_save_path'] = args.model_path
    
    strategy.fit(
        train_df=train_df_scaled,
        val_df=val_df_scaled,
        feature_columns=feature_columns,
        target_column='target',
        training_config=training_config
    )
    
    # 전략 저장
    strategy.save(args.model_path.replace('.pth', '_strategy.pth'))
    
    logger.info("=" * 60)
    logger.info(f"학습 완료!")
    logger.info(f"모델: {args.model_path}")
    logger.info(f"전략: {args.model_path.replace('.pth', '_strategy.pth')}")
    logger.info(f"스케일러: {args.scaler_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
