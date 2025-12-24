"""예측값 분석 스크립트"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data import ParquetCache
from src.preprocessing import DataCleaner, IndicatorCalculator, FeatureScaler, SequenceGenerator, TimeSeriesDataset
from src.strategy import LSTMStrategy

# 설정 로드
with open('configs/strategy.yaml') as f:
    strategy_config = yaml.safe_load(f)

with open('configs/features.yaml') as f:
    features_config = yaml.safe_load(f)

# 데이터 로드
cache = ParquetCache('data/raw')
df = cache.load('BTCUSDT', '1h')

# 데이터 정제
df = DataCleaner.clean(df)
df = IndicatorCalculator.add_all_indicators(df, features_config)

# 타겟 생성
prediction_config = strategy_config['strategy']['prediction']
df = SequenceGenerator.create_target(
    df,
    target_type=prediction_config['target'],
    horizon=prediction_config['horizon']
)
df = df.dropna().reset_index(drop=True)

# 최근 데이터만 (백테스트 기간)
df_test = df[df['timestamp'] >= '2025-11-01'].copy()

print(f"테스트 데이터: {len(df_test)} rows")
print(f"기간: {df_test['timestamp'].min()} ~ {df_test['timestamp'].max()}")

# 타겟 통계
print(f"\n타겟(수익률) 통계:")
print(df_test['target'].describe())
print(f"타겟 > 2%: {(df_test['target'] > 0.02).sum()} ({(df_test['target'] > 0.02).mean()*100:.1f}%)")
print(f"타겟 < -2%: {(df_test['target'] < -0.02).sum()} ({(df_test['target'] < -0.02).mean()*100:.1f}%)")

# 전략 로드
strategy = LSTMStrategy.load('models/lstm/model_strategy.pth')

# 스케일러 로드 및 스케일링
scaler = FeatureScaler()
scaler.load('models/lstm/scaler.pkl')

exclude_cols = ['timestamp', 'target']
df_scaled = scaler.transform(df_test, exclude_columns=exclude_cols)

# 신호 생성
signals = strategy.generate_signals(df_scaled)

print(f"\n신호 통계:")
print(signals['side'].value_counts())

print(f"\n예측값 통계:")
print(signals['prediction'].describe())
print(f"예측 > 2%: {(signals['prediction'] > 0.02).sum()}")
print(f"예측 < -2%: {(signals['prediction'] < -0.02).sum()}")

print(f"\n예측값 범위:")
print(f"최소: {signals['prediction'].min():.6f}")
print(f"최대: {signals['prediction'].max():.6f}")
print(f"평균: {signals['prediction'].mean():.6f}")
print(f"표준편차: {signals['prediction'].std():.6f}")

# 예측 vs 실제
merged = df_test.merge(signals[['timestamp', 'prediction', 'side']], on='timestamp', how='left')
print(f"\n예측 vs 실제 상관계수: {merged['prediction'].corr(merged['target']):.4f}")
