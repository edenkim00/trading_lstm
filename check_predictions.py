"""모델 예측값 직접 확인"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data import ParquetCache
from src.preprocessing import DataCleaner, IndicatorCalculator, FeatureScaler, SequenceGenerator
from src.strategy import LSTMStrategy

print("=" * 60)
print("모델 예측값 분석")
print("=" * 60)

# 설정 로드
with open('configs/strategy.yaml') as f:
    strategy_config = yaml.safe_load(f)

with open('configs/features.yaml') as f:
    features_config = yaml.safe_load(f)

# Threshold 확인
signal_rules = strategy_config['strategy']['signal_rules']
print(f"\n현재 Threshold 설정:")
print(f"  Long:  {signal_rules['long_threshold']}")
print(f"  Short: {signal_rules['short_threshold']}")

# 데이터 로드
cache = ParquetCache('data/raw')
df = cache.load('BTCUSDT', '1h')

# 전처리
df = DataCleaner.clean(df)
df = IndicatorCalculator.add_all_indicators(df, features_config)

prediction_config = strategy_config['strategy']['prediction']
df = SequenceGenerator.create_target(
    df,
    target_type=prediction_config['target'],
    horizon=prediction_config['horizon']
)
df = df.dropna().reset_index(drop=True)

# 백테스트 기간만
df_test = df[df['timestamp'] >= '2025-11-01'].copy()
print(f"\n백테스트 기간: {len(df_test)} rows")
print(f"  {df_test['timestamp'].min()} ~ {df_test['timestamp'].max()}")

# 전체 데이터 스케일링 (학습 때와 동일하게)
scaler = FeatureScaler.load('models/lstm/scaler.pkl')

exclude_cols = ['timestamp', 'target']
df_all_scaled = scaler.transform(df, exclude_columns=exclude_cols)

# 백테스트 기간만 추출
df_scaled = df_all_scaled[df_all_scaled['timestamp'] >= '2025-11-01'].copy()

# 전략 로드 및 신호 생성
strategy = LSTMStrategy.load('models/lstm/model_strategy.pth')
signals = strategy.generate_signals(df_scaled)

print(f"\n예측값 통계:")
print(signals['prediction'].describe())

print(f"\n예측값 분포:")
print(f"  예측 > 0.002:  {(signals['prediction'] > 0.002).sum()} ({(signals['prediction'] > 0.002).mean()*100:.1f}%)")
print(f"  예측 > 0.001:  {(signals['prediction'] > 0.001).sum()} ({(signals['prediction'] > 0.001).mean()*100:.1f}%)")
print(f"  예측 > 0.0005: {(signals['prediction'] > 0.0005).sum()} ({(signals['prediction'] > 0.0005).mean()*100:.1f}%)")
print(f"  예측 < -0.002:  {(signals['prediction'] < -0.002).sum()} ({(signals['prediction'] < -0.002).mean()*100:.1f}%)")
print(f"  예측 < -0.001:  {(signals['prediction'] < -0.001).sum()} ({(signals['prediction'] < -0.001).mean()*100:.1f}%)")
print(f"  예측 < -0.0005: {(signals['prediction'] < -0.0005).sum()} ({(signals['prediction'] < -0.0005).mean()*100:.1f}%)")

print(f"\n신호 분포:")
print(signals['side'].value_counts())

# 상위/하위 예측값
print(f"\n가장 높은 예측 수익률 Top 10:")
print(signals.nlargest(10, 'prediction')[['timestamp', 'prediction', 'side']])

print(f"\n가장 낮은 예측 수익률 Top 10:")
print(signals.nsmallest(10, 'prediction')[['timestamp', 'prediction', 'side']])
