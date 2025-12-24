"""신호 생성 디버깅"""
import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data import ParquetCache
from src.preprocessing import (
    DataCleaner,
    IndicatorCalculator,
    FeatureScaler,
    SequenceGenerator
)
from src.strategy import LSTMStrategy

# 설정 로드
with open('configs/features.yaml', 'r') as f:
    features_config = yaml.safe_load(f)

with open('configs/backtest.yaml', 'r') as f:
    backtest_config = yaml.safe_load(f)['backtest']

# 데이터 로드
cache = ParquetCache('data/raw')
df = cache.load('BTCUSDT', '1h')

print(f"원본 데이터: {len(df)} rows")
print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print()

# 데이터 전처리
df = DataCleaner.clean(df)
df = IndicatorCalculator.add_all_indicators(df, features_config)
df = df.dropna().reset_index(drop=True)

print(f"전처리 후: {len(df)} rows")
print()

# 백테스트 기간 필터링
start_date = backtest_config['execution']['start_date']
end_date = backtest_config['execution']['end_date']
df_backtest = df[
    (df['timestamp'] >= start_date) & 
    (df['timestamp'] <= end_date)
].reset_index(drop=True)

print(f"백테스트 기간 ({start_date} ~ {end_date}): {len(df_backtest)} rows")
print()

# Feature Scaling
scaler = FeatureScaler.load('models/lstm/scaler.pkl')
exclude_cols = ['timestamp']
df_scaled = scaler.transform(df_backtest, exclude_columns=exclude_cols)

print("Scaled 데이터:")
print(df_scaled.head())
print()

# 전략 로드
strategy = LSTMStrategy.load('models/lstm/model_strategy.pth')

print(f"전략 정보:")
print(f"  - Feature columns: {len(strategy.feature_columns)}")
print(f"  - Long threshold: {strategy.long_threshold}")
print(f"  - Short threshold: {strategy.short_threshold}")
print(f"  - Sequence length: {strategy.sequence_length}")
print()

# 신호 생성
signals = strategy.generate_signals(df_scaled)

print(f"생성된 신호: {len(signals)}")
print(f"  - Long: {(signals['side'] == 'long').sum()}")
print(f"  - Short: {(signals['side'] == 'short').sum()}")
print(f"  - Flat: {(signals['side'] == 'flat').sum()}")
print()

print("Probability 분포:")
print(f"  - Min: {signals['probability'].min():.4f}")
print(f"  - Max: {signals['probability'].max():.4f}")
print(f"  - Mean: {signals['probability'].mean():.4f}")
print(f"  - Std: {signals['probability'].std():.4f}")
print()

# 확률 구간별 개수
bins = [0, 0.3, 0.45, 0.55, 0.7, 1.0]
labels = ['<0.3', '0.3-0.45', '0.45-0.55', '0.55-0.7', '>0.7']
signals['prob_bin'] = pd.cut(signals['probability'], bins=bins, labels=labels)
print("확률 분포:")
print(signals['prob_bin'].value_counts().sort_index())
print()

# 샘플 출력
print("샘플 신호 (처음 20개):")
print(signals[['timestamp', 'side', 'probability', 'confidence']].head(20))
