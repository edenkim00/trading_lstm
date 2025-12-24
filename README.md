# Trading LSTM - 백테스트 프레임워크

LSTM 기반 암호화폐 트레이딩 전략의 백테스트를 위한 확장 가능한 프레임워크입니다.

## 🏗️ 아키텍처 개요

```
데이터 수집 → 전처리 → 전략 레이어 → 백테스트 엔진 → 분석/리포팅
  (Binance)   (Feature)  (Signal)    (Portfolio)   (Metrics)
```

### 주요 레이어

1. **데이터 레이어** (`src/data/`)
   - Binance API를 통한 실시간 및 과거 데이터 수집
   - Parquet 기반 로컬 캐싱 (심볼/인터벌별 파티셔닝)
   - Rate limit 처리 및 재시도 로직

2. **전처리 레이어** (`src/preprocessing/`)
   - OHLCV 데이터 정제 (중복 제거, 결측치 처리)
   - 기술적 지표 계산 (RSI, MACD, Bollinger Bands 등)
   - Feature scaling 및 LSTM용 시퀀스 생성
   - Train/Val/Test 시간 기반 분할

3. **전략 레이어** (`src/strategy/`)
   - 플러그인 방식의 전략 인터페이스
   - 다양한 전략 구현 가능 (LSTM, Rule-based, Ensemble 등)
   - 신호 생성: `{timestamp, side, size, confidence}`
   - 첫 번째 구현: LSTM 기반 방향성 예측

4. **백테스트 엔진** (`src/backtest/`)
   - 벡터화된 백테스트 실행
   - 포트폴리오 관리 (현금, 포지션, 레버리지)
   - 실행 시뮬레이션 (슬리피지, 수수료, 주문 체결)
   - 리스크 관리 (포지션 사이즈, 손절/익절)

5. **분석 레이어** (`src/metrics/`)
   - 성과 지표: Sharpe Ratio, Sortino Ratio, Max Drawdown
   - 거래 분석: Win Rate, Profit Factor, Turnover
   - 시각화: 수익 곡선, 드로우다운 차트, 신호 정확도

## 📁 프로젝트 구조

```
trading_lstm/
├── data/                          # 데이터 저장소
│   ├── raw/                      # 원본 OHLCV 데이터
│   │   └── {symbol}/{interval}/  # Parquet 파일
│   └── feature_store/            # 전처리된 피처
│       └── {symbol}/{interval}/
├── models/                        # 학습된 모델 가중치
│   └── lstm/
├── configs/                       # 설정 파일 (YAML)
│   ├── data.yaml
│   ├── features.yaml
│   ├── strategy.yaml
│   └── backtest.yaml
├── notebooks/                     # EDA 및 실험
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── config/                   # 설정 관리
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic 모델
│   ├── data/                     # 데이터 레이어
│   │   ├── __init__.py
│   │   ├── client.py            # Binance API 클라이언트
│   │   ├── downloader.py        # 데이터 다운로드
│   │   └── cache.py             # Parquet 캐시 관리
│   ├── preprocessing/            # 전처리 레이어
│   │   ├── __init__.py
│   │   ├── cleaner.py           # 데이터 정제
│   │   ├── indicators.py        # 기술적 지표
│   │   ├── scaler.py            # Feature scaling
│   │   └── sequence.py          # LSTM 시퀀스 생성
│   ├── models/                   # ML 모델
│   │   ├── __init__.py
│   │   ├── lstm.py              # LSTM 아키텍처
│   │   └── trainer.py           # 학습 유틸리티
│   ├── strategy/                 # 전략 레이어
│   │   ├── __init__.py
│   │   ├── base.py              # Strategy 추상 클래스
│   │   ├── lstm_strategy.py     # LSTM 전략 구현
│   │   └── factory.py           # 전략 팩토리
│   ├── backtest/                 # 백테스트 엔진
│   │   ├── __init__.py
│   │   ├── portfolio.py         # 포트폴리오 관리
│   │   ├── execution.py         # 주문 실행 시뮬레이션
│   │   ├── engine.py            # 백테스트 실행 엔진
│   │   └── models.py            # 수수료/슬리피지 모델
│   ├── metrics/                  # 성과 분석
│   │   ├── __init__.py
│   │   ├── performance.py       # 성과 지표 계산
│   │   └── visualization.py     # 차트 생성
│   └── utils/                    # 유틸리티
│       ├── __init__.py
│       └── logging.py
├── tests/                        # 테스트
│   ├── fixtures/                # 테스트용 데이터
│   ├── test_data/
│   ├── test_preprocessing/
│   ├── test_strategy/
│   └── test_backtest/
├── scripts/                      # 실행 스크립트
│   ├── download_data.py
│   ├── train_model.py
│   ├── run_backtest.py
│   └── generate_report.py
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 기술 스택

- **데이터 수집**: `python-binance`, `ccxt`
- **데이터 처리**: `pandas`, `numpy`, `pyarrow` (Parquet)
- **기술적 지표**: `ta`, `pandas-ta`
- **머신러닝**: `torch` (PyTorch), `scikit-learn`
- **시각화**: `matplotlib`, `plotly`, `seaborn`
- **설정 관리**: `pydantic`, `PyYAML`
- **테스트**: `pytest`, `pytest-cov`

## 🎯 LSTM 전략 (v1)

### 모델 아키텍처
```
Input: [batch, seq_len, features]
  ↓
LSTM Layer 1 (hidden_size=128)
  ↓
Dropout (0.2)
  ↓
LSTM Layer 2 (hidden_size=64)
  ↓
Dropout (0.2)
  ↓
Linear (hidden_size → 1)
  ↓
Sigmoid
  ↓
Output: Probability of price increase
```

### 입력 피처
- OHLCV (scaled)
- Returns (1, 5, 15 periods)
- Rolling volatility (10, 30 windows)
- Technical indicators: RSI, MACD, Bollinger Bands, OBV
- Volume features

### 학습 설정
- **Sequence Length**: 100 bars
- **Prediction Horizon**: 다음 1 bar의 방향
- **Loss**: Binary Cross Entropy
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Early Stopping**: 10 epochs patience

### 신호 생성 로직
- `p > 0.55`: Long 진입
- `p < 0.45`: Short 진입 (또는 청산)
- `0.45 ≤ p ≤ 0.55`: 관망 (Hold)

## 🚀 빠른 시작

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 다운로드
```bash
python scripts/download_data.py --symbol BTCUSDT --interval 1h --start 2023-01-01
```

### 3. 모델 학습
```bash
python scripts/train_model.py --config configs/strategy.yaml
```

### 4. 백테스트 실행
```bash
python scripts/run_backtest.py --strategy lstm --config configs/backtest.yaml
```

### 5. 결과 분석
```bash
python scripts/generate_report.py --backtest-id latest
```

## 📊 백테스트 설정

### 기본 파라미터
- **Initial Capital**: $10,000
- **Trading Fee**: 0.1% (Binance Spot)
- **Slippage**: 0.05% (시장가 주문 가정)
- **Position Sizing**: Kelly Criterion 또는 Fixed %
- **Max Leverage**: 1x (현물)
- **Risk per Trade**: 2% of equity

### 성과 지표
- Total Return
- Sharpe Ratio (annualized)
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Average Trade Duration
- Turnover Rate

## 🔌 전략 확장성

새로운 전략 추가는 `Strategy` 베이스 클래스를 상속받아 구현:

```python
from src.strategy.base import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        # 전략 초기화
    
    def generate_signals(self, market_data):
        """
        Args:
            market_data: DataFrame with OHLCV + features
        Returns:
            DataFrame with columns: [timestamp, side, size, confidence]
        """
        # 신호 생성 로직
        return signals
    
    @property
    def warmup_period(self):
        return 100  # 필요한 데이터 포인트 수
```

전략 등록:
```python
# src/strategy/factory.py
from .my_custom_strategy import MyCustomStrategy

STRATEGY_REGISTRY = {
    'lstm': LstmStrategy,
    'my_custom': MyCustomStrategy,
}
```

## 📈 로드맵

- [x] 프로젝트 구조 설계
- [ ] 데이터 수집 및 캐싱 시스템
- [ ] 전처리 파이프라인
- [ ] LSTM 모델 학습
- [ ] 백테스트 엔진 구현
- [ ] 성과 분석 및 시각화
- [ ] 멀티 전략 앙상블
- [ ] 실시간 트레이딩 모드
- [ ] 웹 대시보드

## 📝 라이선스

MIT License

## 🤝 기여

이슈와 PR을 환영합니다!