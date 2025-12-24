"""설정 관리 모듈"""
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
import yaml


class DataConfig(BaseModel):
    """데이터 수집 설정"""
    symbols: List[str] = Field(..., description="거래 심볼 리스트")
    interval: str = Field(..., description="시간 간격 (1m, 5m, 1h, 1d 등)")
    start_date: str = Field(..., description="시작 날짜 (YYYY-MM-DD)")
    end_date: str = Field(..., description="종료 날짜 (YYYY-MM-DD)")
    storage_path: str = Field(default="data/raw", description="데이터 저장 경로")
    exchange: str = Field(default="binance", description="거래소")
    
    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v):
        valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of {valid_intervals}")
        return v


class APIConfig(BaseModel):
    """API 설정"""
    rate_limit: int = Field(default=1200, description="분당 요청 제한")
    retry_attempts: int = Field(default=3, description="재시도 횟수")
    retry_delay: int = Field(default=5, description="재시도 대기 시간(초)")


class IndicatorConfig(BaseModel):
    """기술적 지표 설정"""
    name: str
    periods: Optional[List[int]] = None
    fast: Optional[int] = None
    slow: Optional[int] = None
    signal: Optional[int] = None
    period: Optional[int] = None
    std: Optional[int] = None


class FeatureConfig(BaseModel):
    """피처 엔지니어링 설정"""
    returns_periods: List[int] = Field(default=[1, 5, 15], alias="returns")
    volatility_windows: List[int] = Field(default=[10, 30], alias="volatility")
    indicators: List[IndicatorConfig] = Field(default_factory=list)
    scaler_type: str = Field(default="standard", description="스케일러 타입")
    sequence_length: int = Field(default=100, description="LSTM 시퀀스 길이")
    sequence_stride: int = Field(default=1, description="시퀀스 스트라이드")
    train_split: float = Field(default=0.7, description="학습 데이터 비율")
    val_split: float = Field(default=0.15, description="검증 데이터 비율")
    test_split: float = Field(default=0.15, description="테스트 데이터 비율")


class ModelConfig(BaseModel):
    """모델 아키텍처 설정"""
    type: str = Field(default="lstm")
    input_size: Optional[int] = None
    hidden_size: int = Field(default=128)
    num_layers: int = Field(default=2)
    dropout: float = Field(default=0.2)
    bidirectional: bool = Field(default=False)


class TrainingConfig(BaseModel):
    """학습 설정"""
    batch_size: int = Field(default=64)
    epochs: int = Field(default=100)
    learning_rate: float = Field(default=0.001)
    optimizer: str = Field(default="adam")
    loss: str = Field(default="bce")
    early_stopping_patience: int = Field(default=10)
    early_stopping_min_delta: float = Field(default=0.0001)


class SignalRulesConfig(BaseModel):
    """신호 생성 규칙"""
    long_threshold: float = Field(default=0.55)
    short_threshold: float = Field(default=0.45)
    confidence_filter: float = Field(default=0.0)


class StrategyConfig(BaseModel):
    """전략 설정"""
    name: str = Field(..., description="전략 이름")
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    signal_rules: SignalRulesConfig = Field(default_factory=SignalRulesConfig)
    warmup_period: int = Field(default=100)


class FeeConfig(BaseModel):
    """수수료 설정"""
    maker: float = Field(default=0.001)
    taker: float = Field(default=0.001)


class SlippageConfig(BaseModel):
    """슬리피지 설정"""
    type: str = Field(default="fixed")
    bps: float = Field(default=5)


class PositionConfig(BaseModel):
    """포지션 관리 설정"""
    sizing_method: str = Field(default="fixed_percent")
    risk_per_trade: float = Field(default=0.02)
    max_position_size: float = Field(default=1.0)
    leverage: int = Field(default=1)
    allow_short: bool = Field(default=False)


class StopLossConfig(BaseModel):
    """손절 설정"""
    enabled: bool = Field(default=True)
    type: str = Field(default="percent")
    value: float = Field(default=0.05)


class TakeProfitConfig(BaseModel):
    """익절 설정"""
    enabled: bool = Field(default=True)
    type: str = Field(default="percent")
    value: float = Field(default=0.10)


class RiskManagementConfig(BaseModel):
    """리스크 관리 설정"""
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    max_drawdown_stop: float = Field(default=0.20)
    daily_loss_limit: float = Field(default=0.05)


class TradingRulesConfig(BaseModel):
    """거래 규칙"""
    min_hold_period: int = Field(default=1)
    cooldown_period: int = Field(default=0)
    max_trades_per_day: int = Field(default=10)


class BacktestConfig(BaseModel):
    """백테스트 설정"""
    initial_capital: float = Field(default=10000.0)
    currency: str = Field(default="USDT")
    fees: FeeConfig = Field(default_factory=FeeConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    trading_rules: TradingRulesConfig = Field(default_factory=TradingRulesConfig)
    start_date: str = Field(..., description="백테스트 시작 날짜")
    end_date: str = Field(..., description="백테스트 종료 날짜")
    warmup_bars: int = Field(default=100)


def load_config(config_path: str, config_class: type) -> BaseModel:
    """YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        config_class: Pydantic 모델 클래스
        
    Returns:
        설정 객체
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_class(**config_dict)


def save_config(config: BaseModel, config_path: str) -> None:
    """설정 객체를 YAML 파일로 저장
    
    Args:
        config: Pydantic 설정 객체
        config_path: 저장할 파일 경로
    """
    config_dict = config.model_dump()
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
