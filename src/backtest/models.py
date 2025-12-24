"""백테스트용 모델 클래스"""
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class Trade:
    """개별 거래 기록"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    side: str = 'long'  # long, short
    size: float = 1.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage: float = 0.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    
    def close(self, exit_time: datetime, exit_price: float, exit_fee: float = 0.0):
        """거래 청산"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_fee = exit_fee
        
        # P&L 계산
        if self.side == 'long':
            gross_pnl = (exit_price - self.entry_price) * self.size
        else:  # short
            gross_pnl = (self.entry_price - exit_price) * self.size
        
        # 수수료 차감
        self.pnl = gross_pnl - self.entry_fee - self.exit_fee
        self.return_pct = self.pnl / (self.entry_price * self.size) if self.size > 0 else 0.0
    
    @property
    def is_open(self) -> bool:
        """포지션 오픈 여부"""
        return self.exit_time is None
    
    @property
    def duration(self) -> Optional[float]:
        """거래 기간 (초)"""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return None


@dataclass
class FeeModel:
    """수수료 모델"""
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    use_maker: bool = False  # 기본은 taker 수수료
    
    def calculate(self, price: float, size: float, is_maker: Optional[bool] = None) -> float:
        """수수료 계산"""
        if is_maker is None:
            is_maker = self.use_maker
        
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return price * size * fee_rate


@dataclass
class SlippageModel:
    """슬리피지 모델"""
    type: str = 'fixed'  # fixed, proportional
    bps: float = 5  # 0.05% (5 basis points)
    
    def calculate(self, price: float, side: str) -> float:
        """슬리피지 적용된 가격 계산
        
        Args:
            price: 원래 가격
            side: 'buy' 또는 'sell'
            
        Returns:
            슬리피지 적용 가격
        """
        slippage_rate = self.bps / 10000.0
        
        if side == 'buy':
            # 매수시 가격 상승
            return price * (1 + slippage_rate)
        else:
            # 매도시 가격 하락
            return price * (1 - slippage_rate)
