"""포트폴리오 관리"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .models import Trade

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """포트폴리오 상태 관리"""
    initial_capital: float
    cash: float = field(init=False)
    positions: Dict[str, float] = field(default_factory=dict)
    open_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.cash = self.initial_capital
    
    @property
    def equity(self) -> float:
        """총 자산 (현금 + 포지션 가치)"""
        position_value = sum(self.positions.values())
        return self.cash + position_value
    
    @property
    def total_return(self) -> float:
        """총 수익률"""
        return (self.equity - self.initial_capital) / self.initial_capital
    
    @property
    def total_pnl(self) -> float:
        """총 손익"""
        return self.equity - self.initial_capital
    
    def get_position_size(self, symbol: str = 'default') -> float:
        """현재 포지션 크기"""
        return self.positions.get(symbol, 0.0)
    
    def update_position(
        self,
        symbol: str,
        size: float,
        price: float
    ):
        """포지션 업데이트
        
        Args:
            symbol: 심볼
            size: 포지션 크기 (양수: long, 음수: short, 0: 청산)
            price: 현재 가격
        """
        self.positions[symbol] = size * price
    
    def record_equity(
        self,
        timestamp: datetime,
        price: float,
        symbol: str = 'default'
    ):
        """자산 곡선 기록"""
        # 현재 포지션 가치 업데이트
        current_position_size = self.get_position_size(symbol)
        if current_position_size != 0:
            # 포지션이 있으면 현재 가격으로 업데이트
            position_qty = current_position_size / price  # 대략적 계산
            self.positions[symbol] = position_qty * price
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.equity,
            'cash': self.cash,
            'position_value': sum(self.positions.values()),
            'return': self.total_return
        })
    
    def open_trade(self, trade: Trade) -> bool:
        """거래 시작
        
        Args:
            trade: Trade 객체
            
        Returns:
            성공 여부
        """
        # 거래 비용 계산
        cost = trade.entry_price * trade.size + trade.entry_fee + trade.slippage
        
        # 현금 확인
        if cost > self.cash:
            logger.warning(f"Insufficient cash: {self.cash} < {cost}")
            return False
        
        # 현금 차감
        self.cash -= cost
        
        # 거래 기록
        self.open_trades.append(trade)
        
        return True
    
    def close_trade(
        self,
        trade: Trade,
        exit_time: datetime,
        exit_price: float,
        exit_fee: float = 0.0
    ):
        """거래 청산
        
        Args:
            trade: 청산할 Trade 객체
            exit_time: 청산 시간
            exit_price: 청산 가격
            exit_fee: 청산 수수료
        """
        # 거래 청산
        trade.close(exit_time, exit_price, exit_fee)
        
        # 현금 회수
        if trade.side == 'long':
            proceeds = exit_price * trade.size - exit_fee
        else:
            proceeds = trade.entry_price * trade.size + (trade.entry_price - exit_price) * trade.size - exit_fee
        
        self.cash += proceeds
        
        # 오픈 거래에서 제거, 청산 거래에 추가
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        
        logger.debug(f"Closed trade: PnL={trade.pnl:.2f}, Return={trade.return_pct:.2%}")
    
    def get_stats(self) -> Dict:
        """포트폴리오 통계"""
        winning_trades = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl and t.pnl < 0]
        
        total_trades = len(self.closed_trades)
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity,
            'total_return': self.total_return,
            'total_pnl': self.total_pnl,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades if total_trades > 0 else 0,
            'avg_win': sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'largest_win': max((t.pnl for t in winning_trades), default=0),
            'largest_loss': min((t.pnl for t in losing_trades), default=0),
        }
