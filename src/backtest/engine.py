"""백테스트 엔진"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from ..strategy import Strategy
from .portfolio import Portfolio
from .execution import ExecutionEngine
from .models import FeeModel, SlippageModel, Trade

logger = logging.getLogger(__name__)


class BacktestEngine:
    """벡터화된 백테스트 실행 엔진"""
    
    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 10000.0,
        fee_model: Optional[FeeModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        position_sizing: str = 'fixed_percent',
        risk_per_trade: float = 0.02,
        max_position_size: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ):
        """
        Args:
            strategy: 트레이딩 전략
            initial_capital: 초기 자본
            fee_model: 수수료 모델
            slippage_model: 슬리피지 모델
            position_sizing: 포지션 사이징 방법
            risk_per_trade: 거래당 리스크
            max_position_size: 최대 포지션 크기
            stop_loss_pct: 손절 비율
            take_profit_pct: 익절 비율
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # 포트폴리오 및 실행 엔진
        self.portfolio = Portfolio(initial_capital)
        self.execution_engine = ExecutionEngine(fee_model, slippage_model)
        
        # 결과
        self.results = None
    
    def run(
        self,
        market_data: pd.DataFrame,
        warmup_bars: int = 0
    ) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            market_data: 시장 데이터 (OHLCV + features)
            warmup_bars: 워밍업 기간 (신호 생성 전)
            
        Returns:
            백테스트 결과
        """
        logger.info(f"Starting backtest with {len(market_data)} bars")
        
        # 신호 생성
        signals = self.strategy.generate_signals(market_data)
        
        # 신호를 시장 데이터와 병합
        combined = market_data.merge(
            signals[['timestamp', 'side', 'size', 'confidence']],
            on='timestamp',
            how='left'
        )
        combined['side'] = combined['side'].fillna('flat')
        combined['size'] = combined['size'].fillna(0.0)
        combined['confidence'] = combined['confidence'].fillna(0.0)
        
        # 워밍업 기간 제외
        if warmup_bars > 0:
            combined = combined.iloc[warmup_bars:].reset_index(drop=True)
        
        # 백테스트 실행
        current_position = None  # 현재 오픈 포지션
        
        for idx, row in combined.iterrows():
            timestamp = row['timestamp']
            price = row['close']
            signal_side = row['side']
            
            # 자산 곡선 기록
            self.portfolio.record_equity(timestamp, price)
            
            # 포지션이 있는 경우 손절/익절 체크
            if current_position is not None:
                should_exit = False
                
                # 손절 체크
                if self.stop_loss_pct:
                    if current_position.side == 'long':
                        if price <= current_position.entry_price * (1 - self.stop_loss_pct):
                            should_exit = True
                            logger.debug(f"Stop loss triggered at {price}")
                
                # 익절 체크
                if self.take_profit_pct:
                    if current_position.side == 'long':
                        if price >= current_position.entry_price * (1 + self.take_profit_pct):
                            should_exit = True
                            logger.debug(f"Take profit triggered at {price}")
                
                # 청산
                if should_exit or (signal_side == 'flat' and current_position.side == 'long'):
                    exit_fee = self.execution_engine.execute_exit(
                        current_position,
                        timestamp,
                        price
                    )
                    self.portfolio.close_trade(current_position, timestamp, price, exit_fee)
                    self.portfolio.update_position('default', 0, price)
                    current_position = None
            
            # 새 포지션 진입
            if signal_side == 'long' and current_position is None:
                # 포지션 사이즈 계산
                position_value = self._calculate_position_size(price)
                position_size = position_value / price
                
                if position_size > 0:
                    # 진입 주문 실행
                    trade = self.execution_engine.execute_entry(
                        timestamp,
                        price,
                        'long',
                        position_size
                    )
                    
                    # 포트폴리오에 추가
                    if self.portfolio.open_trade(trade):
                        current_position = trade
                        self.portfolio.update_position('default', position_size, price)
        
        # 마지막 포지션 청산
        if current_position is not None:
            last_row = combined.iloc[-1]
            exit_fee = self.execution_engine.execute_exit(
                current_position,
                last_row['timestamp'],
                last_row['close']
            )
            self.portfolio.close_trade(
                current_position,
                last_row['timestamp'],
                last_row['close'],
                exit_fee
            )
        
        # 결과 저장
        self.results = self._compile_results(combined)
        
        logger.info(f"Backtest completed: {len(self.portfolio.closed_trades)} trades")
        logger.info(f"Final equity: ${self.portfolio.equity:.2f}, "
                   f"Return: {self.portfolio.total_return:.2%}")
        
        return self.results
    
    def _calculate_position_size(self, price: float) -> float:
        """포지션 사이즈 계산"""
        if self.position_sizing == 'fixed_percent':
            # 고정 비율
            position_value = self.portfolio.cash * self.risk_per_trade
        else:
            # 기본: 전체 자본
            position_value = self.portfolio.cash
        
        # 최대 포지션 제한
        max_value = self.portfolio.cash * self.max_position_size
        position_value = min(position_value, max_value)
        
        return position_value
    
    def _compile_results(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """결과 취합"""
        # 포트폴리오 통계
        portfolio_stats = self.portfolio.get_stats()
        
        # 자산 곡선
        equity_curve_df = pd.DataFrame(self.portfolio.equity_curve)
        
        # 거래 내역
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'side': t.side,
                'size': t.size,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'duration': t.duration,
            }
            for t in self.portfolio.closed_trades
        ])
        
        results = {
            'portfolio_stats': portfolio_stats,
            'equity_curve': equity_curve_df,
            'trades': trades_df,
            'market_data': market_data,
        }
        
        return results
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """백테스트 결과 조회"""
        return self.results
