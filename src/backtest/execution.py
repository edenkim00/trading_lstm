"""주문 실행 시뮬레이션"""
import logging
from typing import Optional
from datetime import datetime

from .models import Trade, FeeModel, SlippageModel

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """주문 체결 시뮬레이터"""
    
    def __init__(
        self,
        fee_model: Optional[FeeModel] = None,
        slippage_model: Optional[SlippageModel] = None
    ):
        """
        Args:
            fee_model: 수수료 모델
            slippage_model: 슬리피지 모델
        """
        self.fee_model = fee_model or FeeModel()
        self.slippage_model = slippage_model or SlippageModel()
    
    def execute_entry(
        self,
        timestamp: datetime,
        price: float,
        side: str,
        size: float
    ) -> Trade:
        """
        진입 주문 실행
        
        Args:
            timestamp: 시간
            price: 가격
            side: 'long' or 'short'
            size: 포지션 크기
            
        Returns:
            Trade 객체
        """
        # 슬리피지 적용
        execution_price = self.slippage_model.calculate(
            price,
            side='buy' if side == 'long' else 'sell'
        )
        
        slippage_cost = abs(execution_price - price) * size
        
        # 수수료 계산
        fee = self.fee_model.calculate(execution_price, size)
        
        # Trade 생성
        trade = Trade(
            entry_time=timestamp,
            entry_price=execution_price,
            side=side,
            size=size,
            entry_fee=fee,
            slippage=slippage_cost
        )
        
        logger.debug(
            f"Entry: {side} {size} @ {execution_price:.2f} "
            f"(fee: {fee:.2f}, slippage: {slippage_cost:.4f})"
        )
        
        return trade
    
    def execute_exit(
        self,
        trade: Trade,
        timestamp: datetime,
        price: float
    ) -> float:
        """
        청산 주문 실행
        
        Args:
            trade: 청산할 거래
            timestamp: 청산 시간
            price: 청산 가격
            
        Returns:
            청산 수수료
        """
        # 슬리피지 적용
        execution_price = self.slippage_model.calculate(
            price,
            side='sell' if trade.side == 'long' else 'buy'
        )
        
        # 수수료 계산
        fee = self.fee_model.calculate(execution_price, trade.size)
        
        logger.debug(
            f"Exit: {trade.side} {trade.size} @ {execution_price:.2f} "
            f"(fee: {fee:.2f})"
        )
        
        return fee
