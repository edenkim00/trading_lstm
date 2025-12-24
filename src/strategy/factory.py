"""전략 팩토리"""
import logging
from typing import Dict, Any, Optional, Type

from .base import Strategy
from .lstm_strategy import LSTMStrategy

logger = logging.getLogger(__name__)


# 전략 레지스트리
STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {
    'lstm': LSTMStrategy,
}


def create_strategy(
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> Strategy:
    """
    전략 팩토리 함수
    
    Args:
        name: 전략 이름
        config: 전략 설정
        
    Returns:
        Strategy 인스턴스
    """
    if name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: {name}. Available strategies: {available}"
        )
    
    strategy_class = STRATEGY_REGISTRY[name]
    strategy = strategy_class(config)
    
    logger.info(f"Created strategy: {name}")
    
    return strategy


def register_strategy(name: str, strategy_class: Type[Strategy]) -> None:
    """
    새로운 전략 등록
    
    Args:
        name: 전략 이름
        strategy_class: Strategy 클래스
    """
    if not issubclass(strategy_class, Strategy):
        raise TypeError(f"{strategy_class} must inherit from Strategy")
    
    STRATEGY_REGISTRY[name] = strategy_class
    logger.info(f"Registered strategy: {name}")


def list_strategies() -> list:
    """등록된 전략 목록 조회"""
    return list(STRATEGY_REGISTRY.keys())
