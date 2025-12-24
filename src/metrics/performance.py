"""성과 지표 계산"""
import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """백테스트 성과 지표 계산"""
    
    @staticmethod
    def calculate_all(
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        모든 성과 지표 계산
        
        Args:
            equity_curve: 자산 곡선 데이터프레임
            trades: 거래 내역 데이터프레임
            initial_capital: 초기 자본
            risk_free_rate: 무위험 수익률 (연율)
            
        Returns:
            성과 지표 딕셔너리
        """
        metrics = {}
        
        # 기본 통계
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        metrics['initial_capital'] = initial_capital
        metrics['final_equity'] = final_equity
        metrics['total_return'] = total_return
        metrics['total_pnl'] = final_equity - initial_capital
        
        # 수익률 시계열
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Sharpe Ratio
        metrics['sharpe_ratio'] = PerformanceMetrics.sharpe_ratio(
            returns, risk_free_rate
        )
        
        # Sortino Ratio
        metrics['sortino_ratio'] = PerformanceMetrics.sortino_ratio(
            returns, risk_free_rate
        )
        
        # Drawdown
        dd_metrics = PerformanceMetrics.drawdown_metrics(equity_curve['equity'])
        metrics.update(dd_metrics)
        
        # 거래 통계
        if not trades.empty:
            trade_metrics = PerformanceMetrics.trade_metrics(trades)
            metrics.update(trade_metrics)
        
        # Calmar Ratio
        if metrics.get('max_drawdown', 0) != 0:
            metrics['calmar_ratio'] = total_return / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0.0
        
        return metrics
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365
    ) -> float:
        """
        Sharpe Ratio 계산
        
        Args:
            returns: 수익률 시계열
            risk_free_rate: 무위험 수익률
            periods_per_year: 연간 기간 수
            
        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        
        return sharpe
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365
    ) -> float:
        """
        Sortino Ratio 계산 (하방 리스크만 고려)
        
        Args:
            returns: 수익률 시계열
            risk_free_rate: 무위험 수익률
            periods_per_year: 연간 기간 수
            
        Returns:
            Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        
        # 하방 편차
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
        
        return sortino
    
    @staticmethod
    def drawdown_metrics(equity: pd.Series) -> Dict[str, float]:
        """
        Drawdown 지표 계산
        
        Args:
            equity: 자산 시계열
            
        Returns:
            Drawdown 관련 지표
        """
        # Running maximum
        running_max = equity.expanding().max()
        
        # Drawdown
        drawdown = (equity - running_max) / running_max
        
        # Max drawdown
        max_dd = drawdown.min()
        
        # Max drawdown duration
        in_drawdown = drawdown < 0
        dd_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    dd_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            dd_periods.append(current_period)
        
        max_dd_duration = max(dd_periods) if dd_periods else 0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_dd_duration,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
        }
    
    @staticmethod
    def trade_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
        """
        거래 관련 지표 계산
        
        Args:
            trades: 거래 내역 데이터프레임
            
        Returns:
            거래 관련 지표
        """
        total_trades = len(trades)
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade_duration': 0.0
            }
        
        # 승/패 구분
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        # Win rate
        win_rate = len(winning_trades) / total_trades
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # 평균 손익
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
        
        # 거래 기간
        avg_duration = trades['duration'].mean() if 'duration' in trades.columns else 0
        
        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            'avg_trade_duration_seconds': avg_duration,
            'expectancy': expectancy
        }
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """지표를 읽기 쉽게 포맷팅"""
        lines = [
            "=" * 60,
            "백테스트 성과 리포트",
            "=" * 60,
            "",
            "## 자본 및 수익",
            f"초기 자본:        ${metrics['initial_capital']:,.2f}",
            f"최종 자본:        ${metrics['final_equity']:,.2f}",
            f"총 수익:          ${metrics['total_pnl']:,.2f}",
            f"총 수익률:        {metrics['total_return']:.2%}",
            "",
            "## 리스크 조정 수익률",
            f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio:    {metrics.get('sortino_ratio', 0):.2f}",
            f"Calmar Ratio:     {metrics.get('calmar_ratio', 0):.2f}",
            "",
            "## Drawdown",
            f"최대 Drawdown:    {metrics.get('max_drawdown', 0):.2%}",
            f"평균 Drawdown:    {metrics.get('avg_drawdown', 0):.2%}",
            f"최대 DD 기간:     {metrics.get('max_drawdown_duration', 0)} 바",
            "",
            "## 거래 통계",
            f"총 거래:          {metrics.get('total_trades', 0)}",
            f"승률:             {metrics.get('win_rate', 0):.2%}",
            f"Profit Factor:    {metrics.get('profit_factor', 0):.2f}",
            f"평균 수익:        ${metrics.get('avg_win', 0):,.2f}",
            f"평균 손실:        ${metrics.get('avg_loss', 0):,.2f}",
            f"최대 수익:        ${metrics.get('largest_win', 0):,.2f}",
            f"최대 손실:        ${metrics.get('largest_loss', 0):,.2f}",
            f"Expectancy:       ${metrics.get('expectancy', 0):,.2f}",
            "=" * 60,
        ]
        
        return "\n".join(lines)
