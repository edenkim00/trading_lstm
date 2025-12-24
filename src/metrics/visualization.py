"""시각화 모듈"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# 스타일 설정
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class BacktestVisualizer:
    """백테스트 결과 시각화"""
    
    @staticmethod
    def plot_equity_curve(
        equity_curve: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """자산 곡선 플롯"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(
            equity_curve['timestamp'],
            equity_curve['equity'],
            label='Equity',
            linewidth=2
        )
        
        ax.set_title('Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_drawdown(
        equity_curve: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Drawdown 플롯"""
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.fill_between(
            equity_curve['timestamp'],
            drawdown,
            0,
            alpha=0.3,
            color='red',
            label='Drawdown'
        )
        ax.plot(
            equity_curve['timestamp'],
            drawdown,
            color='darkred',
            linewidth=1.5
        )
        
        ax.set_title('Drawdown', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Y축 포맷 (백분율)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown chart saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_returns_distribution(
        trades: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """수익률 분포 플롯"""
        if trades.empty:
            logger.warning("No trades to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # P&L 분포
        ax1.hist(trades['pnl'], bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('P&L ($)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 수익률 분포
        ax2.hist(trades['return_pct'] * 100, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax2.set_title('Return Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Return (%)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns distribution saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_summary(
        results: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False
    ):
        """전체 요약 차트 생성"""
        equity_curve = results['equity_curve']
        trades = results['trades']
        
        # 자산 곡선
        save_path = f"{save_dir}/equity_curve.png" if save_dir else None
        BacktestVisualizer.plot_equity_curve(equity_curve, save_path, show)
        
        # Drawdown
        save_path = f"{save_dir}/drawdown.png" if save_dir else None
        BacktestVisualizer.plot_drawdown(equity_curve, save_path, show)
        
        # 수익률 분포
        if not trades.empty:
            save_path = f"{save_dir}/returns_distribution.png" if save_dir else None
            BacktestVisualizer.plot_returns_distribution(trades, save_path, show)
        
        logger.info("All charts generated")
