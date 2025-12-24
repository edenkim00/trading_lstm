"""LSTM 기반 트레이딩 전략"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import Strategy
from ..models import LSTMModel, ModelTrainer, predict
from ..preprocessing import SequenceGenerator, TimeSeriesDataset

logger = logging.getLogger(__name__)


class LSTMStrategy(Strategy):
    """LSTM 기반 방향성 예측 전략"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 전략 설정
                - model: 모델 아키텍처 설정
                - signal_rules: 신호 생성 규칙
                - sequence_length: 시퀀스 길이
                - feature_columns: 사용할 피처 컬럼 리스트
        """
        super().__init__(config)
        
        # 모델 설정
        model_config = self.config.get('model', {})
        self.input_size = model_config.get('input_size')
        self.hidden_size = model_config.get('hidden_size', 128)
        self.num_layers = model_config.get('num_layers', 2)
        self.dropout = model_config.get('dropout', 0.2)
        self.bidirectional = model_config.get('bidirectional', False)
        
        # 신호 규칙
        signal_rules = self.config.get('signal_rules', {})
        self.long_threshold = signal_rules.get('long_threshold', 0.55)
        self.short_threshold = signal_rules.get('short_threshold', 0.45)
        self.confidence_filter = signal_rules.get('confidence_filter', 0.0)
        
        # 예측 타입 (분류 vs 회귀)
        prediction_config = self.config.get('prediction', {})
        self.target_type = prediction_config.get('target', 'direction')  # direction or return
        self.is_regression = (self.target_type == 'return')
        
        # 시퀀스 설정
        self.sequence_length = self.config.get('warmup_period', 100)
        self.feature_columns = self.config.get('feature_columns', None)
        
        # 모델 (나중에 초기화)
        self.model: Optional[LSTMModel] = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 시퀀스 생성기
        self.sequence_generator = SequenceGenerator(
            sequence_length=self.sequence_length,
            prediction_horizon=1,
            stride=1
        )
        
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'target',
        training_config: Optional[Dict[str, Any]] = None
    ) -> 'LSTMStrategy':
        """
        LSTM 모델 학습
        
        Args:
            train_df: 학습 데이터 (스케일링 완료)
            val_df: 검증 데이터 (스케일링 완료)
            feature_columns: 입력 피처 컬럼 리스트
            target_column: 타겟 컬럼명
            training_config: 학습 설정
            
        Returns:
            self
        """
        self.feature_columns = feature_columns
        self.input_size = len(feature_columns)
        
        logger.info(f"Training LSTM with {self.input_size} features")
        
        # 시퀀스 생성
        X_train, y_train, _ = self.sequence_generator.create_sequences(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            include_timestamp=False
        )
        
        X_val, y_val, _ = self.sequence_generator.create_sequences(
            val_df,
            feature_columns=feature_columns,
            target_column=target_column,
            include_timestamp=False
        )
        
        logger.info(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")
        
        # 데이터셋 생성
        train_dataset = TimeSeriesDataset(X_train, y_train, device=self.device)
        val_dataset = TimeSeriesDataset(X_val, y_val, device=self.device)
        
        # 데이터로더
        training_cfg = training_config or self.config.get('training', {})
        batch_size = training_cfg.get('batch_size', 64)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 모델 생성
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            output_size=1
        )
        
        # 학습
        trainer = ModelTrainer(
            model=self.model,
            device=self.device,
            learning_rate=training_cfg.get('learning_rate', 0.001),
            optimizer_name=training_cfg.get('optimizer', 'adam'),
            loss_type=training_cfg.get('loss', 'bce')
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_cfg.get('epochs', 100),
            early_stopping_patience=training_cfg.get('early_stopping_patience', 10),
            early_stopping_min_delta=training_cfg.get('early_stopping_min_delta', 0.0001),
            save_best_model=True,
            model_save_path=training_cfg.get('model_save_path', 'models/lstm/best_model.pth'),
            verbose=True
        )
        
        self.is_fitted = True
        logger.info("LSTM training completed")
        
        return self
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        트레이딩 신호 생성
        
        Args:
            market_data: 전처리 완료된 시장 데이터
            
        Returns:
            신호 데이터프레임
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first or load a trained model.")
        
        if self.feature_columns is None:
            raise ValueError("feature_columns not set")
        
        # 시퀀스 생성 (타겟 없이)
        X, _, timestamps = self.sequence_generator.create_sequences(
            market_data,
            feature_columns=self.feature_columns,
            target_column='_dummy',  # 타겟 없음
            include_timestamp=True
        )
        
        # 데이터셋 및 로더
        dataset = TimeSeriesDataset(X, device=self.device)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # 예측
        predictions = predict(self.model, dataloader, self.device, is_regression=self.is_regression)
        
        # 신호 생성
        signals = []
        
        for i, (ts, pred) in enumerate(zip(timestamps, predictions)):
            # 포지션 결정
            if self.is_regression:
                # 회귀: 예상 수익률 기반
                expected_return = float(pred)
                if expected_return > self.long_threshold:
                    side = 'long'
                    confidence = abs(expected_return)
                elif expected_return < self.short_threshold:
                    side = 'short'
                    confidence = abs(expected_return)
                else:
                    side = 'flat'
                    confidence = abs(expected_return)
            else:
                # 분류: 확률 기반
                prob = float(pred)
                if prob > self.long_threshold:
                    side = 'long'
                    confidence = prob
                elif prob < self.short_threshold:
                    side = 'short'
                    confidence = 1 - prob
                else:
                    side = 'flat'
                    confidence = 0.5
            
            # 최소 신뢰도 필터
            if confidence < self.confidence_filter:
                side = 'flat'
            
            signals.append({
                'timestamp': ts,
                'side': side,
                'size': 1.0 if side != 'flat' else 0.0,
                'confidence': confidence,
                'prediction': float(pred)
            })
        
        signals_df = pd.DataFrame(signals)
        
        logger.info(f"Generated {len(signals_df)} signals")
        logger.info(f"Long: {(signals_df['side'] == 'long').sum()}, "
                   f"Short: {(signals_df['side'] == 'short').sum()}, "
                   f"Flat: {(signals_df['side'] == 'flat').sum()}")
        
        return signals_df
    
    @property
    def warmup_period(self) -> int:
        """시퀀스 길이만큼 warmup 필요"""
        return self.sequence_length
    
    def save(self, filepath: str):
        """전략 저장"""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 모델 가중치 + 설정 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_columns': self.feature_columns,
            'input_size': self.input_size,
        }, filepath)
        
        logger.info(f"Strategy saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = None) -> 'LSTMStrategy':
        """저장된 전략 로드"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # 전략 생성
        config = checkpoint['config']
        strategy = cls(config)
        strategy.feature_columns = checkpoint['feature_columns']
        strategy.input_size = checkpoint['input_size']
        strategy.device = device
        
        # 모델 생성 및 가중치 로드
        strategy.model = LSTMModel(
            input_size=strategy.input_size,
            hidden_size=strategy.hidden_size,
            num_layers=strategy.num_layers,
            dropout=strategy.dropout,
            bidirectional=strategy.bidirectional,
            output_size=1
        )
        strategy.model.load_state_dict(checkpoint['model_state_dict'])
        strategy.model.to(device)
        strategy.model.eval()
        
        strategy.is_fitted = True
        
        logger.info(f"Strategy loaded from {filepath}")
        
        return strategy
