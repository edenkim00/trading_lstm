"""모델 학습 유틸리티"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelTrainer:
    """LSTM 모델 학습"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        optimizer_name: str = 'adam',
        loss_type: str = 'bce'
    ):
        """
        Args:
            model: PyTorch 모델
            device: 디바이스 (cpu, cuda)
            learning_rate: 학습률
            optimizer_name: 옵티마이저 (adam, sgd, rmsprop)
            loss_type: 손실 함수 타입 (bce, mse)
        """
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type.lower()
        
        # Optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function
        if self.loss_type == 'bce':
            self.criterion = nn.BCELoss()
        elif self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        verbose: bool = True
    ) -> Tuple[float, float]:
        """
        한 에폭 학습
        
        Args:
            train_loader: 학습 데이터로더
            verbose: 진행 상황 출력 여부
            
        Returns:
            (평균 loss, 정확도)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        iterator = tqdm(train_loader, desc="Training") if verbose else train_loader
        
        for batch_idx, (batch_X, batch_y) in enumerate(iterator):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # 분류인 경우에만 accuracy 계산
            if self.loss_type == 'bce':
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
            
            # 진행 상황 업데이트
            if verbose:
                current_loss = total_loss / (batch_idx + 1)
                if self.loss_type == 'bce':
                    current_acc = correct / total if total > 0 else 0
                    iterator.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{current_acc:.4f}'
                    })
                else:
                    iterator.set_postfix({'loss': f'{current_loss:.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if self.loss_type == 'bce' else 0.0
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        검증
        
        Args:
            val_loader: 검증 데이터로더
            
        Returns:
            (평균 loss, 정확도)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                # 분류인 경우에만 accuracy 계산
                if self.loss_type == 'bce':
                    predictions = (outputs > 0.5).float()
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if self.loss_type == 'bce' else 0.0
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0001,
        save_best_model: bool = True,
        model_save_path: str = 'models/lstm/best_model.pth',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            epochs: 최대 에폭 수
            early_stopping_patience: Early stopping patience
            early_stopping_min_delta: Early stopping 최소 개선량
            save_best_model: 최상 모델 저장 여부
            model_save_path: 모델 저장 경로
            verbose: 진행 상황 출력 여부
            
        Returns:
            학습 히스토리
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, verbose=True)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # History 기록
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if verbose:
                if self.loss_type == 'bce':
                    logger.info(
                        f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
            
            # Early stopping 체크
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 최상 모델 저장
                if save_best_model:
                    self.save_model(model_save_path)
                    logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        return self.history
    
    def save_model(self, filepath: str):
        """모델 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Model loaded from {filepath}")


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    is_regression: bool = False
) -> np.ndarray:
    """
    모델 예측
    
    Args:
        model: PyTorch 모델
        dataloader: 데이터로더
        device: 디바이스
        is_regression: 회귀 모드 여부
        
    Returns:
        predictions (분류: 0/1, 회귀: 실수값)
    """
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch_X = batch[0]
            else:
                batch_X = batch
            
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            all_outputs.append(outputs.cpu().numpy())
    
    outputs = np.concatenate(all_outputs).flatten()
    
    # 회귀면 그대로, 분류면 확률값 반환
    return outputs
