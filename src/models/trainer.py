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
        optimizer_name: str = 'adam'
    ):
        """
        Args:
            model: PyTorch 모델
            device: 디바이스 (cpu, cuda)
            learning_rate: 학습률
            optimizer_name: 옵티마이저 (adam, sgd, rmsprop)
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()
        
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
        
        for batch_X, batch_y in iterator:
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
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
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
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
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
            train_loss, train_acc = self.train_epoch(train_loader, verbose=False)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # History 기록
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if verbose:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
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
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    모델 예측
    
    Args:
        model: PyTorch 모델
        dataloader: 데이터로더
        device: 디바이스
        
    Returns:
        (predictions, probabilities)
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch_X = batch[0]
            else:
                batch_X = batch
            
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            all_probs.append(outputs.cpu().numpy())
    
    probs = np.concatenate(all_probs)
    preds = (probs > 0.5).astype(int)
    
    return preds, probs
