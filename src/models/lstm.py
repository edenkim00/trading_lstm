"""LSTM 모델 아키텍처"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM 기반 가격 예측 모델"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1
    ):
        """
        Args:
            input_size: 입력 피처 수
            hidden_size: LSTM hidden state 크기
            num_layers: LSTM 레이어 수
            dropout: Dropout 비율
            bidirectional: 양방향 LSTM 사용 여부
            output_size: 출력 크기 (1: 단일 값, 2+: 멀티 클래스)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # FC 레이어
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        
        # 출력 활성화 함수 (이진 분류)
        if output_size == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
        
        logger.info(f"Created LSTM model: input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, bidirectional={bidirectional}")
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, input_size)
            hidden: 초기 hidden state (optional)
            
        Returns:
            출력 텐서 (batch_size, output_size)
        """
        # LSTM
        if hidden is not None:
            lstm_out, _ = self.lstm(x, hidden)
        else:
            lstm_out, _ = self.lstm(x)
        
        # 마지막 시점의 출력만 사용
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # FC
        output = self.fc(dropped)
        
        # 활성화
        output = self.activation(output)
        
        # 마지막 차원만 squeeze (batch 차원 유지)
        if output.dim() > 1:
            return output.squeeze(-1)
        return output
    
    def init_hidden(self, batch_size: int, device: str = 'cpu'):
        """
        Hidden state 초기화
        
        Args:
            batch_size: 배치 크기
            device: 디바이스
            
        Returns:
            (h0, c0) 튜플
        """
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        
        return (h0, c0)


class AttentionLSTM(nn.Module):
    """Attention 메커니즘을 추가한 LSTM (향후 확장용)"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.Linear(hidden_size, 1)
        
        # FC
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            
        Returns:
            (batch_size, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Attention weights
        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),  # (batch, seq_len)
            dim=1
        )
        
        # Weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_out  # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)
        
        # FC
        output = self.fc(self.dropout(context))
        output = self.activation(output)
        
        # 마지막 차원만 squeeze (batch 차원 유지)
        if output.dim() > 1:
            return output.squeeze(-1)
        return output
