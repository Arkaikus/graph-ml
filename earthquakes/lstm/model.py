"""Base LSTM model for sequence-to-output prediction."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SequenceAttention(nn.Module):
    """Attention over LSTM sequence output to weight timesteps by relevance."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size)
        weights = F.softmax(self.attention(lstm_out), dim=1)
        return (weights * lstm_out).sum(dim=1)


class BaseLSTMModel(nn.Module):
    """
    Base LSTM model for sequence-to-output prediction.

    Input shape: (batch, seq_len, input_size)
    - seq_len: number of features (each feature is a time series of length lookback)
    - input_size: lookback (length of each feature's history)

    Output shape: (batch, outputs)

    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    def __init__(
        self,
        lookback: int,
        outputs: int,
        hidden_size: int,
        num_layers: int,
        *,
        dropout: float = 0.0,
        num_features: Optional[int] = None,
        use_attention: bool = False,
    ):
        """
        Args:
            lookback: Length of each feature's history (LSTM input_size).
            outputs: Number of output units (classes or regression targets).
            hidden_size: LSTM hidden dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability (applied between LSTM layers if num_layers > 1).
            num_features: Number of feature sequences (seq_len). If provided, uses
                explicit Linear for reproducibility; otherwise uses LazyLinear.
        """
        super().__init__()
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if outputs < 1:
            raise ValueError("outputs must be >= 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.lookback = lookback
        self.outputs = outputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_features = num_features
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            lookback,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if use_attention:
            self.attention = SequenceAttention(hidden_size)
            self.linear = nn.Linear(hidden_size, outputs)
        else:
            self.attention = None
            self.flatten = nn.Flatten()
            if num_features is not None:
                self.linear = nn.Linear(num_features * hidden_size, outputs)
            else:
                self.linear = nn.LazyLinear(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, lookback)

        Returns:
            (batch, outputs)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, lookback), got {x.dim()}D")
        if x.size(2) != self.lookback:
            raise ValueError(
                f"Input last dim {x.size(2)} != lookback {self.lookback}"
            )

        out, _ = self.lstm(x)
        if self.attention is not None:
            agg = self.attention(out)
        else:
            agg = self.flatten(out)
        return self.linear(agg)

    @classmethod
    def from_config(cls, config: dict) -> BaseLSTMModel:
        return cls(
            lookback=config["lookback"],
            outputs=config["outputs"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config.get("dropout", 0.0),
            num_features=config.get("num_features"),
            use_attention=config.get("use_attention", False),
        )


# Backward compatibility alias
LSTMModel = BaseLSTMModel
