"""LSTM model for earthquake magnitude prediction."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class NextMagnitudeLSTM(nn.Module):
    """
    LSTM for predicting next earthquake magnitude.

    Combines:
    - Geohash embedding: spatial location as sequence tokens
    - Numerical features: magnitude, depth, time deltas, location deltas
    - LSTM layer: sequence modeling
    - MLP head: final prediction
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_numeric: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize model.

        Parameters
        ----------
        vocab_size : int
            Size of geohash vocabulary (including PAD).
        embedding_dim : int
            Dimension of geohash embeddings.
        num_numeric : int
            Number of numeric input features.
        hidden_size : int
            LSTM hidden dimension.
        num_layers : int
            Number of LSTM layers. Default 1.
        dropout : float
            Dropout rate (only applied if num_layers > 1). Default 0.0.
        """
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + num_numeric,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        gh_ids: torch.Tensor,
        x_num: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        gh_ids : torch.Tensor
            Shape (batch_size, seq_len) — geohash token IDs.
        x_num : torch.Tensor
            Shape (batch_size, seq_len, num_numeric) — numeric features.
        lengths : torch.Tensor
            Shape (batch_size,) — actual sequence lengths (before padding).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 1) — predicted magnitude.
        """
        # Embed geohashes
        gh_emb = self.embed(gh_ids)  # (batch_size, seq_len, embedding_dim)

        # Concatenate embeddings and numeric features
        x = torch.cat([gh_emb, x_num], dim=-1)  # (batch_size, seq_len, embedding_dim + num_numeric)

        # Pack padded sequence for efficient LSTM processing
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM: output last hidden state
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers, batch_size, hidden_size)

        # Use last layer's hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_size)

        # Predict magnitude
        pred = self.head(last_hidden)  # (batch_size, 1)

        return pred
