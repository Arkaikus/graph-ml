"""LSTM model for earthquake magnitude prediction."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GeohashEncoder(nn.Module):
    """Flat cell-level or hierarchical char-level geohash encoder."""

    def __init__(
        self,
        encoding: str,
        vocab_size: int,
        embedding_dim: int,
        geohash_precision: int = 4,
        char_vocab_size: int = 34,
    ):
        super().__init__()
        self.encoding = encoding
        self.geohash_precision = geohash_precision

        if encoding == "flat":
            self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.char_embed = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)

    def forward(self, gh_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        gh_ids : torch.Tensor
            flat: (batch, seq_len)
            hierarchical: (batch, seq_len, precision)

        Returns
        -------
        torch.Tensor
            (batch, seq_len, embedding_dim)
        """
        if self.encoding == "flat":
            return self.embed(gh_ids)

        # Char embeddings summed per geohash cell
        char_emb = self.char_embed(gh_ids)  # (B, S, P, D)
        return char_emb.sum(dim=2)


class NextMagnitudeLSTM(nn.Module):
    """LSTM for predicting next earthquake magnitude."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_numeric: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_mode: str = "full",
        encoding: str = "flat",
        geohash_precision: int = 4,
        char_vocab_size: int = 34,
    ):
        super().__init__()
        self.input_mode = input_mode
        self.encoding = encoding
        self.embedding_dim = embedding_dim
        self.num_numeric = num_numeric

        self.geohash_encoder = GeohashEncoder(
            encoding=encoding,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            geohash_precision=geohash_precision,
            char_vocab_size=char_vocab_size,
        )

        input_size = embedding_dim + num_numeric
        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
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
        gh_emb = self.geohash_encoder(gh_ids)

        if self.input_mode == "numeric_only":
            gh_emb = torch.zeros_like(gh_emb)
        elif self.input_mode == "geohash_only":
            x_num = torch.zeros_like(x_num)

        x = torch.cat([gh_emb, x_num], dim=-1)
        x = self.input_norm(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        return self.head(last_hidden)
