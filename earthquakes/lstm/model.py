import torch.nn as nn


# Define the LSTM model
class LSTMModel(nn.Module):
    """
    LSTM Layers are a type of RNN

    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    def __init__(self, feature_size: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Call LSTM layers take a (seq_size, feature_size) tensor, and outputs a hidden_size tensor
        for batch processing batch_first is enabled so we would process (batch, seq_size, feature_size)
        """
        super().__init__()
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        out, _ = self.lstm(batch)
        out = out[:, -1, :]  # Use the last output of the LSTM
        return self.linear(out)

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            config["feature_size"],
            config["hidden_size"],
            config["num_layers"],
            config["dropout"],
        )
