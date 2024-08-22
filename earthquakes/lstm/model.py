import torch.nn as nn


# Define the LSTM model
class LSTMModel(nn.Module):
    """
    LSTM Layers are a type of RNN

    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    def __init__(self, feature_size: int, hidden_size: int, num_layers: int):
        """
        Call LSTM layers take a (seq_size, feature_size) tensor, and outputs a hidden_size tensor
        for batch processing batch_first is enabled so we would process (batch, seq_size, feature_size)


        """
        super().__init__()
        # num_layers is the number of LSTM layers stacked on top of each other
        # this would preferably be equivalent to the number of lookbacks
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        out, _ = self.lstm(batch)
        # Use the last output of the LSTM
        # out shape: (batch, lookback, hidden_size)
        # hidden size is equivalent to features in a given sequence/lookback
        # lets say x.size is batch = 1, lookback = 6, features = 4
        # out shape: (1, 6, 4) 1 batch 6 sequences 4 features,
        # the last sequence out[:, -1, :] is of shape: (1, 4) 1 batch 4 features
        return self.linear(out[:, -1, :])

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            config["feature_size"],
            config["hidden_size"],
            config["num_layers"],
        )
