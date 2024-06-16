import torch.nn as nn


# Define the LSTM model
class LSTMModel(nn.Module):
    """
    LSTM Layers are a type of RNN

    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        """
        Call LSTM layers take a (seq_size, feature_size) tensor, and outputs a hidden_size tensor
        for batch processing batch_first is enabled so we would process (batch, seq_size, feature_size)
        """
        super().__init__()
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out
