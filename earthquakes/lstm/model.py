import torch.nn as nn


# Define the LSTM model
class LSTMModel(nn.Module):
    """
    LSTM Layers are a type of RNN

    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    def __init__(
        self,
        lookback: int,
        outputs: int,
        hidden_size: int,
        num_layers: int,
    ):
        """
        Call LSTM layers take a (seq_size, lookback) tensor, and outputs a hidden_size tensor
        for batch processing batch_first is enabled so we would process (batch, seq_size, lookback)


        """
        super().__init__()
        # num_layers is the number of LSTM layers stacked on top of each other
        # this would preferably be equivalent to the number of lookbacks
        self.lstm = nn.LSTM(lookback, hidden_size, num_layers, batch_first=True)
        self.flatten = nn.Flatten()  # flatten the output of the LSTM layer
        # self.dense = nn.LazyLinear(lookback * hidden_size)
        # self.dense1 = nn.LazyLinear(lookback)
        self.linear = nn.LazyLinear(outputs)

    def forward(self, batch_input):
        # batch input is (batch_size, sequence, lookback)
        # where: lookback is the length i.e number of features in a sequence
        #        sequence is feature sequence, i.e seq 1 = [latitude t0, latitude t1, ..., latitude tN]
        out, _ = self.lstm(batch_input)

        # this will turn (batch_size, sequence, hidden_size) into (batch_size, sequence * hidden_size)
        # leaving a W*x + b operation to be done by the linear layer with all of the hidden states
        flatten = self.flatten(out)
        dense = nn.functional.sigmoid(self.dense(flatten))
        dense = nn.functional.sigmoid(self.dense1(dense))
        # return is (batch_size, 1)
        return self.linear(dense)

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            config["lookback"],
            config["hidden_size"],
            config["num_layers"],
        )
