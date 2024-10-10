import tensorflow as tf


# Define the LSTM model
class LSTMModel(tf.keras.Model):
    """
    LSTM Layers are a type of RNN

    https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
    """

    def __init__(self, hidden_size: int, num_layers):
        """
        Call LSTM layers take a (seq_size, feature_size) tensor, and outputs a hidden_size tensor
        for batch processing batch_first is enabled so we would process (batch, seq_size, feature_size)
        """
        super().__init__()
        # num_layers is the number of LSTM layers stacked on top of each other
        # this would preferably be equivalent to the number of lookbacks
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

        # self.dense_hidden = []
        # for i in range(num_layers):
        #     if hidden_size // (2**i) < 1:
        #         break
        #     self.dense_hidden.append(tf.keras.layers.Dense(hidden_size // (2**i), activation="relu"))
        # self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(1)

    def call(self, inputs):
        out = self.lstm(inputs)
        # Use the last output of the LSTM
        # out shape: (batch, lookback, hidden_size)
        # hidden size is equivalent to features in a given sequence/lookback
        # lets say x.size is batch = 1, lookback = 6, features = 4
        # out shape: (1, 6, 4) 1 batch 6 sequences 4 features,
        # the last sequence out[:, -1, :] is of shape: (1, 4) 1 batch 4 features
        # the last sequence out[:, -1, :] is of shape: (2, 4) 2 batch 4 features
        # import pdb; pdb.set_trace()
        return self.linear(out[:, -1, :])

    @classmethod
    def from_config(cls, config: dict):
        print()
        print(config)
        print()
        return cls(
            # config["feature_size"],
            config["hidden_size"],
            config["num_layers"],
        )
