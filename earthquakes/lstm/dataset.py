import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MovingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target: str, seq_length, device):
        assert target, "[target] cannot be None"
        assert seq_length, "[seq_length] cannot be None"
        self.inputs, self.targets = self.create_sequences(df, target, seq_length)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32).to(device)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).to(device)
        logger.info("Shape: (batch, seq_length, feature_size)")
        logger.info("Input shape: %s, target shape: %s", tuple(self.inputs.shape), tuple(self.targets.shape))

    def create_sequences(self, df: pd.DataFrame, target: str, seq_length):
        inputs, targets = [], []
        assert df.shape[0] - seq_length > 2, "sequence can't be less than available data"
        for i in range(df.shape[0] - seq_length):
            input_seq = df.iloc[i : i + seq_length].values
            target_v = df.at[i + seq_length, target]
            inputs.append(input_seq)
            targets.append(target_v)
        return np.array(inputs), np.array(targets).reshape(-1, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
