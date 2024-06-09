import pandas as pd
import torch
from torch.utils.data import Dataset


class MovingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target: str, window_size):
        self.inputs, self.targets = self.create_sequences(df, target, window_size)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def create_sequences(self, df: pd.DataFrame, target: str, window_size):
        inputs, targets = [], []
        for i in range(df.shape[0] - window_size):
            input_seq = df.at[i : i + window_size].values
            target = df.at[i + window_size, target].values
            inputs.append(input_seq)
            targets.append(target)
        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
