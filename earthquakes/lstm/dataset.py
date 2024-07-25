import logging

import numpy as np
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class SequencesDataset(Dataset):
    """
    This class represents a dataset of sequences that later will be consumed by torch.Dataloader

    __getitem__(self, idx) will retrieve the batched sequences as (batchidx, seq_lenght, feature_size)
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = sequences
        self.targets = targets
        logger.debug("Shape: (batch, seq_length, feature_size)")
        logger.debug("Input shape: %s, target shape: %s", tuple(self.sequences.shape), tuple(self.targets.shape))

    @property
    def feature_size(self):
        return self.sequences.shape[2]

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
