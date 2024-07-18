import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from earthquakes.data.data import EarthquakeData


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

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(qdata: EarthquakeData, size: int, test_size: float, test=False):
    sequences, targets = qdata.to_sequences(size)

    (
        train_sequences,
        test_sequences,
        train_targets,
        test_targets,
    ) = train_test_split(sequences, targets, test_size=test_size, shuffle=False)

    logger.debug("Train records(%s): %s", 1 - test_size, train_sequences.shape[0])
    logger.debug("Test records(%s): %s", test_size, train_targets.shape[0])
    logger.debug("Total: %s", sequences.shape[0])
    logger.debug("Features: %s", tuple(sequences.shape[-1]))
    logger.debug("Target: %s", qdata.target)

    if test:
        return SequencesDataset(test_sequences, test_targets)

    return SequencesDataset(train_sequences, train_targets)
