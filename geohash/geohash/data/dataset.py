"""Dataset and windowing utilities for LSTM training."""

from typing import Any

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class QuakeWindowDataset(Dataset):
    """PyTorch Dataset for earthquake sequence windows."""

    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        """
        Initialize dataset.

        Parameters
        ----------
        samples : list[dict[str, torch.Tensor]]
            List of windows, each with keys: "gh_ids", "x_num", "y".
        """
        self.samples = samples

    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a window.

        Returns
        -------
        tuple
            (gh_ids, x_num, y) tensors.
        """
        sample = self.samples[idx]
        return sample["gh_ids"], sample["x_num"], sample["y"]


def make_windows(
    df: pd.DataFrame,
    stoi: dict[str, int],
    min_len: int,
    max_len: int,
    stride: int,
) -> list[dict[str, Any]]:
    """
    Create sliding windows from earthquake sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with columns: magnitude, depth_km, time_days, delta_t_days, delta_mag, delta_lat, delta_lon, geohash.
    stoi : dict[str, int]
        Geohash to integer mapping.
    min_len : int
        Minimum window length.
    max_len : int
        Maximum window length.
    stride : int
        Stride between consecutive windows.

    Returns
    -------
    list[dict[str, Any]]
        List of samples with keys: "gh_ids", "x_num", "y".

    Raises
    ------
    RuntimeError
        If no windows can be created.
    """
    samples: list[dict[str, Any]] = []
    numeric_cols = [
        "magnitude",
        "depth_km",
        "time_days",
        "delta_t_days",
        "delta_mag",
        "delta_lat",
        "delta_lon",
    ]

    for end_idx in range(min_len, len(df)):
        current_window_len = min(max_len, end_idx)
        start_idx = end_idx - current_window_len

        while start_idx <= end_idx - min_len:
            hist = df.iloc[start_idx:end_idx]
            target = df.iloc[end_idx]["magnitude"]

            # Encode geohashes as integer IDs
            gh_ids = torch.tensor(
                [stoi[g] for g in hist["geohash"].tolist()],
                dtype=torch.long
            )

            # Extract numeric features
            num_feats = torch.tensor(
                hist[numeric_cols].values,
                dtype=torch.float32
            )

            samples.append({
                "gh_ids": gh_ids,
                "x_num": num_feats,
                "y": torch.tensor([target], dtype=torch.float32),
            })

            start_idx += stride

    if not samples:
        raise RuntimeError("No training windows created. Lower min_len or widen data filters.")

    return samples


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate batch of windows with padding.

    Parameters
    ----------
    batch : list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        List of (gh_ids, x_num, y) tuples.

    Returns
    -------
    tuple
        (gh_ids_padded, x_num_padded, lengths, y_stacked).
    """
    gh_ids_list, x_num_list, y_list = zip(*batch)

    # Get sequence lengths (before padding)
    lengths = torch.tensor([len(x) for x in x_num_list], dtype=torch.long)

    # Pad sequences
    gh_ids_padded = pad_sequence(gh_ids_list, batch_first=True, padding_value=0)
    x_num_padded = pad_sequence(x_num_list, batch_first=True, padding_value=0.0)

    # Stack targets
    y_stacked = torch.stack(y_list)

    return gh_ids_padded, x_num_padded, lengths, y_stacked


def standardize_numeric(
    train_samples: list[dict[str, torch.Tensor]],
    test_samples: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize numeric features using training set statistics.

    Parameters
    ----------
    train_samples : list[dict[str, torch.Tensor]]
        Training samples.
    test_samples : list[dict[str, torch.Tensor]]
        Test samples (modified in-place).

    Returns
    -------
    tuple
        (mean, std) computed from training set.
    """
    # Compute statistics from training set
    train_all = torch.cat([s["x_num"] for s in train_samples], dim=0)
    mean = train_all.mean(dim=0)
    std = train_all.std(dim=0).clamp_min(1e-6)

    # Apply to both train and test
    for samples in (train_samples, test_samples):
        for sample in samples:
            sample["x_num"] = (sample["x_num"] - mean) / std

    return mean, std
