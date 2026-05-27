"""Dataset and windowing utilities for LSTM training."""

import logging
from collections import Counter
from typing import Any

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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


_NUMERIC_COLS = [
    "magnitude",
    "depth_km",
    "time_days",
    "delta_t_days",
    "delta_mag",
    "delta_lat",
    "delta_lon",
]


def _validate_windows(
    samples: list[dict[str, Any]],
    stoi: dict[str, int],
    min_len: int,
    max_len: int,
    stride: int,
) -> None:
    """
    Run sanity checks on generated windows and log diagnostic statistics.

    Parameters
    ----------
    samples : list[dict[str, Any]]
        Windows produced by make_windows.
    stoi : dict[str, int]
        Geohash-to-integer vocab used during window creation.
    min_len, max_len, stride : int
        Window parameters forwarded for overlap calculation.

    Raises
    ------
    RuntimeError
        If NaN/Inf values are detected in features or targets, or if any
        geohash ID falls outside the vocabulary range.
    """
    n = len(samples)
    vocab_size = len(stoi)

    # --- Window length distribution ---
    length_counts: Counter = Counter(len(s["x_num"]) for s in samples)
    logger.info("Window length distribution: %s", dict(sorted(length_counts.items())))
    min_len_count = length_counts.get(min_len, 0)
    if n > 0 and (min_len_count / n) > 0.8:
        logger.warning(
            "%.1f%% of windows are at min_len=%d — data may be too sparse.",
            100.0 * min_len_count / n,
            min_len,
        )

    # --- Overlap ratio ---
    if max_len > 0:
        overlap_pct = max(0.0, (max_len - stride) / max_len) * 100.0
        logger.info(
            "Window overlap: %.1f%% (stride=%d, max_len=%d)",
            overlap_pct,
            stride,
            max_len,
        )

    # --- NaN / Inf detection in features and targets ---
    for i, s in enumerate(samples):
        x = s["x_num"]
        if torch.isnan(x).any() or torch.isinf(x).any():
            bad_cols = [
                _NUMERIC_COLS[c]
                for c in range(x.shape[1])
                if torch.isnan(x[:, c]).any() or torch.isinf(x[:, c]).any()
            ]
            raise RuntimeError(
                f"NaN/Inf detected in x_num at sample index {i}, "
                f"columns: {bad_cols}"
            )
        y = s["y"]
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError(
                f"NaN/Inf detected in target y at sample index {i} (value={y.item()!r})"
            )

    # --- Geohash vocabulary coverage ---
    oov_samples = [
        i
        for i, s in enumerate(samples)
        if (s["gh_ids"] < 0).any() or (s["gh_ids"] >= vocab_size).any()
    ]
    if oov_samples:
        raise RuntimeError(
            f"Out-of-vocabulary geohash IDs found in {len(oov_samples)} samples "
            f"(first offender: sample index {oov_samples[0]}). "
            f"Vocabulary size is {vocab_size}."
        )

    # --- Target magnitude statistics ---
    all_targets = torch.cat([s["y"] for s in samples]).squeeze()
    t_min = all_targets.min().item()
    t_max = all_targets.max().item()
    t_mean = all_targets.mean().item()
    t_std = all_targets.std().item()
    q25 = all_targets.quantile(0.25).item()
    q75 = all_targets.quantile(0.75).item()
    logger.info(
        "Target magnitude: min=%.3f, max=%.3f, mean=%.3f, std=%.3f, Q25=%.3f, Q75=%.3f",
        t_min,
        t_max,
        t_mean,
        t_std,
        q25,
        q75,
    )
    if t_std < 0.1:
        logger.warning(
            "Target std=%.4f is very low — nearly constant targets may hinder learning.",
            t_std,
        )

    # --- Feature value ranges ---
    all_x = torch.cat([s["x_num"] for s in samples], dim=0)
    logger.info("Feature ranges (across all windows):")
    for col_idx, col_name in enumerate(_NUMERIC_COLS):
        col = all_x[:, col_idx]
        logger.info(
            "  %-14s min=%9.4f  max=%9.4f  mean=%9.4f",
            col_name,
            col.min().item(),
            col.max().item(),
            col.mean().item(),
        )


def make_windows(
    df: pd.DataFrame,
    stoi: dict[str, int],
    min_len: int,
    max_len: int,
    stride: int,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """
    Create sliding windows from earthquake sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with columns: magnitude, depth_km, time_days,
        delta_t_days, delta_mag, delta_lat, delta_lon, geohash.
    stoi : dict[str, int]
        Geohash to integer mapping.
    min_len : int
        Minimum window length.
    max_len : int
        Maximum window length.
    stride : int
        Stride between consecutive windows.
    validate : bool
        When True (default), run sanity checks and log diagnostics after
        window creation. Set to False to skip for speed.

    Returns
    -------
    list[dict[str, Any]]
        List of samples with keys: "gh_ids", "x_num", "y".

    Raises
    ------
    RuntimeError
        If no windows can be created, or if validation detects NaN/Inf
        values or out-of-vocabulary geohash IDs.
    """
    samples: list[dict[str, Any]] = []

    for end_idx in range(min_len, len(df)):
        current_window_len = min(max_len, end_idx)
        start_idx = end_idx - current_window_len

        while start_idx <= end_idx - min_len:
            hist = df.iloc[start_idx:end_idx]
            target = df.iloc[end_idx]["magnitude"]

            gh_ids = torch.tensor(
                [stoi[g] for g in hist["geohash"].tolist()],
                dtype=torch.long,
            )

            num_feats = torch.tensor(
                hist[_NUMERIC_COLS].values,
                dtype=torch.float32,
            )

            samples.append({
                "gh_ids": gh_ids,
                "x_num": num_feats,
                "y": torch.tensor([target], dtype=torch.float32),
            })

            start_idx += stride

    if not samples:
        raise RuntimeError(
            "No training windows created. Lower min_len or widen data filters."
        )

    logger.info("Created %d windows from %d events.", len(samples), len(df))

    if validate:
        _validate_windows(samples, stoi, min_len, max_len, stride)

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
