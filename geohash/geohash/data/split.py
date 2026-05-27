"""Temporal event-level train/val/test splitting."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_events_temporal(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split events by time_ms cutoff; no event appears in more than one split.

    Parameters
    ----------
    df : pd.DataFrame
        Event catalog with time_ms column.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.

    Returns
    -------
    tuple
        (train_df, val_df, test_df) each reset_index.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    df = df.sort_values("time_ms").reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise RuntimeError(f"Need at least 3 events for train/val/test split, got {n}")

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    if len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "Split produced empty val or test set. Widen data or adjust ratios."
        )

    logger.info(
        "Temporal split: train=%d, val=%d, test=%d events",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def split_windows_index(
    samples: list,
    train_ratio: float = 0.8,
) -> tuple[list, list]:
    """
    Legacy window-index split (debug/backward compat only).

    WARNING: leaks overlapping contexts when stride=1.
    """
    split_idx = int(len(samples) * train_ratio)
    split_idx = max(1, min(split_idx, len(samples) - 1))
    return samples[:split_idx], samples[split_idx:]
