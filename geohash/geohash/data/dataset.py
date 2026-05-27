"""Dataset and windowing utilities for LSTM training."""

import logging
from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from geohash.data.features import (
    char_ids_for_sequence,
    compute_window_features,
    geohash_to_id,
)

logger = logging.getLogger(__name__)

_STATIC_COLS = ["magnitude", "depth_km", "time_days"]
_DELTA_COLS = ["delta_t_days", "delta_mag", "delta_distance_km"]
_NUMERIC_COLS = _STATIC_COLS + _DELTA_COLS


class QuakeWindowDataset(Dataset):
    """PyTorch Dataset for earthquake sequence windows."""

    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return sample["gh_ids"], sample["x_num"], sample["y"]


def _build_x_num(hist: pd.DataFrame) -> torch.Tensor:
    """Build numeric feature matrix from window history (per-window deltas)."""
    hist = hist.sort_values("time_ms")
    static = hist[_STATIC_COLS].values.astype(np.float32)
    deltas = compute_window_features(hist)
    x = np.column_stack([static, deltas])
    return torch.tensor(x, dtype=torch.float32)


def _build_gh_ids(
    geohashes: list[str],
    stoi: dict[str, int],
    encoding: Literal["flat", "hierarchical"],
    precision: int,
    char_stoi: dict[str, int] | None,
) -> torch.Tensor:
    if encoding == "hierarchical":
        if char_stoi is None:
            raise ValueError("char_stoi required for hierarchical encoding")
        char_seqs = char_ids_for_sequence(geohashes, precision, char_stoi)
        return torch.tensor(char_seqs, dtype=torch.long)

    ids = [geohash_to_id(g, stoi) for g in geohashes]
    return torch.tensor(ids, dtype=torch.long)


def _make_sample_dict(
    hist: pd.DataFrame,
    target: float,
    stoi: dict[str, int],
    encoding: Literal["flat", "hierarchical"],
    precision: int,
    char_stoi: dict[str, int] | None,
    target_time_ms: int | None = None,
    target_idx: int | None = None,
) -> dict[str, Any]:
    hist = hist.sort_values("time_ms")
    gh_ids = _build_gh_ids(
        hist["geohash"].tolist(),
        stoi,
        encoding,
        precision,
        char_stoi,
    )
    sample: dict[str, Any] = {
        "gh_ids": gh_ids,
        "x_num": _build_x_num(hist),
        "y": torch.tensor([target], dtype=torch.float32),
    }
    if target_time_ms is not None:
        sample["target_time_ms"] = target_time_ms
    if target_idx is not None:
        sample["target_idx"] = target_idx
    return sample


def _validate_windows(
    samples: list[dict[str, Any]],
    stoi: dict[str, int],
    min_len: int,
    max_len: int,
    stride: int,
) -> None:
    n = len(samples)
    vocab_size = len(stoi)

    length_counts: Counter = Counter(len(s["x_num"]) for s in samples)
    logger.info("Window length distribution: %s", dict(sorted(length_counts.items())))
    min_len_count = length_counts.get(min_len, 0)
    if n > 0 and (min_len_count / n) > 0.8:
        logger.warning(
            "%.1f%% of windows are at min_len=%d — data may be too sparse.",
            100.0 * min_len_count / n,
            min_len,
        )

    if max_len > 0:
        overlap_pct = max(0.0, (max_len - stride) / max_len) * 100.0
        logger.info(
            "Window overlap: %.1f%% (stride=%d, max_len=%d)",
            overlap_pct,
            stride,
            max_len,
        )

    for i, s in enumerate(samples):
        x = s["x_num"]
        if torch.isnan(x).any() or torch.isinf(x).any():
            bad_cols = [_NUMERIC_COLS[c] for c in range(x.shape[1]) if torch.isnan(x[:, c]).any() or torch.isinf(x[:, c]).any()]
            raise RuntimeError(f"NaN/Inf detected in x_num at sample index {i}, columns: {bad_cols}")
        y = s["y"]
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError(f"NaN/Inf detected in target y at sample index {i} (value={y.item()!r})")

    if samples and samples[0]["gh_ids"].dim() == 1:
        oov_samples = [i for i, s in enumerate(samples) if (s["gh_ids"] < 0).any() or (s["gh_ids"] >= vocab_size).any()]
        if oov_samples:
            raise RuntimeError(f"Out-of-vocabulary geohash IDs in {len(oov_samples)} samples (first: index {oov_samples[0]}). Vocab size: {vocab_size}.")

    all_targets = torch.cat([s["y"] for s in samples]).squeeze()
    logger.info(
        "Target magnitude: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
        all_targets.min().item(),
        all_targets.max().item(),
        all_targets.mean().item(),
        all_targets.std().item(),
    )


def make_windows_temporal(
    df: pd.DataFrame,
    stoi: dict[str, int],
    min_len: int,
    max_len: int,
    stride: int,
    encoding: Literal["flat", "hierarchical"] = "flat",
    geohash_precision: int = 4,
    char_stoi: dict[str, int] | None = None,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Create sliding windows from time-ordered earthquake sequence."""
    samples: list[dict[str, Any]] = []

    for end_idx in range(min_len, len(df)):
        current_window_len = min(max_len, end_idx)
        start_idx = end_idx - current_window_len

        while start_idx <= end_idx - min_len:
            hist = df.iloc[start_idx:end_idx]
            target = float(df.iloc[end_idx]["magnitude"])
            target_time_ms = int(df.iloc[end_idx]["time_ms"])

            samples.append(
                _make_sample_dict(
                    hist,
                    target,
                    stoi,
                    encoding,
                    geohash_precision,
                    char_stoi,
                    target_time_ms=target_time_ms,
                    target_idx=end_idx,
                )
            )
            start_idx += stride

    if not samples:
        raise RuntimeError("No training windows created. Lower min_len or widen data filters.")

    logger.info("Created %d temporal windows from %d events.", len(samples), len(df))

    if validate:
        _validate_windows(samples, stoi, min_len, max_len, stride)

    return samples


def make_windows_spatial(
    df: pd.DataFrame,
    stoi: dict[str, int],
    min_len: int,
    max_len: int,
    spatial_radius_km: float = 50.0,
    temporal_window_days: float = 30.0,
    encoding: Literal["flat", "hierarchical"] = "flat",
    geohash_precision: int = 4,
    char_stoi: dict[str, int] | None = None,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Create spatiotemporal windows from geographically clustered events."""
    from scipy.spatial import cKDTree

    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    tree = cKDTree(coords_rad)
    radius_rad = spatial_radius_km / 6371.0
    temporal_ms = temporal_window_days * 86_400_000.0
    time_ms_arr = df["time_ms"].values

    samples: list[dict[str, Any]] = []

    for idx in range(len(df)):
        current_time_ms = float(time_ms_arr[idx])
        neighbor_indices: list[int] = tree.query_ball_point(coords_rad[idx], r=radius_rad)

        valid = [j for j in neighbor_indices if j != idx and time_ms_arr[j] < current_time_ms and (current_time_ms - float(time_ms_arr[j])) <= temporal_ms]

        if len(valid) < min_len:
            continue

        hist = df.iloc[valid].sort_values("time_ms").tail(max_len)
        target = float(df.iloc[idx]["magnitude"])

        samples.append(
            _make_sample_dict(
                hist,
                target,
                stoi,
                encoding,
                geohash_precision,
                char_stoi,
                target_time_ms=int(current_time_ms),
                target_idx=idx,
            )
        )

    if not samples:
        raise RuntimeError("No spatial windows created. Try increasing --spatial-radius-km or --temporal-window-days, or lower --min-len.")

    logger.info(
        "Created %d spatial windows from %d events (radius=%.1f km, lookback=%.1f days).",
        len(samples),
        len(df),
        spatial_radius_km,
        temporal_window_days,
    )

    if validate:
        _validate_windows(samples, stoi, min_len, max_len, stride=1)

    return samples


def make_windows_hybrid(
    df: pd.DataFrame,
    stoi: dict[str, int],
    min_len: int,
    max_len: int,
    stride: int,
    spatial_radius_km: float = 50.0,
    temporal_window_days: float = 30.0,
    encoding: Literal["flat", "hierarchical"] = "flat",
    geohash_precision: int = 4,
    char_stoi: dict[str, int] | None = None,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Combine temporal and spatial windows."""
    window_kwargs = dict(
        encoding=encoding,
        geohash_precision=geohash_precision,
        char_stoi=char_stoi,
    )
    temporal = make_windows_temporal(df, stoi, min_len, max_len, stride, validate=False, **window_kwargs)
    try:
        spatial = make_windows_spatial(
            df,
            stoi,
            min_len,
            max_len,
            spatial_radius_km,
            temporal_window_days,
            validate=False,
            **window_kwargs,
        )
    except RuntimeError:
        logger.warning("Spatial windows produced no samples; using temporal only.")
        spatial = []

    samples = temporal + spatial

    if not samples:
        raise RuntimeError("No windows created in hybrid mode.")

    logger.info(
        "Hybrid mode: %d temporal + %d spatial = %d total windows.",
        len(temporal),
        len(spatial),
        len(samples),
    )

    if validate:
        _validate_windows(samples, stoi, min_len, max_len, stride)

    return samples


make_windows = make_windows_temporal


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate batch of windows with padding."""
    gh_ids_list, x_num_list, y_list = zip(*batch)
    lengths = torch.tensor([len(x) for x in x_num_list], dtype=torch.long)

    if gh_ids_list[0].dim() == 1:
        gh_ids_padded = pad_sequence(gh_ids_list, batch_first=True, padding_value=0)
    else:
        max_seq = max(g.size(0) for g in gh_ids_list)
        prec = gh_ids_list[0].size(1)
        batch_size = len(gh_ids_list)
        gh_ids_padded = torch.zeros(batch_size, max_seq, prec, dtype=torch.long)
        for i, g in enumerate(gh_ids_list):
            gh_ids_padded[i, : g.size(0)] = g

    x_num_padded = pad_sequence(x_num_list, batch_first=True, padding_value=0.0)
    y_stacked = torch.stack(y_list)

    return gh_ids_padded, x_num_padded, lengths, y_stacked


def standardize_numeric(
    train_samples: list[dict[str, torch.Tensor]],
    *other_sample_sets: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize numeric features using training set statistics."""
    train_all = torch.cat([s["x_num"] for s in train_samples], dim=0)
    mean = train_all.mean(dim=0)
    std = train_all.std(dim=0).clamp_min(1e-6)

    all_sets = (train_samples,) + other_sample_sets
    for samples in all_sets:
        for sample in samples:
            sample["x_num"] = (sample["x_num"] - mean) / std

    return mean, std


def build_windows_for_df(
    df: pd.DataFrame,
    stoi: dict[str, int],
    window_config: Any,
    geohash_config: Any,
    char_stoi: dict[str, int] | None = None,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Dispatch window creation based on config mode."""
    kwargs = dict(
        stoi=stoi,
        min_len=window_config.min_len,
        max_len=window_config.max_len,
        encoding=geohash_config.encoding,
        geohash_precision=geohash_config.precision,
        char_stoi=char_stoi,
        validate=validate,
    )
    mode = window_config.mode
    if mode == "temporal":
        return make_windows_temporal(df=df, stride=window_config.stride, **kwargs)
    if mode == "spatial":
        return make_windows_spatial(
            df=df,
            spatial_radius_km=window_config.spatial_radius_km,
            temporal_window_days=window_config.temporal_window_days,
            **kwargs,
        )
    return make_windows_hybrid(
        df=df,
        stride=window_config.stride,
        spatial_radius_km=window_config.spatial_radius_km,
        temporal_window_days=window_config.temporal_window_days,
        **kwargs,
    )
