"""Parquet-based dataset precomputation for LSTM. Moves to_sequences off workers."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOOKBACK_CHOICES = [10, 30, 50, 80, 120, 150]
NETWORK_LOOKBACK_CHOICES = [1, 3, 5, 7, 10]


def sequences_path_for_config(
    config: dict,
    cache_dir: Path,
    task: str = "regression",
    quantiles: int | None = None,
) -> Path:
    """Resolve parquet path from trial config."""
    lookback = config.get("lookback")
    network_lookback = config.get("network_lookback", 5)
    if task == "classification":
        q = quantiles or config.get("quantiles", 2)
        return cache_dir / f"seq_lookback_{lookback}_nx_{network_lookback}_q_{q}.parquet"
    return cache_dir / f"seq_lookback_{lookback}_nx_{network_lookback}.parquet"


def save_sequences_parquet(
    sequences: np.ndarray,
    targets: np.ndarray,
    path: Path,
) -> None:
    """Save sequences and targets to parquet with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n, lookback, num_features = sequences.shape
    seq_flat = sequences.reshape(n, -1)
    tgt_flat = targets.reshape(n, -1)
    seq_cols = [f"seq_{i}" for i in range(seq_flat.shape[1])]
    tgt_cols = [f"tgt_{i}" for i in range(tgt_flat.shape[1])]
    df = pd.DataFrame(
        np.hstack([seq_flat, tgt_flat]),
        columns=seq_cols + tgt_cols,
    )
    df.to_parquet(path, index=False)
    meta = {
        "lookback": int(lookback),
        "num_features": int(num_features),
        "target_cols": tgt_flat.shape[1],
    }
    meta_path = path.with_suffix(".parquet.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_sequences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (sequences, targets) from parquet."""
    meta_path = path.with_suffix(".parquet.meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    lookback = meta["lookback"]
    num_features = meta["num_features"]
    target_cols = meta["target_cols"]

    df = pd.read_parquet(path)
    seq_cols = [c for c in df.columns if c.startswith("seq_")]
    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    seq_flat = df[seq_cols].values
    tgt_flat = df[tgt_cols].values
    n = seq_flat.shape[0]
    sequences = seq_flat.reshape(n, lookback, num_features)
    targets = tgt_flat.reshape(n, target_cols)
    return sequences, targets


def precompute_sequences(
    qdata,
    task: str,
    networkx: bool,
    quantiles: int,
    lookback_choices: list[int] | None = None,
    network_lookback_choices: list[int] | None = None,
    out_dir: Path | str = "cache/parquet",
    force: bool = False,
) -> Path:
    """Precompute sequences for all (lookback, network_lookback, quantiles?) combos. Returns cache_dir.

    Skips combos whose parquet + meta files already exist unless force=True.
    """
    out_dir = Path(out_dir)
    cache_dir = out_dir / qdata.hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    lookbacks = lookback_choices or LOOKBACK_CHOICES
    nx_lookbacks = network_lookback_choices or NETWORK_LOOKBACK_CHOICES if networkx else [5]

    network_features = (
        [
            "degree_centrality",
            "clustering",
            "betweenness_centrality",
            "closeness_centrality",
            "pagerank",
        ]
        if networkx
        else []
    )

    for lookback in lookbacks:
        for nx_lb in nx_lookbacks:
            path = sequences_path_for_config(
                {"lookback": lookback, "network_lookback": nx_lb},
                cache_dir,
                task=task,
                quantiles=quantiles if task == "classification" else None,
            )
            meta_path = path.with_suffix(".parquet.meta.json")
            if not force and path.exists() and meta_path.exists():
                logger.debug("Skipping (exists): %s", path)
                continue

            if task == "classification":
                one_hot, binned = qdata.categorical(quantiles, keep_node=bool(network_features))
                features = list(one_hot.columns)
                (target,) = qdata.targets
                one_hot["target"] = binned[f"{target}_binned"]
                sequences, targets = qdata.to_sequences(
                    one_hot,
                    lookback,
                    features=features,
                    targets=["target"],
                    network_features=network_features,
                    network_lookback=nx_lb,
                )
                targets_for_split = targets[:, -1:].astype(np.int64)
            else:
                sequences, targets = qdata.to_sequences(
                    qdata.normalized_data,
                    lookback,
                    network_features=network_features,
                    network_lookback=nx_lb,
                )
                targets_for_split = targets

            save_sequences_parquet(sequences, targets_for_split, path)
            logger.info("Precomputed %s", path)

    return cache_dir
