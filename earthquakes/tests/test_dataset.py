"""Tests for parquet dataset precomputation (load_sequences, precompute_sequences, path resolution)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from data.data import EarthquakeData
from lstm.dataset import (
    load_sequences,
    precompute_sequences,
    save_sequences_parquet,
    sequences_path_for_config,
)


def _make_synthetic_qdata(n_rows: int = 100, with_time: bool = True) -> EarthquakeData:
    """Create minimal EarthquakeData for testing."""
    base = {
        "latitude": np.random.randn(n_rows).cumsum() + 5,
        "longitude": np.random.randn(n_rows).cumsum() - 75,
        "depth": np.random.uniform(0, 50, n_rows),
        "mag": np.random.uniform(2, 6, n_rows),
        "magType": ["mb"] * n_rows,
    }
    if with_time:
        raw = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=n_rows, freq="h"), **base})
    else:
        raw = pd.DataFrame(base)
    return EarthquakeData(
        raw,
        features=["latitude", "longitude", "depth", "mag"],
        targets=["mag"],
        time_column=with_time,
        min_magnitude=0,
        max_magnitude=10,
    )


def test_load_sequences_roundtrip():
    """Save sequences to parquet, load, assert shape and values match."""
    np.random.seed(42)
    n, lookback, num_features = 50, 10, 4
    sequences = np.random.randn(n, lookback, num_features).astype(np.float32)
    targets = np.random.randn(n, lookback).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seq.parquet"
        save_sequences_parquet(sequences, targets, path)
        loaded_seq, loaded_tgt = load_sequences(path)
        np.testing.assert_array_almost_equal(loaded_seq, sequences)
        np.testing.assert_array_almost_equal(loaded_tgt, targets)
        assert loaded_seq.shape == sequences.shape
        assert loaded_tgt.shape == targets.shape


def test_precompute_sequences_creates_expected_files():
    """Precompute with minimal param_space; assert parquet files exist."""
    qdata = _make_synthetic_qdata(n_rows=150)
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = precompute_sequences(
            qdata,
            task="regression",
            networkx=False,
            quantiles=2,
            lookback_choices=[10, 30],
            out_dir=tmp,
        )
        assert cache_dir.exists()
        parquet_files = list(cache_dir.glob("*.parquet"))
        assert len(parquet_files) == 2
        for pf in parquet_files:
            meta = pf.with_suffix(".parquet.meta.json")
            assert meta.exists()
        config = {"lookback": 10, "network_lookback": 5}
        path = sequences_path_for_config(config, cache_dir, task="regression")
        assert path.exists()


def test_sequences_path_for_config():
    """Path matches naming convention for config."""
    cache_dir = Path("/cache/parquet/abc123")
    p1 = sequences_path_for_config(
        {"lookback": 30, "network_lookback": 5},
        cache_dir,
        task="regression",
    )
    assert p1 == cache_dir / "seq_lookback_30_nx_5.parquet"
    p2 = sequences_path_for_config(
        {"lookback": 50, "network_lookback": 7, "quantiles": 3},
        cache_dir,
        task="classification",
        quantiles=3,
    )
    assert p2 == cache_dir / "seq_lookback_50_nx_7_q_3.parquet"


def test_load_sequences_shape_consistency():
    """Loaded sequences have shape (N, lookback, features)."""
    np.random.seed(42)
    n, lookback, num_features = 80, 15, 4
    sequences = np.random.randn(n, lookback, num_features).astype(np.float32)
    targets = np.random.randn(n, lookback).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seq.parquet"
        save_sequences_parquet(sequences, targets, path)
        loaded_seq, loaded_tgt = load_sequences(path)
        assert loaded_seq.shape == (n, lookback, num_features)
        assert loaded_tgt.shape == (n, lookback)


def test_precompute_classification_vs_regression():
    """Classification uses categorical; regression uses normalized_data; feature dims differ."""
    qdata = _make_synthetic_qdata(n_rows=150)
    with tempfile.TemporaryDirectory() as tmp:
        cache_reg = precompute_sequences(
            qdata,
            task="regression",
            networkx=False,
            quantiles=2,
            lookback_choices=[20],
            out_dir=tmp,
        )
        cache_cls = precompute_sequences(
            qdata,
            task="classification",
            networkx=False,
            quantiles=2,
            lookback_choices=[20],
            out_dir=tmp,
        )
        path_reg = cache_reg / "seq_lookback_20_nx_5.parquet"
        path_cls = cache_cls / "seq_lookback_20_nx_5_q_2.parquet"
        seq_reg, tgt_reg = load_sequences(path_reg)
        seq_cls, tgt_cls = load_sequences(path_cls)
        # to_sequences returns (N, features, lookback)
        assert seq_reg.shape[2] == 20
        assert seq_cls.shape[2] == 20
        assert seq_reg.shape[1] == len(qdata.features)
        assert seq_cls.shape[1] != seq_reg.shape[1]
        assert tgt_reg.shape[1] >= 1
        assert tgt_cls.shape[1] == 1


def test_precompute_sequences_skips_existing_unless_force():
    """Without force, existing parquet+meta are skipped; with force, recompute."""
    qdata = _make_synthetic_qdata(n_rows=100)
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = precompute_sequences(
            qdata,
            task="regression",
            networkx=False,
            quantiles=2,
            lookback_choices=[10],
            out_dir=tmp,
        )
        path = cache_dir / "seq_lookback_10_nx_5.parquet"
        meta = path.with_suffix(".parquet.meta.json")
        assert path.exists()
        assert meta.exists()
        mtime_before = path.stat().st_mtime

        # Second call without force: should skip (file unchanged)
        precompute_sequences(
            qdata,
            task="regression",
            networkx=False,
            quantiles=2,
            lookback_choices=[10],
            out_dir=tmp,
            force=False,
        )
        mtime_after_skip = path.stat().st_mtime
        assert mtime_after_skip == mtime_before

        # With force: should recompute (file updated)
        precompute_sequences(
            qdata,
            task="regression",
            networkx=False,
            quantiles=2,
            lookback_choices=[10],
            out_dir=tmp,
            force=True,
        )
        mtime_after_force = path.stat().st_mtime
        assert mtime_after_force >= mtime_after_skip
