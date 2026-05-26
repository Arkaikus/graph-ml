"""Test data module functions."""

import pandas as pd
import pytest
import torch

from geohash.data import (
    encode_geohash,
    add_features,
    build_vocab,
    make_windows,
    collate_batch,
    standardize_numeric,
)


class TestGeohashEncoding:
    """Test geohash encoding."""

    def test_encode_geohash_basic(self):
        """Test basic geohash encoding."""
        gh = encode_geohash(37.7749, -122.4194, precision=4)
        assert isinstance(gh, str)
        assert len(gh) == 4

    def test_encode_geohash_precision(self):
        """Test different precision levels."""
        for precision in [1, 4, 8, 12]:
            gh = encode_geohash(37.7749, -122.4194, precision=precision)
            assert len(gh) == precision

    def test_encode_geohash_consistency(self):
        """Test encoding is consistent."""
        gh1 = encode_geohash(37.7749, -122.4194, precision=4)
        gh2 = encode_geohash(37.7749, -122.4194, precision=4)
        assert gh1 == gh2


class TestAddFeatures:
    """Test feature engineering."""

    @pytest.fixture
    def sample_df(self):
        """Create sample earthquake dataframe."""
        import pandas as pd
        from datetime import datetime, timezone

        return pd.DataFrame({
            "time_ms": [0, 1000, 2000, 3000],
            "latitude": [37.0, 37.1, 37.2, 37.3],
            "longitude": [-122.0, -122.1, -122.2, -122.3],
            "magnitude": [2.0, 2.5, 3.0, 2.8],
            "depth_km": [10.0, 12.0, 8.0, 15.0],
            "place": ["test"] * 4,
        })

    def test_add_features_creates_columns(self, sample_df):
        """Test that feature engineering creates expected columns."""
        df = add_features(sample_df, geohash_precision=4)
        expected_cols = ["geohash", "time_days", "delta_t_days", "delta_mag", "delta_lat", "delta_lon"]
        for col in expected_cols:
            assert col in df.columns

    def test_add_features_geohash(self, sample_df):
        """Test geohash column is populated."""
        df = add_features(sample_df, geohash_precision=4)
        assert df["geohash"].notna().all()
        assert (df["geohash"].str.len() == 4).all()

    def test_add_features_delta_magnitude(self, sample_df):
        """Test magnitude delta calculation."""
        df = add_features(sample_df, geohash_precision=4)
        assert df["delta_mag"].iloc[0] == 0.0  # First is NaN -> fillna(0)
        assert df["delta_mag"].iloc[1] == pytest.approx(0.5)
        assert df["delta_mag"].iloc[2] == pytest.approx(0.5)


class TestBuildVocab:
    """Test vocabulary building."""

    def test_build_vocab(self):
        """Test vocab creation."""
        geohashes = ["u10x", "u10z", "u10x", "u11p"]
        stoi = build_vocab(geohashes)
        assert "<PAD>" in stoi
        assert stoi["<PAD>"] == 0
        assert len(stoi) == 4  # 3 unique + PAD

    def test_build_vocab_sorted(self):
        """Test vocab is sorted."""
        geohashes = ["z", "a", "m"]
        stoi = build_vocab(geohashes)
        # Non-PAD entries should be sorted
        non_pad = {k: v for k, v in stoi.items() if k != "<PAD>"}
        assert list(non_pad.keys()) == sorted(non_pad.keys())


class TestMakeWindows:
    """Test sliding window creation."""

    @pytest.fixture
    def sample_df_with_features(self):
        """Create sample earthquake dataframe with features."""
        from geohash.data import add_features
        df = pd.DataFrame({
            "time_ms": [0, 1000, 2000, 3000, 4000],
            "latitude": [37.0, 37.1, 37.2, 37.3, 37.4],
            "longitude": [-122.0, -122.1, -122.2, -122.3, -122.4],
            "magnitude": [2.0, 2.5, 3.0, 2.8, 3.2],
            "depth_km": [10.0, 12.0, 8.0, 15.0, 9.0],
            "place": ["test"] * 5,
        })
        df = add_features(df, geohash_precision=4)
        return df

    def test_make_windows(self, sample_df_with_features):
        """Test window creation."""
        stoi = build_vocab(sample_df_with_features["geohash"].tolist())
        windows = make_windows(
            df=sample_df_with_features,
            stoi=stoi,
            min_len=2,
            max_len=3,
            stride=1,
        )
        assert len(windows) > 0
        for window in windows:
            assert "gh_ids" in window
            assert "x_num" in window
            assert "y" in window

    def test_windows_have_correct_shape(self, sample_df_with_features):
        """Test window shapes are correct."""
        stoi = build_vocab(sample_df_with_features["geohash"].tolist())
        windows = make_windows(
            df=sample_df_with_features,
            stoi=stoi,
            min_len=2,
            max_len=3,
            stride=1,
        )
        for window in windows:
            assert window["gh_ids"].dim() == 1  # 1D tensor
            assert window["x_num"].dim() == 2  # 2D tensor
            assert window["y"].shape == (1,)


class TestCollate:
    """Test batch collation."""

    def test_collate_batch(self):
        """Test collate_batch function."""
        batch = [
            (torch.tensor([1, 2]), torch.randn(2, 7), torch.tensor([3.0])),
            (torch.tensor([1, 2, 3]), torch.randn(3, 7), torch.tensor([3.5])),
        ]
        gh_pad, x_pad, lengths, y = collate_batch(batch)

        assert gh_pad.shape[0] == 2  # batch_size
        assert lengths[0] == 2
        assert lengths[1] == 3


class TestStandardize:
    """Test numeric standardization."""

    def test_standardize_numeric(self):
        """Test standardization."""
        train_samples = [
            {"gh_ids": torch.tensor([1]), "x_num": torch.tensor([[1.0, 2.0]]), "y": torch.tensor([3.0])},
            {"gh_ids": torch.tensor([2]), "x_num": torch.tensor([[3.0, 4.0]]), "y": torch.tensor([3.5])},
        ]
        test_samples = [
            {"gh_ids": torch.tensor([1]), "x_num": torch.tensor([[5.0, 6.0]]), "y": torch.tensor([4.0])},
        ]

        mean, std = standardize_numeric(train_samples, test_samples)

        # Check mean and std are computed
        assert mean.shape == (2,)
        assert std.shape == (2,)

        # Check train samples are standardized
        assert train_samples[0]["x_num"].mean().item() != 1.0  # Should be different now
