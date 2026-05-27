"""Test data module functions."""

import math

import pandas as pd
import pytest
import torch

from geohash.data import (
    add_base_features,
    add_features,
    build_vocab,
    collate_batch,
    compute_window_features,
    encode_geohash,
    geohash_to_id,
    haversine_distance,
    make_windows,
    make_windows_spatial,
    make_windows_temporal,
    standardize_numeric,
)
from geohash.data.features import UNK_TOKEN


class TestHaversineDistance:
    def test_same_point_is_zero(self):
        assert haversine_distance(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_known_distance(self):
        dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert dist == pytest.approx(5570, rel=0.02)


class TestGeohashEncoding:
    def test_encode_geohash_basic(self):
        gh = encode_geohash(37.7749, -122.4194, precision=4)
        assert isinstance(gh, str)
        assert len(gh) == 4


class TestAddFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "time_ms": [0, 86_400_000, 172_800_000, 259_200_000],
            "latitude": [37.0, 37.1, 37.2, 37.3],
            "longitude": [-122.0, -122.1, -122.2, -122.3],
            "magnitude": [2.0, 2.5, 3.0, 2.8],
            "depth_km": [10.0, 12.0, 8.0, 15.0],
            "place": ["test"] * 4,
        })

    def test_add_base_features_creates_columns(self, sample_df):
        df = add_base_features(sample_df, geohash_precision=4)
        assert "geohash" in df.columns
        assert "time_days" in df.columns
        assert "delta_mag" not in df.columns

    def test_add_features_alias(self, sample_df):
        df = add_features(sample_df, geohash_precision=4)
        assert "geohash" in df.columns


class TestComputeWindowFeatures:
    def test_first_row_deltas_zero(self):
        hist = pd.DataFrame({
            "time_ms": [0, 86_400_000],
            "time_days": [0.0, 1.0],
            "magnitude": [2.0, 2.5],
            "latitude": [37.0, 37.1],
            "longitude": [-122.0, -122.1],
        })
        deltas = compute_window_features(hist)
        assert deltas.shape == (2, 3)
        assert deltas[0].tolist() == [0.0, 0.0, 0.0]
        assert deltas[1, 0] == pytest.approx(1.0)
        assert deltas[1, 1] == pytest.approx(0.5)

    def test_spatial_window_uses_within_window_deltas(self):
        """Non-consecutive global rows should still get correct within-window deltas."""
        hist = pd.DataFrame({
            "time_ms": [300_000, 100_000],
            "time_days": [3.0, 1.0],
            "magnitude": [3.0, 2.0],
            "latitude": [37.2, 37.0],
            "longitude": [-122.2, -122.0],
        }).sort_values("time_ms")

        deltas = compute_window_features(hist)
        assert deltas[0].tolist() == [0.0, 0.0, 0.0]
        assert deltas[1, 0] == pytest.approx(2.0)
        assert deltas[1, 1] == pytest.approx(1.0)
        assert deltas[1, 2] > 0


class TestBuildVocab:
    def test_build_vocab_with_unk(self):
        geohashes = ["u10x", "u10z", "u10x"]
        stoi = build_vocab(geohashes, include_unk=True)
        assert stoi["<PAD>"] == 0
        assert UNK_TOKEN in stoi
        assert len(stoi) == 4  # PAD, UNK, u10x, u10z

    def test_geohash_to_id_oov(self):
        stoi = build_vocab(["abc"], include_unk=True)
        assert geohash_to_id("missing", stoi) == stoi[UNK_TOKEN]


class TestMakeWindows:
    @pytest.fixture
    def sample_df_with_features(self):
        df = pd.DataFrame({
            "time_ms": [i * 86_400_000 for i in range(5)],
            "latitude": [37.0 + i * 0.1 for i in range(5)],
            "longitude": [-122.0 - i * 0.1 for i in range(5)],
            "magnitude": [2.0, 2.5, 3.0, 2.8, 3.2],
            "depth_km": [10.0, 12.0, 8.0, 15.0, 9.0],
            "place": ["test"] * 5,
        })
        return add_base_features(df, geohash_precision=4)

    def test_make_windows(self, sample_df_with_features):
        stoi = build_vocab(sample_df_with_features["geohash"].tolist())
        windows = make_windows(
            df=sample_df_with_features,
            stoi=stoi,
            min_len=2,
            max_len=3,
            stride=1,
        )
        assert len(windows) > 0

    def test_windows_have_correct_shape(self, sample_df_with_features):
        stoi = build_vocab(sample_df_with_features["geohash"].tolist())
        windows = make_windows(
            df=sample_df_with_features,
            stoi=stoi,
            min_len=2,
            max_len=3,
            stride=1,
        )
        for window in windows:
            assert window["x_num"].shape[1] == 6
            assert window["y"].shape == (1,)

    def test_make_windows_temporal_alias(self, sample_df_with_features):
        stoi = build_vocab(sample_df_with_features["geohash"].tolist())
        kwargs = dict(df=sample_df_with_features, stoi=stoi, min_len=2, max_len=3, stride=1)
        assert len(make_windows(**kwargs)) == len(make_windows_temporal(**kwargs))


class TestSpatialWindows:
    @pytest.fixture
    def clustered_df(self):
        base_lat, base_lon = 37.0, -122.0
        rows = []
        for i in range(10):
            rows.append({
                "time_ms": i * 86_400_000,
                "latitude": base_lat + i * 0.01,
                "longitude": base_lon + i * 0.01,
                "magnitude": 2.0 + i * 0.1,
                "depth_km": 10.0,
                "place": "test",
            })
        return add_base_features(pd.DataFrame(rows), geohash_precision=4)

    def test_spatial_windows_basic(self, clustered_df):
        stoi = build_vocab(clustered_df["geohash"].tolist())
        windows = make_windows_spatial(
            df=clustered_df,
            stoi=stoi,
            min_len=2,
            max_len=5,
            spatial_radius_km=200.0,
            temporal_window_days=30.0,
        )
        assert len(windows) > 0

    def test_spatial_windows_sample_structure(self, clustered_df):
        stoi = build_vocab(clustered_df["geohash"].tolist())
        windows = make_windows_spatial(
            df=clustered_df,
            stoi=stoi,
            min_len=2,
            max_len=5,
            spatial_radius_km=200.0,
            temporal_window_days=30.0,
        )
        for w in windows:
            assert "gh_ids" in w and "x_num" in w and "y" in w


class TestCollate:
    def test_collate_batch(self):
        batch = [
            (torch.tensor([1, 2]), torch.randn(2, 6), torch.tensor([3.0])),
            (torch.tensor([1, 2, 3]), torch.randn(3, 6), torch.tensor([3.5])),
        ]
        gh_pad, x_pad, lengths, y = collate_batch(batch)
        assert gh_pad.shape[0] == 2
        assert lengths[1] == 3


class TestStandardize:
    def test_standardize_numeric(self):
        train_samples = [
            {"gh_ids": torch.tensor([1]), "x_num": torch.tensor([[1.0, 2.0]]), "y": torch.tensor([3.0])},
            {"gh_ids": torch.tensor([2]), "x_num": torch.tensor([[3.0, 4.0]]), "y": torch.tensor([3.5])},
        ]
        test_samples = [
            {"gh_ids": torch.tensor([1]), "x_num": torch.tensor([[5.0, 6.0]]), "y": torch.tensor([4.0])},
        ]
        mean, std = standardize_numeric(train_samples, test_samples)
        assert mean.shape == (2,)
        assert std.shape == (2,)
