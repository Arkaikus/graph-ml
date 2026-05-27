"""Test temporal event splitting."""

import pandas as pd
import pytest

from geohash.data.features import build_vocab, geohash_to_id
from geohash.data.split import split_events_temporal
from geohash.data import add_base_features, make_windows_temporal


@pytest.fixture
def event_df():
    df = pd.DataFrame({
        "time_ms": [i * 86_400_000 for i in range(20)],
        "latitude": [37.0 + i * 0.01 for i in range(20)],
        "longitude": [-122.0 - i * 0.01 for i in range(20)],
        "magnitude": [2.0 + (i % 5) * 0.2 for i in range(20)],
        "depth_km": [10.0] * 20,
        "place": ["t"] * 20,
    })
    return add_base_features(df, geohash_precision=4)


class TestSplitEventsTemporal:
    def test_no_time_overlap(self, event_df):
        train_df, val_df, test_df = split_events_temporal(event_df, 0.7, 0.1, 0.2)
        train_times = set(train_df["time_ms"])
        val_times = set(val_df["time_ms"])
        test_times = set(test_df["time_ms"])
        assert train_times.isdisjoint(val_times)
        assert train_times.isdisjoint(test_times)
        assert val_times.isdisjoint(test_times)

    def test_temporal_order(self, event_df):
        train_df, val_df, test_df = split_events_temporal(event_df, 0.7, 0.1, 0.2)
        assert train_df["time_ms"].max() < val_df["time_ms"].min()
        assert val_df["time_ms"].max() < test_df["time_ms"].min()

    def test_ratios_must_sum_to_one(self, event_df):
        with pytest.raises(ValueError):
            split_events_temporal(event_df, 0.5, 0.3, 0.3)

    def test_oov_maps_to_unk(self, event_df):
        train_df, val_df, test_df = split_events_temporal(event_df, 0.7, 0.1, 0.2)
        stoi = build_vocab(train_df["geohash"].tolist(), include_unk=True)
        test_geohashes = set(test_df["geohash"]) - set(stoi.keys())
        for gh in test_geohashes:
            assert geohash_to_id(gh, stoi) == stoi["<UNK>"]

    def test_train_test_target_indices_disjoint(self, event_df):
        train_df, val_df, test_df = split_events_temporal(event_df, 0.7, 0.1, 0.2)
        stoi = build_vocab(train_df["geohash"].tolist(), include_unk=True)

        train_w = make_windows_temporal(train_df, stoi, min_len=2, max_len=4, stride=1, validate=False)
        test_w = make_windows_temporal(test_df, stoi, min_len=2, max_len=4, stride=1, validate=False)

        train_targets = {w["target_time_ms"] for w in train_w if "target_time_ms" in w}
        test_targets = {w["target_time_ms"] for w in test_w if "target_time_ms" in w}
        assert train_targets.isdisjoint(test_targets)
