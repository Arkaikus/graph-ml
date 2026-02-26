"""Tests for EarthquakeData.clean and related data layer."""

import pandas as pd
from data.data import EarthquakeData
from data.hash import Hashable, _serialize_for_hash


def test_clean_no_mutation_of_features():
    """clean() must not mutate self.features when delta_time adds 'delta'."""
    raw = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=10, freq="D"),
            "latitude": [0.0] * 10,
            "longitude": [0.0] * 10,
        }
    )
    features = ["latitude", "longitude"]
    data = EarthquakeData(raw, features=features, targets=[])
    assert data.features == ["latitude", "longitude"]
    data.clean()
    assert data.features == ["latitude", "longitude"]
    assert getattr(data, "_features_used", None) == ["latitude", "longitude", "delta"]


def test_clean_without_time_column():
    raw = pd.DataFrame(
        {
            "latitude": [1.0, 2.0],
            "longitude": [1.0, 2.0],
        }
    )
    data = EarthquakeData(raw, features=["latitude", "longitude"], targets=[])
    data.time_column = False
    out = data.clean()
    assert list(out.columns) == ["latitude", "longitude"]
    assert len(out) == 2


def test_hashable_includes_lists():
    """Hash should include list fields so different configs get different hashes."""

    class C(Hashable):
        def __init__(self, a: list):
            self.a = a

    h1 = C([1, 2]).hash
    # Sorted in _serialize_for_hash so order-normalized
    assert h1 == C([2, 1]).hash
    assert C(["x", "y"]).hash != h1


def test_serialize_for_hash():
    assert _serialize_for_hash([2, 1]) == [1, 2]
    assert _serialize_for_hash(["b", "a"]) == ["a", "b"]
