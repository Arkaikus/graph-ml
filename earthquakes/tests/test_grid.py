"""Tests for Grid (with GridConfig validation)."""

import pytest
from data.grid import Grid


def test_grid_from_bounds():
    lat = (-0.132, 9.796)
    lon = (-80.343, -72.466)
    grid = Grid(lat, lon, distance=100)
    assert grid.min_latitude == -0.132
    assert grid.max_latitude == 9.796
    assert grid.distance == 100
    assert grid.total_positions >= 0


def test_grid_invalid_bounds_raises():
    # GridConfig rejects max_latitude < min_latitude
    with pytest.raises(ValueError, match="max_latitude"):
        Grid((10.0, 0.0), (0.0, 10.0), distance=100)
