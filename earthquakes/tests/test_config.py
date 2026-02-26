"""Tests for Pydantic config models."""

from pathlib import Path

import pytest

from validation import GridConfig, RunConfig, USGSQueryParams, _parse_tuple


def test_parse_tuple_string():
    assert _parse_tuple("1,2") == (1.0, 2.0)
    assert _parse_tuple(" -0.132 , 9.796 ") == (-0.132, 9.796)


def test_parse_tuple_list():
    assert _parse_tuple([1, 2]) == (1.0, 2.0)
    assert _parse_tuple((-80.343, -72.466)) == (-80.343, -72.466)


def test_parse_tuple_invalid():
    with pytest.raises(ValueError, match="two numbers"):
        _parse_tuple("1,2,3")
    with pytest.raises(ValueError, match="two numbers"):
        _parse_tuple("1")
    with pytest.raises(ValueError, match="exactly 2"):
        _parse_tuple([1, 2, 3])


def test_run_config():
    cfg = RunConfig(latitude=(-0.132, 9.796), longitude=(-80.343, -72.466))
    assert cfg.latitude == (-0.132, 9.796)
    assert cfg.longitude == (-80.343, -72.466)
    assert cfg.output_dir == Path("plots")
    assert cfg.seed is None


def test_run_config_coerce_bounds():
    cfg = RunConfig(latitude="1,2", longitude="3,4")
    assert cfg.latitude == (1.0, 2.0)
    assert cfg.longitude == (3.0, 4.0)


def test_usgs_query_params():
    p = USGSQueryParams(
        min_latitude=-0.132,
        max_latitude=9.796,
        min_longitude=-80.343,
        max_longitude=-72.466,
    )
    assert p.eventtype == "earthquake"


def test_usgs_query_params_bounds_ordered():
    with pytest.raises(ValueError, match="max_latitude"):
        USGSQueryParams(
            min_latitude=9,
            max_latitude=-1,
            min_longitude=0,
            max_longitude=1,
        )


def test_grid_config():
    g = GridConfig(
        min_latitude=0,
        max_latitude=10,
        min_longitude=-80,
        max_longitude=-70,
        distance_km=100,
    )
    assert g.latitude == (0, 10)
    assert g.longitude == (-80, -70)
