import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from validation import RunConfig, _parse_tuple

logging.basicConfig(level=logging.INFO)


def load_run_config(
    latitude: tuple[float, float] | list[float] | str | None = None,
    longitude: tuple[float, float] | list[float] | str | None = None,
    output_dir: str | os.PathLike | None = None,
    seed: int | None = None,
) -> RunConfig:
    """Load RunConfig from env with optional overrides (e.g. from CLI)."""
    load_dotenv(override=True)
    lat = latitude
    lon = longitude
    if lat is None:
        raw = os.getenv("LATITUDE", "")
        lat = _parse_tuple(raw) if raw else (0.0, 0.0)
    elif isinstance(lat, str):
        lat = _parse_tuple(lat)
    elif isinstance(lat, (list, tuple)) and len(lat) == 2:
        lat = (float(lat[0]), float(lat[1]))
    if lon is None:
        raw = os.getenv("LONGITUDE", "")
        lon = _parse_tuple(raw) if raw else (0.0, 0.0)
    elif isinstance(lon, str):
        lon = _parse_tuple(lon)
    elif isinstance(lon, (list, tuple)) and len(lon) == 2:
        lon = (float(lon[0]), float(lon[1]))
    out = output_dir
    if out is None:
        out = Path(os.getenv("OUTPUT_DIR", "plots"))
    else:
        out = Path(out) if not isinstance(out, Path) else out
    seed_val = seed
    if seed_val is None and os.getenv("SEED"):
        seed_val = int(os.getenv("SEED"))
    return RunConfig(
        latitude=lat,
        longitude=lon,
        output_dir=out,
        seed=seed_val,
    )


def read_coordinates(
    latitude: tuple[float, float] | list[float] | str | None = None,
    longitude: tuple[float, float] | list[float] | str | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return validated (latitude, longitude) as (min, max) tuples. Uses env if not provided."""
    cfg = load_run_config(latitude=latitude, longitude=longitude)
    return cfg.latitude, cfg.longitude
