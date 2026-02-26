import logging
import os

from dotenv import load_dotenv
from validation import QuakesConfig

logging.basicConfig(level=logging.INFO)


def read_config(
    latitude: tuple[float, float] | list[float] | str | None = None,
    longitude: tuple[float, float] | list[float] | str | None = None,
) -> QuakesConfig:
    """Load QuakesConfig from env with optional overrides (e.g. from CLI). Validation is done by QuakesConfig."""
    load_dotenv(override=True)
    return QuakesConfig(
        latitude=latitude if latitude is not None else os.getenv("LATITUDE", "0,0"),
        longitude=longitude if longitude is not None else os.getenv("LONGITUDE", "0,0"),
    )


def read_coordinates(
    latitude: tuple[float, float] | list[float] | str | None = None,
    longitude: tuple[float, float] | list[float] | str | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return validated (latitude, longitude) as (min, max) tuples. Uses env if not provided."""
    cfg = read_config(latitude=latitude, longitude=longitude)
    return cfg.latitude, cfg.longitude
