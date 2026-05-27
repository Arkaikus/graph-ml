"""USGS earthquake data fetching."""

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_usgs_events(
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
    start_time: str,
    end_time: str,
    min_magnitude: float,
    order_by: str = "time-asc",
    limit: int = 20000,
) -> pd.DataFrame:
    """
    Fetch earthquake events from USGS.

    Parameters
    ----------
    min_latitude, max_latitude : float
        Latitude bounds.
    min_longitude, max_longitude : float
        Longitude bounds.
    start_time, end_time : str
        Date range (YYYY-MM-DD format).
    min_magnitude : float
        Minimum magnitude filter.
    order_by : str
        Sort order (time-asc, time, magnitude).
    limit : int
        Maximum number of events to fetch.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time_ms, time, latitude, longitude, depth_km, magnitude, place.

    Raises
    ------
    RuntimeError
        If no events are returned from the query.
    """
    params: dict[str, Any] = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": min_magnitude,
        "minlatitude": min_latitude,
        "maxlatitude": max_latitude,
        "minlongitude": min_longitude,
        "maxlongitude": max_longitude,
        "orderby": order_by,
        "limit": limit,
    }

    response = requests.get(USGS_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for feature in payload.get("features", []):
        prop = feature.get("properties")
        geom = feature.get("geometry")

        if geom is None or prop is None:
            continue

        coords = geom.get("coordinates")
        if coords is None or len(coords) < 3:
            continue

        lon, lat, depth = coords[0], coords[1], coords[2]
        mag = prop.get("mag")
        t_ms = prop.get("time")

        if mag is None or t_ms is None:
            continue

        rows.append(
            {
                "time_ms": int(t_ms),
                "time": datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc),
                "latitude": float(lat),
                "longitude": float(lon),
                "depth_km": float(depth),
                "magnitude": float(mag),
                "place": prop.get("place", ""),
            }
        )

    if not rows:
        raise RuntimeError("No events returned. Widen the date range or bounding box.")

    df = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    return df
