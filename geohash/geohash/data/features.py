"""Feature engineering for earthquake data."""

from typing import Any

import pandas as pd

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = 4) -> str:
    """
    Encode latitude/longitude to geohash string.

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.
    precision : int
        Geohash character length (1-12). Default 4.

    Returns
    -------
    str
        Geohash string.
    """
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def add_features(df: pd.DataFrame, geohash_precision: int) -> pd.DataFrame:
    """
    Add engineered features to earthquake dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: time_ms, latitude, longitude, magnitude, depth_km.
    geohash_precision : int
        Geohash precision.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: geohash, time_days, delta_t_days, delta_mag, delta_lat, delta_lon.
    """
    df = df.copy()

    # Add geohash encoding
    df["geohash"] = df.apply(
        lambda row: encode_geohash(row["latitude"], row["longitude"], geohash_precision),
        axis=1
    )

    # Add temporal and delta features
    df["time_days"] = (df["time_ms"] - df["time_ms"].min()) / (1000 * 60 * 60 * 24)
    df["delta_t_days"] = df["time_days"].diff().fillna(0.0)
    df["delta_mag"] = df["magnitude"].diff().fillna(0.0)
    df["delta_lat"] = df["latitude"].diff().fillna(0.0)
    df["delta_lon"] = df["longitude"].diff().fillna(0.0)

    return df


def build_vocab(geohashes: list[str]) -> dict[str, int]:
    """
    Build vocabulary mapping from geohash strings.

    Parameters
    ----------
    geohashes : list[str]
        List of geohash strings.

    Returns
    -------
    dict[str, int]
        Mapping from geohash to integer ID. Includes "<PAD>" with ID 0.
    """
    unique_geohashes = sorted(set(geohashes))
    stoi = {gh: i + 1 for i, gh in enumerate(unique_geohashes)}
    stoi["<PAD>"] = 0
    return stoi
