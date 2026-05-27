"""Feature engineering for earthquake data."""

import math

import numpy as np
import pandas as pd

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_EARTH_RADIUS_KM = 6371.0

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in km between two points."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def encode_geohash(lat: float, lon: float, precision: int = 4) -> str:
    """Encode latitude/longitude to geohash string."""
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


def add_base_features(df: pd.DataFrame, geohash_precision: int) -> pd.DataFrame:
    """
    Add base (non-delta) features to earthquake dataframe.

    Delta features are computed per-window via :func:`compute_window_features`.
    """
    df = df.copy()
    df["geohash"] = df.apply(
        lambda row: encode_geohash(row["latitude"], row["longitude"], geohash_precision),
        axis=1,
    )
    df["time_days"] = (df["time_ms"] - df["time_ms"].min()) / (1000 * 60 * 60 * 24)
    return df


def add_features(df: pd.DataFrame, geohash_precision: int) -> pd.DataFrame:
    """Backward-compatible alias for :func:`add_base_features`."""
    return add_base_features(df, geohash_precision)


def compute_window_features(hist: pd.DataFrame) -> np.ndarray:
    """
    Compute per-row delta features within a window sequence.

    Parameters
    ----------
    hist : pd.DataFrame
        Window history sorted by time_ms. Required columns: time_days, magnitude,
        latitude, longitude.

    Returns
    -------
    np.ndarray
        Shape (len(hist), 3) with columns delta_t_days, delta_mag, delta_distance_km.
    """
    n = len(hist)
    deltas = np.zeros((n, 3), dtype=np.float32)

    if n == 0:
        return deltas

    time_days = hist["time_days"].values
    magnitudes = hist["magnitude"].values
    lats = hist["latitude"].values
    lons = hist["longitude"].values

    for i in range(1, n):
        deltas[i, 0] = time_days[i] - time_days[i - 1]
        deltas[i, 1] = magnitudes[i] - magnitudes[i - 1]
        dist_km = haversine_distance(lats[i - 1], lons[i - 1], lats[i], lons[i])
        deltas[i, 2] = math.log1p(dist_km)

    return deltas


def build_vocab(geohashes: list[str], include_unk: bool = True) -> dict[str, int]:
    """
    Build vocabulary mapping from geohash strings (train set only).

    PAD=0, UNK=1 (when include_unk), then sorted unique geohashes from id 2.
    """
    stoi: dict[str, int] = {PAD_TOKEN: 0}
    next_id = 1
    if include_unk:
        stoi[UNK_TOKEN] = next_id
        next_id += 1

    unique_geohashes = sorted(set(geohashes))
    for gh in unique_geohashes:
        if gh not in stoi:
            stoi[gh] = next_id
            next_id += 1

    return stoi


def geohash_to_id(geohash: str, stoi: dict[str, int]) -> int:
    """Map geohash to integer id; OOV maps to UNK or PAD if UNK missing."""
    if geohash in stoi:
        return stoi[geohash]
    return stoi.get(UNK_TOKEN, 0)


def build_char_vocab() -> dict[str, int]:
    """Build char-level vocab for hierarchical encoding: PAD=0, UNK=1, then BASE32 chars."""
    stoi: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, ch in enumerate(BASE32):
        stoi[ch] = i + 2
    return stoi


def geohash_to_char_ids(geohash: str, precision: int, char_stoi: dict[str, int]) -> list[int]:
    """Convert geohash string to fixed-length char id list (pad/truncate to precision)."""
    unk_id = char_stoi[UNK_TOKEN]
    pad_id = char_stoi[PAD_TOKEN]
    gh = geohash[:precision]
    ids = [char_stoi.get(ch, unk_id) for ch in gh]
    while len(ids) < precision:
        ids.append(pad_id)
    return ids


def char_ids_for_sequence(geohashes: list[str], precision: int, char_stoi: dict[str, int]) -> list[list[int]]:
    """Convert a list of geohash strings to char id sequences."""
    return [geohash_to_char_ids(g, precision, char_stoi) for g in geohashes]
