import math
import random
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

CONFIG = {
    "minlatitude": 32.0,
    "maxlatitude": 42.0,
    "minlongitude": -125.0,
    "maxlongitude": -114.0,
    "starttime": "2018-01-01",
    "endtime": "2024-12-31",
    "minmagnitude": 1.0,
    "orderby": "time-asc",
    "limit": 20000,
    "geohash_precision": 4,
    "window_min_len": 5,
    "window_max_len": 30,
    "stride": 1,
    "batch_size": 64,
    "embedding_dim": 16,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "lr": 1e-3,
    "epochs": 12,
    "train_split": 0.8,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_geohash(lat, lon, precision=4):
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
            geohash.append(_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def fetch_usgs_events(cfg):
    params = {
        "format": "geojson",
        "starttime": cfg["starttime"],
        "endtime": cfg["endtime"],
        "minmagnitude": cfg["minmagnitude"],
        "minlatitude": cfg["minlatitude"],
        "maxlatitude": cfg["maxlatitude"],
        "minlongitude": cfg["minlongitude"],
        "maxlongitude": cfg["maxlongitude"],
        "orderby": cfg["orderby"],
        "limit": cfg["limit"],
    }

    r = requests.get(USGS_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    rows = []
    for f in payload["features"]:
        prop = f["properties"]
        geom = f["geometry"]
        if geom is None or prop is None:
            continue
        coords = geom["coordinates"]
        if coords is None or len(coords) < 3:
            continue

        lon, lat, depth = coords[0], coords[1], coords[2]
        mag = prop.get("mag")
        t_ms = prop.get("time")

        if mag is None or t_ms is None:
            continue

        rows.append({
            "time_ms": int(t_ms),
            "time": datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc),
            "latitude": float(lat),
            "longitude": float(lon),
            "depth_km": float(depth),
            "magnitude": float(mag),
            "place": prop.get("place", ""),
        })

    df = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No events returned. Widen the date range or bounding box.")
    return df


def add_features(df, geohash_precision):
    df = df.copy()
    df["geohash"] = df.apply(
        lambda r: encode_geohash(r["latitude"], r["longitude"], geohash_precision), axis=1
    )
    df["time_days"] = (df["time_ms"] - df["time_ms"].min()) / (1000 * 60 * 60 * 24)
    df["delta_t_days"] = df["time_days"].diff().fillna(0.0)
    df["delta_mag"] = df["magnitude"].diff().fillna(0.0)
    df["delta_lat"] = df["latitude"].diff().fillna(0.0)
    df["delta_lon"] = df["longitude"].diff().fillna(0.0)
    return df


def build_vocab(geohashes):
    uniq = sorted(set(geohashes))
    stoi = {gh: i + 1 for i, gh in enumerate(uniq)}
    stoi["<PAD>"] = 0
    return stoi


def make_windows(df, stoi, min_len, max_len, stride):
    samples = []

    for end_idx in range(min_len, len(df)):
        max_start = max(0, end_idx - max_len)
        min_start = max(0, end_idx - max_len)
        current_window_len = min(max_len, end_idx)

        start_idx = end_idx - current_window_len
        while start_idx <= end_idx - min_len:
            hist = df.iloc[start_idx:end_idx]
            target = df.iloc[end_idx]["magnitude"]

            gh_ids = torch.tensor(
                [stoi[g] for g in hist["geohash"].tolist()],
                dtype=torch.long
            )

            num_feats = torch.tensor(
                hist[[
                    "magnitude",
                    "depth_km",
                    "time_days",
                    "delta_t_days",
                    "delta_mag",
                    "delta_lat",
                    "delta_lon",
                ]].values,
                dtype=torch.float32
            )

            samples.append({
                "gh_ids": gh_ids,
                "x_num": num_feats,
                "y": torch.tensor([target], dtype=torch.float32),
            })
            start_idx += stride

    if len(samples) == 0:
        raise RuntimeError("No training windows created. Lower min_len or widen data filters.")
    return samples


class QuakeWindowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["gh_ids"], s["x_num"], s["y"]


def collate_batch(batch):
    gh_ids, x_num, y = zip(*batch)
    lengths = torch.tensor([len(x) for x in x_num], dtype=torch.long)
    gh_pad = pad_sequence(gh_ids, batch_first=True, padding_value=0)
    x_pad = pad_sequence(x_num, batch_first=True, padding_value=0.0)
    y = torch.stack(y)
    return gh_pad, x_pad, lengths, y


def standardize_numeric(train_samples, test_samples):
    train_all = torch.cat([s["x_num"] for s in train_samples], dim=0)
    mean = train_all.mean(dim=0)
    std = train_all.std(dim=0).clamp_min(1e-6)

    for bucket in (train_samples, test_samples):
        for s in bucket:
            s["x_num"] = (s["x_num"] - mean) / std

    return mean, std


class NextMagnitudeLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_numeric, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + num_numeric,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, gh_ids, x_num, lengths):
        gh_emb = self.embed(gh_ids)
        x = torch.cat([gh_emb, x_num], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        return self.head(last_hidden)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    losses = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for gh_ids, x_num, lengths, y in loader:
            gh_ids = gh_ids.to(device)
            x_num = x_num.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            pred = model(gh_ids, x_num, lengths)
            loss = criterion(pred, y)
            losses.append(loss.item())

            preds.extend(pred.squeeze(1).cpu().numpy().tolist())
            targets.extend(y.squeeze(1).cpu().numpy().tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    return float(np.mean(losses)), rmse, mae


def train(model, train_loader, test_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for gh_ids, x_num, lengths, y in train_loader:
            gh_ids = gh_ids.to(device)
            x_num = x_num.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(gh_ids, x_num, lengths)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss, rmse, mae = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"rmse={rmse:.4f} | mae={mae:.4f}"
        )


def main():
    set_seed(CONFIG["seed"])

    print("Fetching USGS data...")
    df = fetch_usgs_events(CONFIG)
    print(f"Retrieved {len(df)} events")

    print("Adding geohash and engineered features...")
    df = add_features(df, CONFIG["geohash_precision"])

    stoi = build_vocab(df["geohash"].tolist())
    vocab_size = len(stoi)

    print("Building sliding windows...")
    samples = make_windows(
        df=df,
        stoi=stoi,
        min_len=CONFIG["window_min_len"],
        max_len=CONFIG["window_max_len"],
        stride=CONFIG["stride"],
    )
    print(f"Created {len(samples)} training windows")
    print(f"Geohash vocab size: {vocab_size}")

    split_idx = int(len(samples) * CONFIG["train_split"])
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    standardize_numeric(train_samples, test_samples)

    train_ds = QuakeWindowDataset(train_samples)
    test_ds = QuakeWindowDataset(test_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    num_numeric = train_samples[0]["x_num"].shape[1]

    model = NextMagnitudeLSTM(
        vocab_size=vocab_size,
        embedding_dim=CONFIG["embedding_dim"],
        num_numeric=num_numeric,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])

    print(f"Training on {CONFIG['device']} ...")
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=CONFIG["device"],
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
    )


if __name__ == "__main__":
    main()