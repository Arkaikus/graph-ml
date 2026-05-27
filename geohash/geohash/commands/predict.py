"""Predict command using saved run artifacts."""

import logging
from pathlib import Path

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader

from geohash.data import (
    QuakeWindowDataset,
    add_base_features,
    build_windows_for_df,
    collate_batch,
    standardize_numeric,
)
from geohash.inference.artifacts import build_model_from_bundle, load_run_bundle

logger = logging.getLogger(__name__)


@click.command(name="predict")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False), required=True, help="Saved run directory")
@click.option("--input-csv", type=click.Path(exists=True, dir_okay=False), required=True, help="Input events CSV")
@click.option("--output", type=click.Path(), default="predictions_out.csv", help="Output predictions CSV")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", help="Device")
def predict_cmd(run_dir: str, input_csv: str, output: str, device: str):
    """Run magnitude prediction using a saved training run bundle."""
    logging.basicConfig(level=logging.INFO)
    run_path = Path(run_dir)

    bundle = load_run_bundle(run_path)
    if "preprocess" not in bundle or "model_state_dict" not in bundle:
        raise click.ClickException(
            f"Run dir missing preprocess.json or model_final.pt: {run_path}"
        )

    pp = bundle["preprocess"]
    cfg = bundle.get("config", {})
    window_cfg = cfg.get("window", {})
    geohash_cfg = cfg.get("geohash", {})

    required_cols = {"time_ms", "latitude", "longitude", "magnitude", "depth_km"}
    df = pd.read_csv(input_csv)
    missing = required_cols - set(df.columns)
    if missing:
        raise click.ClickException(f"input-csv missing columns: {sorted(missing)}")

    df = df.sort_values("time_ms").reset_index(drop=True)
    df = add_base_features(df, pp["geohash_precision"])

    from geohash.config import GeohashConfig, WindowConfig

    window = WindowConfig(**{k: window_cfg[k] for k in WindowConfig.model_fields if k in window_cfg})
    geohash = GeohashConfig(**{k: geohash_cfg[k] for k in GeohashConfig.model_fields if k in geohash_cfg})
    if pp.get("encoding"):
        geohash.encoding = pp["encoding"]

    stoi = pp["stoi"]
    char_stoi = pp.get("char_stoi")

    samples = build_windows_for_df(
        df, stoi, window, geohash, char_stoi=char_stoi, validate=False
    )
    if not samples:
        raise click.ClickException("No windows could be built from input CSV.")

    mean = torch.tensor(pp["numeric_mean"], dtype=torch.float32)
    std = torch.tensor(pp["numeric_std"], dtype=torch.float32)
    for s in samples:
        s["x_num"] = (s["x_num"] - mean) / std

    loader = DataLoader(
        QuakeWindowDataset(samples),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = build_model_from_bundle(bundle)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    preds: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for gh_ids, x_num, lengths, y in loader:
            gh_ids = gh_ids.to(device)
            x_num = x_num.to(device)
            lengths = lengths.to(device)
            pred = model(gh_ids, x_num, lengths)
            preds.extend(pred.squeeze(1).cpu().tolist())
            targets.extend(y.squeeze(1).cpu().tolist())

    out_df = pd.DataFrame({"target": targets, "predicted": preds})
    out_path = Path(output)
    out_df.to_csv(out_path, index=False)
    click.echo(f"✓ Wrote {len(out_df)} predictions to {out_path}")
