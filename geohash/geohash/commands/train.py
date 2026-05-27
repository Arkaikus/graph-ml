"""Train command."""

import logging

import click
import torch
from torch.utils.data import DataLoader

from geohash.config import RunConfig
from geohash.data import (
    QuakeWindowDataset,
    _NUMERIC_COLS,
    add_base_features,
    build_char_vocab,
    build_vocab,
    build_windows_for_df,
    collate_batch,
    fetch_usgs_events,
    plot_window_grid,
    split_events_temporal,
    split_windows_index,
    standardize_numeric,
)
from geohash.inference.artifacts import save_model_config_artifact, save_preprocess_artifact
from geohash.model import NextMagnitudeLSTM
from geohash.store import RunStore
from geohash.training import TrainingVisualizer, train
from geohash.training.baselines import beats_baseline, run_baselines
from geohash.training.evaluator import collect_predictions, evaluate_full
from geohash.utils import set_seed

logger = logging.getLogger(__name__)


def _vocab_size(config: RunConfig, stoi: dict, char_stoi: dict | None) -> int:
    if config.geohash.encoding == "hierarchical":
        return len(char_stoi) if char_stoi else 34
    return len(stoi)


@click.command(name="train")
@click.option("--experiment-name", default=None, help="Experiment name")
@click.option("--epochs", type=int, default=12, help="Number of epochs")
@click.option("--batch-size", type=int, default=64, help="Batch size")
@click.option("--learning-rate", type=float, default=1e-3, help="Learning rate")
@click.option("--hidden-size", type=int, default=64, help="LSTM hidden size")
@click.option("--embedding-dim", type=int, default=16, help="Embedding dimension")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", help="Device")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--min-lat", type=float, default=-0.132, help="Min latitude")
@click.option("--max-lat", type=float, default=9.796, help="Max latitude")
@click.option("--min-lon", type=float, default=-80.343, help="Min longitude")
@click.option("--max-lon", type=float, default=-72.466, help="Max longitude")
@click.option(
    "--window-mode",
    type=click.Choice(["temporal", "spatial", "hybrid"]),
    default="temporal",
    help="Windowing strategy",
)
@click.option("--spatial-radius-km", type=float, default=50.0, help="Spatial radius km")
@click.option("--temporal-window-days", type=float, default=30.0, help="Temporal lookback days")
@click.option(
    "--split-strategy",
    type=click.Choice(["temporal_event", "window_index"]),
    default="temporal_event",
    help="Train/val/test split strategy",
)
@click.option("--train-ratio", type=float, default=0.7, help="Train event ratio")
@click.option("--val-ratio", type=float, default=0.1, help="Val event ratio")
@click.option("--test-ratio", type=float, default=0.2, help="Test event ratio")
@click.option(
    "--input-mode",
    type=click.Choice(["full", "numeric_only", "geohash_only"]),
    default="full",
    help="Model ablation input mode",
)
@click.option(
    "--geohash-encoding",
    type=click.Choice(["flat", "hierarchical"]),
    default="flat",
    help="Geohash encoding strategy",
)
@click.option("--early-stop-patience", type=int, default=5, help="Early stopping patience (0=off)")
@click.option(
    "--lr-scheduler",
    type=click.Choice(["none", "cosine", "plateau"]),
    default="plateau",
    help="Learning rate scheduler",
)
@click.option("--gradient-clip", type=float, default=1.0, help="Gradient clipping max norm")
def train_cmd(
    experiment_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    embedding_dim: int,
    device: str,
    seed: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    window_mode: str,
    spatial_radius_km: float,
    temporal_window_days: float,
    split_strategy: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    input_mode: str,
    geohash_encoding: str,
    early_stop_patience: int,
    lr_scheduler: str,
    gradient_clip: float,
):
    """Train earthquake magnitude prediction model."""
    logging.basicConfig(level=logging.INFO)

    config = RunConfig(
        experiment={"experiment_name": experiment_name or "geohash_train"},
        training={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": device,
            "early_stopping_patience": early_stop_patience,
            "lr_scheduler": lr_scheduler,
            "gradient_clip": gradient_clip,
            "split_strategy": split_strategy,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        model={
            "hidden_size": hidden_size,
            "embedding_dim": embedding_dim,
            "input_mode": input_mode,
        },
        geohash={
            "encoding": geohash_encoding,
        },
        usgs={
            "min_latitude": min_lat,
            "max_latitude": max_lat,
            "min_longitude": min_lon,
            "max_longitude": max_lon,
        },
        window={
            "mode": window_mode,
            "spatial_radius_km": spatial_radius_km,
            "temporal_window_days": temporal_window_days,
        },
    )

    set_seed(config.training.seed)

    click.echo("Fetching USGS data...")
    df = fetch_usgs_events(
        min_latitude=config.usgs.min_latitude,
        max_latitude=config.usgs.max_latitude,
        min_longitude=config.usgs.min_longitude,
        max_longitude=config.usgs.max_longitude,
        start_time=config.usgs.start_time,
        end_time=config.usgs.end_time,
        min_magnitude=config.usgs.min_magnitude,
        limit=config.usgs.limit,
    )
    click.echo(f"✓ Retrieved {len(df)} events")

    click.echo("Adding base features...")
    df = add_base_features(df, config.geohash.precision)

    char_stoi = build_char_vocab() if config.geohash.encoding == "hierarchical" else None

    if config.training.split_strategy == "temporal_event":
        train_df, val_df, test_df = split_events_temporal(
            df,
            train_ratio=config.training.train_ratio,
            val_ratio=config.training.val_ratio,
            test_ratio=config.training.test_ratio,
        )
        stoi = build_vocab(train_df["geohash"].tolist(), include_unk=True)
        click.echo(f"✓ Geohash vocab size: {len(stoi)} (train events only)")

        click.echo(f"Building {config.window.mode} windows per split...")
        train_samples = build_windows_for_df(train_df, stoi, config.window, config.geohash, char_stoi)
        val_samples = build_windows_for_df(val_df, stoi, config.window, config.geohash, char_stoi)
        test_samples = build_windows_for_df(test_df, stoi, config.window, config.geohash, char_stoi)
        click.echo(
            f"✓ Windows: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )
    else:
        click.echo("WARNING: window_index split leaks overlapping contexts — debug only.")
        stoi = build_vocab(df["geohash"].tolist(), include_unk=True)
        all_samples = build_windows_for_df(df, stoi, config.window, config.geohash, char_stoi)
        train_samples, rest = split_windows_index(all_samples, config.training.train_split)
        val_count = max(1, len(rest) // 2)
        val_samples = rest[:val_count]
        test_samples = rest[val_count:]
        click.echo(f"✓ Legacy split windows: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    run_dir = config.get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    spot_samples = train_samples[:16] if len(train_samples) >= 16 else train_samples
    if spot_samples:
        click.echo("Creating window spot check...")
        spot_fig = plot_window_grid(spot_samples, num_samples=min(16, len(spot_samples)), seed=config.training.seed)
        spot_check_path = run_dir / "window_spot_check.png"
        spot_fig.savefig(spot_check_path, dpi=150, bbox_inches="tight")
        click.echo(f"✓ Saved window spot check to {spot_check_path}")

    mean, std = standardize_numeric(train_samples, val_samples, test_samples)

    train_loader = DataLoader(
        QuakeWindowDataset(train_samples),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        QuakeWindowDataset(val_samples),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        QuakeWindowDataset(test_samples),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    num_numeric = train_samples[0]["x_num"].shape[1]
    vocab_size = _vocab_size(config, stoi, char_stoi)

    model = NextMagnitudeLSTM(
        vocab_size=vocab_size,
        embedding_dim=config.model.embedding_dim,
        num_numeric=num_numeric,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        input_mode=config.model.input_mode,
        encoding=config.geohash.encoding,
        geohash_precision=config.geohash.precision,
        char_vocab_size=len(char_stoi) if char_stoi else 34,
    ).to(config.training.device)

    click.echo(f"Training on {config.training.device}...")
    visualizer = TrainingVisualizer(enabled=True)
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.training.device,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        early_stopping_patience=config.training.early_stopping_patience,
        lr_scheduler=config.training.lr_scheduler,
        lr_patience=config.training.lr_patience,
        gradient_clip=config.training.gradient_clip,
        visualizer=visualizer,
    )

    click.echo("Evaluating on test set...")
    test_metrics = evaluate_full(model, test_loader, config.training.device)
    pred_result = collect_predictions(model, test_loader, config.training.device)
    click.echo(f"  RMSE = {test_metrics['rmse']:.4f}  MAE = {test_metrics['mae']:.4f}  R² = {test_metrics['r2']:.4f}")

    click.echo("Running baselines...")
    baseline_metrics = run_baselines(train_samples, test_samples)
    beat_flags = beats_baseline(test_metrics, baseline_metrics)
    click.echo(f"  Persistence baseline RMSE = {baseline_metrics['persistence']['rmse']:.4f}")
    click.echo(f"  Beats persistence: {beat_flags.get('persistence', False)}")

    save_preprocess_artifact(
        run_dir=run_dir,
        stoi=stoi,
        numeric_mean=mean.tolist(),
        numeric_std=std.tolist(),
        numeric_cols=list(_NUMERIC_COLS),
        geohash_precision=config.geohash.precision,
        encoding=config.geohash.encoding,
        input_mode=config.model.input_mode,
        char_stoi=char_stoi,
    )
    save_model_config_artifact(
        run_dir,
        {
            "vocab_size": vocab_size,
            "embedding_dim": config.model.embedding_dim,
            "num_numeric": num_numeric,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "input_mode": config.model.input_mode,
            "encoding": config.geohash.encoding,
            "geohash_precision": config.geohash.precision,
        },
    )

    click.echo("Saving run...")
    store = RunStore(config.experiment.output_dir)
    store.save_run(
        run_dir=run_dir,
        config=config.model_dump(),
        history=history,
        model=model,
        predictions=pred_result,
        baselines=baseline_metrics,
        beats_baseline=beat_flags,
    )

    click.echo(f"\n✅ Training complete! Run saved to:\n   {run_dir}")
