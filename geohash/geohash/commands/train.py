"""Train command."""

import logging
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from geohash.config import RunConfig
from geohash.data import (
    add_features,
    build_vocab,
    collate_batch,
    fetch_usgs_events,
    make_windows,
    standardize_numeric,
    QuakeWindowDataset,
)
from geohash.model import NextMagnitudeLSTM
from geohash.store import RunStore
from geohash.training import TrainingVisualizer, evaluate, train
from geohash.utils import set_seed

logger = logging.getLogger(__name__)

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
):
    """Train earthquake magnitude prediction model."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create config
    config = RunConfig(
        experiment={"experiment_name": experiment_name or f"geohash_train"},
        training={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": device,
        },
        model={
            "hidden_size": hidden_size,
            "embedding_dim": embedding_dim,
        },
        usgs={
            "min_latitude": min_lat,
            "max_latitude": max_lat,
            "min_longitude": min_lon,
            "max_longitude": max_lon,
        },
    )

    set_seed(config.training.seed)

    # Get USGS data
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

    # Feature engineering
    click.echo("Adding features...")
    df = add_features(df, config.geohash.precision)

    stoi = build_vocab(df["geohash"].tolist())
    vocab_size = len(stoi)
    click.echo(f"✓ Geohash vocab size: {vocab_size}")

    # Make windows
    click.echo("Building sliding windows...")
    samples = make_windows(
        df=df,
        stoi=stoi,
        min_len=config.window.min_len,
        max_len=config.window.max_len,
        stride=config.window.stride,
    )
    click.echo(f"✓ Created {len(samples)} training windows")

    # Split and standardize
    split_idx = int(len(samples) * config.training.train_split)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    standardize_numeric(train_samples, test_samples)

    # Create datasets and loaders
    train_ds = QuakeWindowDataset(train_samples)
    test_ds = QuakeWindowDataset(test_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    num_numeric = train_samples[0]["x_num"].shape[1]

    # Create model
    model = NextMagnitudeLSTM(
        vocab_size=vocab_size,
        embedding_dim=config.model.embedding_dim,
        num_numeric=num_numeric,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(config.training.device)

    # Train
    click.echo(f"Training on {config.training.device}...")
    visualizer = TrainingVisualizer(enabled=True)
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=config.training.device,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        visualizer=visualizer,
    )

    # Evaluate on test set and get predictions
    click.echo("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for gh_ids, x_num, lengths, y in test_loader:
            gh_ids = gh_ids.to(config.training.device)
            x_num = x_num.to(config.training.device)
            lengths = lengths.to(config.training.device)

            pred = model(gh_ids, x_num, lengths)
            all_preds.extend(pred.squeeze(1).cpu().numpy().tolist())
            all_targets.extend(y.squeeze(1).cpu().numpy().tolist())

    predictions = {
        "targets": all_targets,
        "predictions": all_preds,
    }

    # Save run
    click.echo("Saving run...")
    run_dir = config.get_run_dir()
    store = RunStore(config.experiment.output_dir)
    store.save_run(
        run_dir=run_dir,
        config=config.model_dump(),
        history=history,
        model=model,
        predictions=predictions,
    )

    click.echo(f"\n✅ Training complete! Run saved to:\n   {run_dir}")
