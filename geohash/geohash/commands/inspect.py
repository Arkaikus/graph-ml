"""Inspect commands for run management."""

import json
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import matplotlib.pyplot as plt

from geohash.store import RunStore


@click.group(name="inspect")
def inspect_group():
    """Inspect training runs."""
    pass


@inspect_group.command(name="list-runs")
def list_runs_cmd():
    """List all saved training runs."""
    store = RunStore(Path.home() / ".geohash-runs")
    runs = store.list_runs()

    if not runs:
        click.echo("No runs found.")
        return

    click.echo(f"\n{'Run Name':<50} {'RMSE':<10} {'MAE':<10} {'Loss':<10}")
    click.echo("=" * 80)

    for run in runs:
        rmse = f"{run['final_rmse']:.4f}" if run['final_rmse'] is not None else "N/A"
        mae = f"{run['final_mae']:.4f}" if run['final_mae'] is not None else "N/A"
        loss = f"{run['final_test_loss']:.4f}" if run['final_test_loss'] is not None else "N/A"
        click.echo(f"{run['name']:<50} {rmse:<10} {mae:<10} {loss:<10}")

    click.echo(f"\nTotal runs: {len(runs)}")


@inspect_group.command(name="inspect-run")
@click.argument("run_name")
def inspect_run_cmd(run_name: str):
    """Inspect detailed metrics for a specific run."""
    store = RunStore(Path.home() / ".geohash-runs")
    run_dir = store.get_run_by_name(run_name)

    if run_dir is None:
        click.echo(f"Run '{run_name}' not found.")
        return

    run_data = store.load_run(run_dir)

    # Display config
    if "config" in run_data:
        click.echo(f"\n📋 Configuration ({run_dir.name}):")
        click.echo("-" * 60)
        config = run_data["config"]
        if "training" in config:
            training = config["training"]
            click.echo(f"  Epochs: {training.get('epochs')}")
            click.echo(f"  Batch size: {training.get('batch_size')}")
            click.echo(f"  Learning rate: {training.get('learning_rate')}")
        if "model" in config:
            model = config["model"]
            click.echo(f"  Embedding dim: {model.get('embedding_dim')}")
            click.echo(f"  Hidden size: {model.get('hidden_size')}")

    # Display metrics
    if "metrics" in run_data:
        click.echo(f"\n📊 Final Metrics:")
        click.echo("-" * 60)
        metrics = run_data["metrics"]
        click.echo(f"  Train Loss: {metrics.get('train_loss', 'N/A')}")
        click.echo(f"  Test Loss: {metrics.get('final_test_loss'):.4f}")
        click.echo(f"  RMSE: {metrics.get('final_rmse'):.4f}")
        click.echo(f"  MAE: {metrics.get('final_mae'):.4f}")

    # Display predictions summary
    if "predictions_df" in run_data:
        df = run_data["predictions_df"]
        click.echo(f"\n📈 Predictions ({len(df)} test samples):")
        click.echo("-" * 60)
        click.echo(df.head(10).to_string(index=False))
        if len(df) > 10:
            click.echo(f"  ... and {len(df) - 10} more")

    # Display artifact paths
    click.echo(f"\n📁 Artifacts:")
    click.echo("-" * 60)
    for artifact in ["config.json", "metrics.json", "predictions.csv", "model_final.pt", "training_curves.png"]:
        artifact_path = run_dir / artifact
        if artifact_path.exists():
            click.echo(f"  ✓ {artifact}")
        else:
            click.echo(f"  ✗ {artifact}")


@inspect_group.command(name="compare-runs")
@click.argument("run_names", nargs=-1, required=True)
def compare_runs_cmd(run_names: tuple[str, ...]):
    """Compare metrics across multiple runs."""
    if len(run_names) < 2:
        click.echo("Please provide at least 2 run names to compare.")
        return

    store = RunStore(Path.home() / ".geohash-runs")
    runs_data = []

    for run_name in run_names:
        run_dir = store.get_run_by_name(run_name)
        if run_dir is None:
            click.echo(f"⚠ Run '{run_name}' not found, skipping.")
            continue

        run_data = store.load_run(run_dir)
        if "metrics" in run_data:
            metrics = run_data["metrics"]
            runs_data.append({
                "name": run_dir.name,
                "rmse": metrics.get("final_rmse"),
                "mae": metrics.get("final_mae"),
                "test_loss": metrics.get("final_test_loss"),
            })

    if not runs_data:
        click.echo("No valid runs to compare.")
        return

    # Create comparison table
    df = pd.DataFrame(runs_data)
    click.echo("\n📊 Comparison:")
    click.echo("=" * 100)
    click.echo(df.to_string(index=False))

    # Find best run
    best_rmse_idx = df["rmse"].idxmin()
    click.echo(f"\n✨ Best RMSE: {df.iloc[best_rmse_idx]['name']} ({df.iloc[best_rmse_idx]['rmse']:.4f})")


@inspect_group.command(name="plot-run")
@click.argument("run_name")
def plot_run_cmd(run_name: str):
    """Display training curves plot for a run."""
    store = RunStore(Path.home() / ".geohash-runs")
    run_dir = store.get_run_by_name(run_name)

    if run_dir is None:
        click.echo(f"Run '{run_name}' not found.")
        return

    plot_path = run_dir / "training_curves.png"
    if not plot_path.exists():
        click.echo(f"Plot not found at {plot_path}")
        return

    click.echo(f"Opening plot: {plot_path}")
    try:
        import subprocess
        subprocess.Popen(["xdg-open", str(plot_path)])
    except Exception as e:
        click.echo(f"Could not open image viewer: {e}")
        click.echo(f"Plot saved at: {plot_path}")
