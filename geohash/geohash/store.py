"""Run storage and management."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class RunStore:
    """Manages saving and loading training runs."""

    def __init__(self, output_dir: Path):
        """
        Initialize store.

        Parameters
        ----------
        output_dir : Path
            Base directory for storing runs (e.g., ~/.geohash-runs).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        history: dict[str, list[float]],
        model: torch.nn.Module,
        predictions: dict[str, np.ndarray],
    ) -> Path:
        """
        Save complete training run artifacts.

        Parameters
        ----------
        run_dir : Path
            Directory for this run.
        config : dict[str, Any]
            Configuration dictionary (from Pydantic model.model_dump()).
        history : dict[str, list[float]]
            Training history with keys: train_loss, test_loss, rmse, mae.
        model : torch.nn.Module
            Trained model.
        predictions : dict[str, np.ndarray]
            Predictions dict with keys: "targets" and "predictions".

        Returns
        -------
        Path
            Path to the saved run directory.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving run to {run_dir}")

        # Save configuration
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            # Convert any non-serializable types
            config_serializable = self._make_serializable(config)
            json.dump(config_serializable, f, indent=2)
        logger.info(f"Saved config to {config_path}")

        # Save metrics
        metrics = {
            "train_loss": history["train_loss"],
            "test_loss": history["test_loss"],
            "rmse": history["rmse"],
            "mae": history["mae"],
            "final_rmse": float(history["rmse"][-1]),
            "final_mae": float(history["mae"][-1]),
            "final_test_loss": float(history["test_loss"][-1]),
            "r2": float(predictions.get("r2", float("nan"))),
        }
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        # Save predictions
        predictions_df = pd.DataFrame({
            "target": predictions["targets"],
            "predicted": predictions["predictions"],
        })
        predictions_path = run_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")

        # Save model
        model_path = run_dir / "model_final.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")

        # Save training curves
        plot_path = run_dir / "training_curves.png"
        self._plot_history(history, plot_path)
        logger.info(f"Saved plot to {plot_path}")

        # Save predictions scatter
        scatter_path = run_dir / "predictions_scatter.png"
        self._plot_predictions_scatter(
            targets=predictions["targets"],
            preds=predictions["predictions"],
            r2=float(predictions.get("r2", float("nan"))),
            rmse=float(history["rmse"][-1]),
            mae=float(history["mae"][-1]),
            output_path=scatter_path,
        )
        logger.info(f"Saved scatter plot to {scatter_path}")

        return run_dir

    def load_run(self, run_dir: Path) -> dict[str, Any]:
        """
        Load run artifacts.

        Parameters
        ----------
        run_dir : Path
            Run directory.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: config, metrics, predictions_df, model_state_dict.
        """
        result: dict[str, Any] = {}

        # Load config
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                result["config"] = json.load(f)

        # Load metrics
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                result["metrics"] = json.load(f)

        # Load predictions
        predictions_path = run_dir / "predictions.csv"
        if predictions_path.exists():
            result["predictions_df"] = pd.read_csv(predictions_path)

        # Load model state
        model_path = run_dir / "model_final.pt"
        if model_path.exists():
            result["model_state_dict"] = torch.load(model_path, map_location="cpu")

        return result

    def list_runs(self) -> list[dict[str, Any]]:
        """
        List all saved runs.

        Returns
        -------
        list[dict[str, Any]]
            List of run metadata sorted by date (newest first).
            Each dict has keys: name, path, timestamp, final_rmse, final_mae.
        """
        runs = []

        for run_dir in sorted(self.output_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)

                # Extract timestamp from directory name (format: {name}-{YYYYMMDD_HHMMSS})
                run_name = run_dir.name
                run_metadata = {
                    "name": run_name,
                    "path": str(run_dir),
                    "final_rmse": metrics.get("final_rmse", None),
                    "final_mae": metrics.get("final_mae", None),
                    "final_test_loss": metrics.get("final_test_loss", None),
                }
                runs.append(run_metadata)

        return runs

    def get_run_by_name(self, experiment_name: str) -> Optional[Path]:
        """
        Find run directory by experiment name (most recent if multiple matches).

        Parameters
        ----------
        experiment_name : str
            Experiment name prefix.

        Returns
        -------
        Path or None
            Path to run directory, or None if not found.
        """
        matches = [
            d for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith(experiment_name)
        ]

        if not matches:
            return None

        # Return most recent
        return sorted(matches, reverse=True)[0]

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert non-JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: RunStore._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [RunStore._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    @staticmethod
    def _plot_history(history: dict[str, list[float]], output_path: Path) -> None:
        """
        Plot training history.

        Parameters
        ----------
        history : dict[str, list[float]]
            History dict with train_loss, test_loss, rmse, mae.
        output_path : Path
            Path to save PNG.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        epochs = np.arange(1, len(history["train_loss"]) + 1)

        # Loss
        axes[0, 0].plot(epochs, history["train_loss"], label="Train", marker="o")
        axes[0, 0].plot(epochs, history["test_loss"], label="Test", marker="s")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss (MSE)")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # RMSE
        axes[0, 1].plot(epochs, history["rmse"], marker="o", color="orange")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].set_title("Test RMSE")
        axes[0, 1].grid(True, alpha=0.3)

        # MAE
        axes[1, 0].plot(epochs, history["mae"], marker="o", color="green")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MAE")
        axes[1, 0].set_title("Test MAE")
        axes[1, 0].grid(True, alpha=0.3)

        # Summary stats
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = (
            f"Final Metrics\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Train Loss: {history['train_loss'][-1]:.4f}\n"
            f"Test Loss: {history['test_loss'][-1]:.4f}\n"
            f"RMSE: {history['rmse'][-1]:.4f}\n"
            f"MAE: {history['mae'][-1]:.4f}\n"
            f"Epochs: {len(history['train_loss'])}"
        )
        ax.text(0.1, 0.5, summary_text, fontfamily="monospace", fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _plot_predictions_scatter(
        targets: list[float],
        preds: list[float],
        r2: float,
        rmse: float,
        mae: float,
        output_path: Path,
    ) -> None:
        """
        Plot predicted vs actual magnitudes scatter for the test set.

        Parameters
        ----------
        targets : list[float]
            Ground-truth magnitudes.
        preds : list[float]
            Model predictions.
        r2 : float
            R² coefficient of determination.
        rmse : float
            Root mean squared error.
        mae : float
            Mean absolute error.
        output_path : Path
            Path to save PNG.
        """
        t = np.array(targets)
        p = np.array(preds)

        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        pad = (hi - lo) * 0.05
        diag = [lo - pad, hi + pad]

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.scatter(t, p, alpha=0.35, s=18, color="steelblue", edgecolors="none")
        ax.plot(diag, diag, color="tomato", linewidth=1.2, linestyle="--", label="Perfect fit")

        annotation = f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nn = {len(t)}"
        ax.text(
            0.04, 0.96, annotation,
            transform=ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Actual magnitude", fontsize=12)
        ax.set_ylabel("Predicted magnitude", fontsize=12)
        ax.set_title("Test set: predicted vs actual magnitude", fontsize=13)
        ax.set_xlim(diag)
        ax.set_ylim(diag)
        ax.set_aspect("equal")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
