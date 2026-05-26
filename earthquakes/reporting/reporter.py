"""Centralized Reporter for experiment artifacts and MLflow logging."""

import logging
import requests
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urljoin

if TYPE_CHECKING:
    from ray.tune import ResultGrid

logger = logging.getLogger(__name__)


def safe_join(url, path):
    """Join a URL and a path safely."""
    try:
        result = urlparse(url)
        assert all([result.scheme, result.netloc])
        return urljoin(url, path)
    except ValueError:
        return None


def healthcheck(tracking_uri: str) -> bool:
    """Check if the tracking URI is valid."""
    if url := safe_join(tracking_uri, "health"):
        response = requests.get(url)
        return response.status_code == 200
    return False


class Reporter:
    """Single entry point for experiment artifacts: local output_dir + optional MLflow."""

    def __init__(
        self,
        output_dir: Path | str = "./runs",
        tracking_uri: str = "http://localhost:5000",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_remote = healthcheck(tracking_uri)
        self.tracking_uri = tracking_uri if self.is_remote else "./mlruns"

    def log_artifact(self, local_path: Path | str, artifact_path: str | None = None) -> None:
        """Save file to output_dir and optionally log to MLflow."""
        path = Path(local_path)
        if not path.exists():
            logger.warning("Artifact path does not exist: %s", path)
            return
        if self.is_remote:
            try:
                import mlflow

                mlflow.log_artifact(str(path), artifact_path=artifact_path)
            except Exception as e:
                logger.warning("Failed to log artifact to MLflow: %s", e)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to MLflow if tracking URI is set."""
        if self.is_remote:
            try:
                import mlflow

                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning("Failed to log metrics to MLflow: %s", e)

    def log_experiment_results(
        self,
        results: "ResultGrid",
        metric: str,
        mode: str,
        qdata,
        task: str = "classification",
        networkx: bool = False,
    ) -> None:
        """Build results DataFrame, save .tex + .csv, optionally log to MLflow."""
        results_df = results.get_dataframe()
        sort_by = metric if metric in results_df.columns else ("test_loss" if "test_loss" in results_df.columns else results_df.columns[0])
        results_df = results_df.sort_values(by=sort_by, ascending=(mode == "min"))

        def _fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else x

        results_df = results_df.apply(lambda col: col.map(_fmt) if col.dtype.kind == "f" else col)
        extra_columns = ["accuracy", "config/quantiles", "config/loss_type"] if task == "classification" else []

        if networkx:
            extra_columns += [
                "config/network_features",
                "config/network_lookback",
                "config/node_size",
            ]

        base_cols = [
            "loss",
            "mean_loss",
            "test_loss",
            "config/lookback",
            "config/test_size",
            "config/batch_size",
            "config/hidden_size",
            "config/lstm_layers",
            "config/lr",
            "config/max_epochs",
            "config/dropout",
        ]
        select_cols = [c for c in base_cols + extra_columns if c in results_df.columns]
        results_df = results_df[select_cols].rename(columns={c: c.replace("config/", "").replace("_", " ") for c in results_df.columns})
        if "network features" in results_df.columns:
            results_df["network features"] = results_df["network features"].apply(lambda x: ",".join(x) if isinstance(x, (list, tuple)) else x)

        latex_table = results_df.to_latex(index=False)
        save_to = self.output_dir / qdata.hash
        save_to.mkdir(parents=True, exist_ok=True)
        experiment_name = Path(results.experiment_path).stem

        tex_path = save_to / f"{experiment_name}_results_table.tex"
        csv_path = save_to / f"{experiment_name}_results_table.csv"
        with open(tex_path, "w") as f:
            f.write(latex_table)
        results_df.to_csv(csv_path, index=False)
        logger.info("Saved .tex table to %s", tex_path)
        logger.info("Saved .csv table to %s", csv_path)

        if self.tracking_uri and self._is_remote_mlflow():
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment("lstm")
                with mlflow.start_run(run_name=f"experiment_{experiment_name}"):
                    mlflow.log_artifact(str(tex_path), artifact_path="results")
                    mlflow.log_artifact(str(csv_path), artifact_path="results")
            except Exception as e:
                logger.warning("Failed to log experiment results to MLflow: %s", e)

    def subdir(self, *parts: str) -> Path:
        """Return output_dir / parts for experiment subdirectories."""
        path = self.output_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
