"""Baseline predictors for magnitude forecasting."""

from typing import Any

import numpy as np
import torch


def _metrics_from_arrays(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = float(np.sum((targets - preds) ** 2))
    t_mean = float(np.mean(targets))
    ss_tot = float(np.sum((targets - t_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    loss = float(np.mean((preds - targets) ** 2))
    return {"loss": loss, "rmse": rmse, "mae": mae, "r2": r2}


def _extract_targets_and_last_row_features(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract targets, last magnitudes, and last-timestep numeric features.

    Windows vary in length, so we use the final row of each window (6 fixed
    features) rather than flattening the full sequence.
    """
    targets = []
    last_mags = []
    last_row_feats = []

    for s in samples:
        x = s["x_num"]
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
        else:
            x_np = np.asarray(x)
        targets.append(float(s["y"].item() if isinstance(s["y"], torch.Tensor) else s["y"]))
        last_mags.append(float(x_np[-1, 0]))  # magnitude is first numeric col
        last_row_feats.append(x_np[-1].astype(np.float64))

    return (
        np.array(targets, dtype=np.float64),
        np.array(last_mags, dtype=np.float64),
        np.vstack(last_row_feats),
    )


def run_baselines(
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Compute baseline metrics on test samples.

    Returns
    -------
    dict
        Keys: mean, persistence, linear_numeric — each with loss, rmse, mae, r2.
    """
    if not test_samples:
        raise ValueError("test_samples must not be empty")

    train_targets, _, _ = _extract_targets_and_last_row_features(train_samples)
    test_targets, last_mags, last_row_feats = _extract_targets_and_last_row_features(test_samples)

    train_mean = float(np.mean(train_targets))

    mean_preds = np.full_like(test_targets, train_mean)
    persistence_preds = last_mags.copy()

    # Linear least squares on last-timestep numeric features (fixed width per window)
    _, _, train_feats = _extract_targets_and_last_row_features(train_samples)
    ones = np.ones((len(train_feats), 1))
    x_train = np.hstack([ones, train_feats])
    coef, _, _, _ = np.linalg.lstsq(x_train, train_targets, rcond=None)
    x_test = np.hstack([np.ones((len(last_row_feats), 1)), last_row_feats])
    linear_preds = x_test @ coef

    return {
        "mean": _metrics_from_arrays(mean_preds, test_targets),
        "persistence": _metrics_from_arrays(persistence_preds, test_targets),
        "linear_numeric": _metrics_from_arrays(linear_preds, test_targets),
    }


def beats_baseline(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
) -> dict[str, bool]:
    """True if model RMSE is lower than each baseline."""
    model_rmse = model_metrics["rmse"]
    return {name: model_rmse < metrics["rmse"] for name, metrics in baseline_metrics.items()}
