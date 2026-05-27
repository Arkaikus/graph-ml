"""Evaluation utilities."""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, loss: float) -> dict[str, float]:
    """Compute regression metrics from predictions and targets."""
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = float(np.sum((targets - preds) ** 2))
    t_mean = float(np.mean(targets))
    ss_tot = float(np.sum((targets - t_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"loss": float(loss), "rmse": rmse, "mae": mae, "r2": r2}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, float, float]:
    """
    Evaluate model on dataset.

    Returns
    -------
    tuple
        (loss, rmse, mae) metrics.
    """
    metrics = evaluate_full(model, loader, device)
    return metrics["loss"], metrics["rmse"], metrics["mae"]


def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate model and return full metric dict including R²."""
    model.eval()
    preds: list[float] = []
    targets: list[float] = []
    losses: list[float] = []
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

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    loss_mean = float(np.mean(losses))
    return compute_metrics(preds_arr, targets_arr, loss_mean)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, Any]:
    """Run inference and return targets, predictions, and metrics."""
    model.eval()
    preds: list[float] = []
    targets: list[float] = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for gh_ids, x_num, lengths, y in loader:
            gh_ids = gh_ids.to(device)
            x_num = x_num.to(device)
            lengths = lengths.to(device)

            pred = model(gh_ids, x_num, lengths)
            preds.extend(pred.squeeze(1).cpu().numpy().tolist())
            targets.extend(y.squeeze(1).cpu().numpy().tolist())

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    loss = float(
        criterion(
            torch.tensor(preds_arr).unsqueeze(1),
            torch.tensor(targets_arr).unsqueeze(1),
        ).item()
    )
    metrics = compute_metrics(preds_arr, targets_arr, loss)
    return {
        "targets": targets,
        "predictions": preds,
        **metrics,
    }
