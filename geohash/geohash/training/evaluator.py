"""Evaluation utilities."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, float, float]:
    """
    Evaluate model on dataset.

    Parameters
    ----------
    model : nn.Module
        LSTM model.
    loader : DataLoader
        Data loader.
    device : str
        Device to use.

    Returns
    -------
    tuple
        (loss, rmse, mae) metrics.
    """
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

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)

    rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    loss_mean = float(np.mean(losses))

    return loss_mean, rmse, mae
