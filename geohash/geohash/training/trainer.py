"""Training loop for LSTM model."""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
) -> dict[str, list[float]]:
    """
    Train model for specified epochs.

    Parameters
    ----------
    model : nn.Module
        LSTM model to train.
    train_loader : DataLoader
        Training data loader.
    test_loader : DataLoader
        Test data loader for evaluation.
    device : str
        Device to use ("cpu" or "cuda").
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for optimizer.

    Returns
    -------
    dict[str, list[float]]
        History with keys: train_loss, test_loss, rmse, mae.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_loss": [],
        "rmse": [],
        "mae": [],
    }

    logger.info(f"Starting training on {device} for {epochs} epochs")

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

        # Evaluate on test set
        from .evaluator import evaluate
        test_loss, rmse, mae = evaluate(model, test_loader, device)

        train_loss_mean = float(np.mean(train_losses))
        history["train_loss"].append(train_loss_mean)
        history["test_loss"].append(test_loss)
        history["rmse"].append(rmse)
        history["mae"].append(mae)

        logger.info(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss_mean:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"rmse={rmse:.4f} | mae={mae:.4f}"
        )

    return history
