"""Training loop for LSTM model."""

import copy
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from geohash.training.visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str,
    epochs: int,
    lr_patience: int,
) -> Optional[object]:
    if lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=lr_patience, factor=0.5)
    return None


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int = 5,
    lr_scheduler: str = "plateau",
    lr_patience: int = 3,
    gradient_clip: float = 1.0,
    visualizer: Optional["TrainingVisualizer"] = None,
) -> dict[str, list[float]]:
    """
    Train model with validation-based early stopping.

    Parameters
    ----------
    val_loader : DataLoader
        Validation loader used for early stopping and LR scheduling.
    """
    from .evaluator import evaluate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = _build_scheduler(optimizer, lr_scheduler, epochs, lr_patience)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "rmse": [],
        "mae": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    logger.info(
        "Starting training on %s for up to %d epochs (early_stop_patience=%d, lr_scheduler=%s, grad_clip=%.2f)",
        device,
        epochs,
        early_stopping_patience,
        lr_scheduler,
        gradient_clip,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for gh_ids, x_num, lengths, y in train_loader:
            gh_ids = gh_ids.to(device)
            x_num = x_num.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(gh_ids, x_num, lengths)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, rmse, mae = evaluate(model, val_loader, device)
        train_loss_mean = float(np.mean(train_losses))
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_loss_mean)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(val_loss)  # backward compat alias
        history["rmse"].append(rmse)
        history["mae"].append(mae)
        history["lr"].append(current_lr)

        if scheduler is not None:
            if lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            logger.debug("New best val loss: %.4f (epoch %d)", best_val_loss, epoch)
        else:
            patience_counter += 1

        if visualizer is not None:
            visualizer.update(history, epoch, epochs)

        logger.info(
            "Epoch %02d | train=%.4f | val=%.4f | rmse=%.4f | mae=%.4f | lr=%.2e",
            epoch,
            train_loss_mean,
            val_loss,
            rmse,
            mae,
            current_lr,
        )

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (best was epoch %d, val_loss=%.4f).",
                epoch,
                best_epoch,
                best_val_loss,
            )
            break

    if visualizer is not None:
        visualizer.close()

    model.load_state_dict(best_state_dict)
    logger.info(
        "Restored model weights from best epoch %d (val_loss=%.4f).",
        best_epoch,
        best_val_loss,
    )

    return history
