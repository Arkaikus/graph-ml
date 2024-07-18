import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ray import train
from ray.air import Result
from ray.train import Checkpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

sns.set_theme(style="darkgrid")


from .dataset import create_sequences
from .model import LSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_model(config, feature_size):
    """returns an LSTMModel instance from a config dict and a feature_size"""
    hidden_size = config["hidden_size"]
    lstm_layers = config["layers"]
    dropout = config["dropout"]

    logger.info("Preparing LSTModel and optimizer")
    return LSTMModel(feature_size, hidden_size, lstm_layers, dropout).to(device)


def load_checkpoint(checkpoint: Checkpoint, model: LSTMModel, optimizer: Adam) -> dict:
    """loads the state dict for model and optimizer, returns the current epoch training metrics"""
    with checkpoint.as_directory() as checkpoint_dir:
        logger.info("Loading model state from checkpoint %s", checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        metrics, model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        return metrics


def train_fn(config: dict, data, target):
    logging.basicConfig(level=logging.INFO)
    ctx = train.get_context()
    logger = logging.getLogger(ctx.get_trial_name())

    sequence_size = config["sequence_size"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["lr"]
    epoch_start = 0
    epoch_loss = 0

    assert target, "[target] cannot be None"
    assert sequence_size, "[sequence_size] cannot be None"

    # prepare train test split of data
    logger.info("Creating sequences")
    train_dataset = create_sequences(data, target, sequence_size, test_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(config, data.shape[1])
    mseloss = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # handle checkpoitn
    checkpoint: Checkpoint = train.get_checkpoint()
    if checkpoint:
        metrics = load_checkpoint(checkpoint, model, optimizer)
        epoch_start = metrics["epoch"]
        epoch_loss = metrics["loss"]

    model.train()
    for epoch in range(epoch_start, epochs + 1):
        logger.info("Training epoch %s", epoch)

        for inputs, outputs in tqdm(train_dataloader):
            forecast = model(inputs.to(torch.float32).to(device))
            loss = mseloss(forecast, outputs.to(torch.float32).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(train_dataloader)
        logger.info("epoch: %s loss: %s mean_loss %s", epoch, epoch_loss, mean_loss)

        # Create the checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint_path = os.path.join(temp_checkpoint_dir, f"checkpoint.pth")
            metrics = {"epoch": epoch, "loss": epoch_loss, "mean_loss": mean_loss}
            state = (epoch, model.state_dict(), optimizer.state_dict())
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            torch.save(state, checkpoint_path)
            train.report(metrics, checkpoint=checkpoint)


def test_fn(config: dict, data: pd.DataFrame, target: str, checkpoint: Checkpoint):
    # prepare train test split of data
    sequence_size = config["sequence_size"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    learning_rate = config["lr"]

    test_dataset = create_sequences(data, target, sequence_size, test_size, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(config, data.shape[1])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    metrics = load_checkpoint(checkpoint, model, optimizer)

    original = []
    forecast = []

    logger.info("Loaded model %s", model)
    logger.info("Latest metrics %s", metrics)

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            inputs, outputs = data
            original.append(outputs)
            model_output = model(inputs.to(torch.float32).to(device))
            forecast.append(model_output.cpu())

    return np.vstack(original), np.vstack(forecast)


def test_result(result: Result, data: pd.DataFrame, scalers: dict, target: str):
    logger.info("Loading testing from config")
    best_checkpoint = result.get_best_checkpoint("loss", "min")

    print(result.metrics_dataframe)

    original, forecast = test_fn(result.config, data, target, best_checkpoint)
    target_scaler = scalers.get(target)
    if target_scaler:
        original = target_scaler.inverse_transform(original)
        forecast = target_scaler.inverse_transform(forecast)

    R2 = r2_score(original.squeeze(), forecast.squeeze())
    MSE = mean_squared_error(original.squeeze(), forecast.squeeze())
    MAE = mean_absolute_error(original.squeeze(), forecast.squeeze())
    logger.info("Best trial R2 %s", R2)
    logger.info("Best trial MSE: %s", MSE)
    logger.info("Best trial MAE: %s", MAE)

    hstack = np.hstack((original, forecast))
    logger.info("Hstack %s", hstack.shape)

    g = sns.jointplot(
        x="Real",
        y="Forecast",
        data=pd.DataFrame(hstack, columns=["Real", "Forecast"]),
        kind="reg",
        truncate=False,
        color="m",
        height=7,
    )

    # Add metrics to the plot
    plt.figtext(
        0.15,
        0.70,
        f"R2: {R2:.2f}\nMSE: {MSE:.2f}\nMAE: {MAE:.2f}",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=12,
    )

    save_to = Path(result.path) / "forecast.png"
    logger.info("Figure saved to %s", save_to)
    g.figure.savefig(save_to)
