import logging
import os
import pdb
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sns.set_theme(style="darkgrid")


from data.data import EarthquakeData

from ..model import LSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_checkpoint(checkpoint: Checkpoint, model: LSTMModel, optimizer: Adam) -> dict:
    """loads the state dict for model and optimizer, returns the current epoch training metrics"""
    with checkpoint.as_directory() as checkpoint_dir:
        logger.info("Loading model state from checkpoint %s", checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        metrics, model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        return metrics


def train_fn(config: dict, qdata: EarthquakeData):
    sequence_size = config["sequence_size"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["lr"]
    epoch_start = 0

    assert sequence_size, "[sequence_size] cannot be None"

    # prepare train test split of data
    logging.info("Creating sequences with size %s, test_size %s, batch_size %s", sequence_size, test_size, batch_size)
    x_train, x_test, y_train, y_test = qdata.split(sequence_size, test_size)
    x_train = torch.Tensor(x_train).to(torch.float32)
    x_test = torch.Tensor(x_test).to(torch.float32)
    y_train = torch.Tensor(y_train).to(torch.float32)
    y_test = torch.Tensor(y_test).to(torch.float32)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel.from_config({"feature_size": x_train.shape[2], **config}).to(device)
    mseloss = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # handle checkpoitn
    checkpoint: Checkpoint = train.get_checkpoint()
    if checkpoint:
        metrics = load_checkpoint(checkpoint, model, optimizer)
        epoch_start = metrics["epoch"]

    for epoch in range(epoch_start, epochs + 1):  # loop over the dataset multiple times
        logging.info("Training epoch %s", epoch)
        model.train()
        epoch_loss = 0.0
        for inputs, outputs in train_dataloader:
            inputs, outputs = (
                inputs.to(torch.float32).to(device),
                outputs.to(torch.float32).to(device),
            )
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            forecast = model(inputs)
            loss = mseloss(outputs, forecast)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            test_pred = model(x_test.to(device)).detach().cpu()
            test_loss = mseloss(test_pred, y_test).item()
            test_rmse = np.sqrt(test_loss)
            test_r2 = r2_score(y_test.numpy(), test_pred.numpy())
            metrics = {
                "epoch": epoch,
                "loss": epoch_loss,
                "test_loss": test_loss,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }

            print(metrics)
            # Create the checkpoint.
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint_path = os.path.join(temp_checkpoint_dir, f"checkpoint.pth")

                state = (epoch, model.state_dict(), optimizer.state_dict())
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                torch.save(state, checkpoint_path)
                train.report(metrics, checkpoint=checkpoint)


def test_fn(config: dict, qdata: EarthquakeData, checkpoint: Checkpoint):
    # prepare train test split of data
    sequence_size = config["sequence_size"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    learning_rate = config["lr"]

    _, x_test, _, y_test = qdata.split(sequence_size, test_size, test=True)
    x_test = torch.Tensor(x_test).to(torch.float32)
    y_test = torch.Tensor(y_test).to(torch.float32)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel.from_config({"feature_size": x_test.shape[2], **config})
    optimizer = Adam(model.parameters(), lr=learning_rate)
    metrics = load_checkpoint(checkpoint, model, optimizer)

    original = []
    forecast = []

    logger.info("Loaded model %s", model)
    logger.info("Latest metrics %s", metrics)

    model.eval()
    with torch.no_grad():
        for inputs, outputs in test_dataloader:
            original.append(outputs)
            model_output = model(inputs.to(torch.float32))
            forecast.append(model_output.cpu())

    return np.vstack(original), np.vstack(forecast)


def test_result(result: Result, qdata: EarthquakeData):
    logger.info("Loading testing from config")
    best_checkpoint = result.get_best_checkpoint("loss", "min")

    print(result.metrics_dataframe)

    original, forecast = test_fn(result.config, qdata, best_checkpoint)
    target_scaler = qdata.scalers.get(qdata.target)
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
