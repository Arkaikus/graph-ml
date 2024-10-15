import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Model

from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="darkgrid")

from data.data import EarthquakeData
from .model import LSTMModel

logger = logging.getLogger(__name__)


class LSTMTrainable(tune.Trainable):
    @classmethod
    def with_parameters(cls, config, qdata: EarthquakeData) -> "LSTMTrainable":
        return tune.with_parameters(cls, qdata=qdata)(config=config)

    def setup(self, config: dict, qdata: EarthquakeData):
        """takes hyperparameter values in the config argument"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.trial_id)

        batch_size = config["batch_size"]
        sequence_size = config.get("sequence_size")
        test_size = config.get("test_size")
        scaler = config.get("scaler")
        self.max_epochs = config.get("max_epochs", 100)
        self.epoch = 0

        assert sequence_size, "[sequence_size] cannot be None"

        # prepare train test split of data
        (
            (self.x_train, self.y_train),
            (self.x_val, self.y_val),
            (self.x_test, self.y_test),
        ) = qdata.split(sequence_size, test_size, scaler, torch_tensor=False)

        self.train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(batch_size)
        self.val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(batch_size)
        self.test_data = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size)

        hidden_size = config.get("hidden_size", 1)
        self.model: Model = LSTMModel(hidden_size, config.get("num_layers", 1))
        self.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
        self.last_metrics = None

    def step(self):
        if self.epoch >= self.max_epochs and self.last_metrics:
            return self.last_metrics

        # Training phase
        history = self.model.fit(self.train_data, validation_data=self.val_data, epochs=1, verbose=0)
        self.epoch += 1

        mean_loss = history.history["loss"][0]
        val_loss = history.history["val_loss"][0]

        val_r2 = self.validate()

        self.last_metrics = {
            "epoch": self.epoch,
            "loss": mean_loss,
            "val_loss": val_loss,
            "val_r2": val_r2,
            "checkpoint_dir_name": "",
            "done": False,
        }
        return self.last_metrics

    def validate(self):
        val_pred = self.model.predict(self.val_data)
        val_r2 = r2_score(self.y_val, val_pred)
        return val_r2

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """saves checkpoint at the end of training"""
        self.logger.info("Saving model to %s", checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.weights.h5")
        self.model.save_weights(checkpoint_path)

    def load_checkpoint(self, checkpoint: Checkpoint):
        """loads checkpoint at the start of training if checkpoint exists"""
        self.logger.info("Loading checkpoint %s", checkpoint)
        with checkpoint.as_directory() as loaded_checkpoint_dir:
            logger.info("Loading checkpoint %s", loaded_checkpoint_dir)
            checkpoint_path = os.path.join(loaded_checkpoint_dir, "checkpoint.weights.h5")
            self.model.load_weights(checkpoint_path)


def plot_scatter(original, forecast, file_path, name="scatter"):
    """
    Plots a scatter plot of the forecast against the original data with metrics
    this is useful for visualizing the performance of the model
    it uses seaborn to plot the scatter plot with a regression line
    """
    MSE = mean_squared_error(original, forecast)
    MAE = mean_absolute_error(original, forecast)
    R2 = r2_score(original, forecast)
    RMSE = np.sqrt(MSE)
    logger.info("Best trial R2 %s", R2)
    logger.info("Best trial MSE: %s", MSE)
    logger.info("Best trial RMSE: %s", RMSE)
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

    save_to = Path(file_path) / f"{name}.png"
    logger.info("Figure saved to %s", save_to)
    g.figure.savefig(save_to)
    plt.close(g.figure)


def test_result(result: Result, qdata):
    logger.info("Loading testing from config")
    trainable = LSTMTrainable.with_parameters(result.config, qdata)
    best_checkpoint = result.get_best_checkpoint("loss", "min")
    trainable.load_checkpoint(best_checkpoint)

    print(result.metrics_dataframe)

    train_pred = trainable.model.predict(trainable.train_data)
    val_pred = trainable.model.predict(trainable.val_data)
    test_pred = trainable.model.predict(trainable.test_data)

    plot_scatter(trainable.y_train, train_pred, result.path, name="train_scatter")
    plot_scatter(trainable.y_val, val_pred, result.path, name="val_scatter")
    plot_scatter(trainable.y_test, test_pred, result.path, name="test_scatter")

    fig, ax = plt.subplots(figsize=(30, 5))
    ax.plot(trainable.y_test, label="Real")
    ax.plot(test_pred, label="Forecast")
    ax.legend()
    plt.title("Test Data Real vs Forecast")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    save_to = Path(result.path) / "forecast.png"
    logger.info("Figure saved to %s", save_to)
    fig.savefig(save_to)
    plt.close(fig)

    plt.plot(result.metrics_dataframe["loss"], label="loss")
    plt.plot(result.metrics_dataframe["val_loss"], label="val_loss")
    plt.gcf().savefig(Path(result.path) / "loss.png")
    plt.close()
