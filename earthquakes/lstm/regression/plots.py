"""Regression-specific plotting utilities."""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)


def plot_scatter(original, forecast, save_to):
    """
    Plots a scatter plot of the forecast against the original data with metrics.
    Uses seaborn to plot the scatter plot with a regression line.
    """
    mse = mean_squared_error(original, forecast)
    mae = mean_absolute_error(original, forecast)
    r2 = r2_score(original, forecast)
    mape = mean_absolute_percentage_error(original, forecast)
    rmse = np.sqrt(mse)
    logger.info("Best trial R2 %s", r2)
    logger.info("Best trial MSE: %s", mse)
    logger.info("Best trial RMSE: %s", rmse)
    logger.info("Best trial MAE: %s", mae)
    logger.info("Best trial MAPE: %s", mape)

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

    plt.figtext(
        0.15,
        0.70,
        f"R2: {r2:.2f}\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nMAPE:{mape:.2f}",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=12,
    )

    logger.info("Figure saved to %s", save_to)
    g.figure.savefig(save_to)
    plt.close(g.figure)


def plot_timeseries(original, forecast, target: str, save_to):
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.plot(original, label="Real")
    ax.plot(forecast, label="Forecast")
    ax.legend()
    plt.title("Test Data Real vs Forecast")
    plt.xlabel("Time")
    plt.ylabel(target.capitalize())

    logger.info("Figure saved to %s", save_to)
    fig.savefig(save_to)
    plt.close(fig)
