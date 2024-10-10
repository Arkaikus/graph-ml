import logging

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


def metrics(original, forecast):
    """
    Returns the metrics for the forecasted data
    """
    MSE = mean_squared_error(original, forecast)
    MAE = mean_absolute_error(original, forecast)
    R2 = r2_score(original, forecast)
    MAPE = mean_absolute_percentage_error(original, forecast)
    RMSE = np.sqrt(MSE)
    return {
        "MSE": MSE,
        "MAE": MAE,
        "R2": R2,
        "MAPE": MAPE,
        "RMSE": RMSE,
    }


def plot_scatter(original, forecast, save_to):
    """
    Plots a scatter plot of the forecast against the original data with metrics
    this is useful for visualizing the performance of the model
    it uses seaborn to plot the scatter plot with a regression line
    """

    MSE = mean_squared_error(original, forecast)
    MAE = mean_absolute_error(original, forecast)
    R2 = r2_score(original, forecast)
    MAPE = mean_absolute_percentage_error(original, forecast)
    RMSE = np.sqrt(MSE)
    logger.info("Best trial R2 %s", R2)
    logger.info("Best trial MSE: %s", MSE)
    logger.info("Best trial RMSE: %s", RMSE)
    logger.info("Best trial MAE: %s", MAE)
    logger.info("Best trial MAPE: %s", MAPE)

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
        f"R2: {R2:.2f}\nMSE: {MSE:.2f}\nMAE: {MAE:.2f}\nMAPE:{MAPE:.2f}",
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
