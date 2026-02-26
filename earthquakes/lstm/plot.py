"""Plotting utilities. Generic helpers and re-exports from task-specific modules."""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Re-export task-specific plots for backward compatibility
from lstm.classification.plots import plot_confusion_matrix, plot_roc_auc
from lstm.regression.plots import plot_scatter, plot_timeseries
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)

__all__ = [
    "metrics",
    "plot_analysis",
    "plot_confusion_matrix",
    "plot_roc_auc",
    "plot_scatter",
    "plot_timeseries",
]


def metrics(original, forecast):
    """Returns the metrics for the forecasted data."""
    mse = mean_squared_error(original, forecast)
    mae = mean_absolute_error(original, forecast)
    r2 = r2_score(original, forecast)
    mape = mean_absolute_percentage_error(original, forecast)
    rmse = np.sqrt(mse)
    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "RMSE": rmse,
    }


def plot_analysis(data: pd.DataFrame, features, target, save_to):
    save_to.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(data[target], bins=50, kde=True)
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.savefig(save_to / f"distribution_{target}.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.pairplot(data[features])
    plt.title("Pairplot of Features and Target")
    plt.savefig(save_to / "pairplot_features_target.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    correlation_matrix = data[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig(save_to / "correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    spearman_corr = data[features].corr(method="spearman")
    sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman Correlation Heatmap")
    plt.savefig(save_to / "spearman_correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    kendall_corr = data[features].corr(method="kendall")
    sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Kendall Correlation Heatmap")
    plt.savefig(save_to / "kendall_correlation_heatmap.png")
    plt.close()
