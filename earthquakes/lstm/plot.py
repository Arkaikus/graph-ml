import logging
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

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


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def plot_roc_auc(all_labels, all_preds, quantiles, save_to):
    try:
        labels = list(range(quantiles))
        if quantiles == 2:
            all_labels_bin = np.vstack((all_labels == 0, all_labels == 1), dtype=int).T
            all_preds_bin = np.vstack((all_preds == 0, all_preds == 1), dtype=int).T
        else:
            all_labels_bin = label_binarize(all_labels, classes=labels)
            all_preds_bin = label_binarize(all_preds, classes=labels)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(quantiles):
            fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
            roc_auc[i] = roc_auc_score(all_labels_bin[:, i], all_preds_bin[:, i])

        # Plot ROC curve for each class
        plt.figure(figsize=(8, 6))
        colors = cycle(["blue", "red", "green"])
        for i, color in zip(range(quantiles), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (area = {roc_auc[i]:0.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC AUC Curve for {quantiles} classes")
        plt.legend(loc="lower right")
        plt.gcf().savefig(save_to)
        plt.close()
    except:
        logger.exception("Error plotting ROC AUC curve")
