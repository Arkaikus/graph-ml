"""Classification-specific plotting utilities."""

import logging
from itertools import cycle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)


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

        plt.figure(figsize=(8, 6))
        colors = cycle(["blue", "red", "green"])
        for i, color in zip(range(quantiles), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (area = {roc_auc[i]:0.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC AUC Curve for {quantiles} classes")
        plt.legend(loc="lower right")
        plt.gcf().savefig(save_to)
        plt.close()
    except Exception:
        logger.exception("Error plotting ROC AUC curve")
