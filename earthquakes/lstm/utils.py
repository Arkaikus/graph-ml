import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)


def split_n_parse(string: str, _type: type):
    return [_type(part) for part in string.split(",") if part]


def prompt_experiment():
    ray_results = Path.home() / "ray_results"
    folders = {
        idx: folder
        for idx, folder in enumerate(
            ray_results.glob("*"),
        )
        if folder.is_dir()
        if folder.stem[0].isalpha()
    }
    prompt = "\n".join(f"{idx}) {folder.stem}" for idx, folder in folders.items())
    choice = click.prompt(prompt, type=int, default=None)
    assert choice is not None, choice
    return folders.get(choice)


def plot_analysis(data: pd.DataFrame, features, target, save_to: Path):
    save_to.mkdir(parents=True, exist_ok=True)

    # Plotting the distribution of the target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(data[target], bins=50, kde=True)
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.savefig(save_to / f"distribution_{target}.png")
    plt.close()

    # Plotting pairplot for features and target
    plt.figure(figsize=(12, 8))
    sns.pairplot(data[features])
    plt.title("Pairplot of Features and Target")
    plt.savefig(save_to / "pairplot_features_target.png")
    plt.close()

    # Plotting correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig(save_to / "correlation_heatmap.png")
    plt.close()

    # Plotting Spearman correlation heatmap
    plt.figure(figsize=(10, 8))
    spearman_corr = data[features].corr(method="spearman")
    sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman Correlation Heatmap")
    plt.savefig(save_to / "spearman_correlation_heatmap.png")
    plt.close()

    # Plotting Kendall correlation heatmap
    plt.figure(figsize=(10, 8))
    kendall_corr = data[features].corr(method="kendall")
    sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Kendall Correlation Heatmap")
    plt.savefig(save_to / "kendall_correlation_heatmap.png")
    plt.close()

    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.metrics import mean_squared_error, r2_score
    # from sklearn.preprocessing import StandardScaler

    # # Train a simple model for additional plots
    # scaler = StandardScaler()
    # scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    # X = scaled[features][:-1]
    # y = scaled[target][1:]

    # # Standardize the features
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)

    # print(mean_squared_error(predictions, y_test))
    # print(r2_score(predictions, y_test))

    # # Plot time series
    # plt.figure(figsize=(14, 7))
    # plt.plot(y_test.reset_index(drop=True), label="Actual Values", color="blue")
    # plt.plot(predictions, label="Predicted Values", color="red", linestyle="--")
    # plt.xlabel("Time")
    # plt.ylabel("Values")
    # plt.title("Actual vs Predicted Values Over Time")
    # plt.legend()
    # plt.savefig(save_to / "actual_vs_predicted_timeseries.png")
    # plt.close()

    # # Joint plot of predictions vs actual values using seaborn
    # plt.figure(figsize=(10, 6))
    # sns.jointplot(x=y_test, y=predictions, kind="reg", height=10)
    # plt.xlabel("Actual Values")
    # plt.ylabel("Predicted Values")
    # plt.title("Predictions vs Actual Values")
    # plt.savefig(save_to / "predictions_vs_actual.png")
    # plt.close()

    # # Feature importance plot
    # feature_importances = model.feature_importances_
    # sorted_idx = feature_importances.argsort()
    # plt.figure(figsize=(10, 6))
    # plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
    # plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    # plt.xlabel("Feature Importance")
    # plt.title("Feature Importance Plot")
    # plt.savefig(save_to / "feature_importance.png")
    # plt.close()
