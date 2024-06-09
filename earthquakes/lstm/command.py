import logging

import click
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.preprocessing import MinMaxScaler

from lstm.train import build_trainer
from processing.data import Data

logger = logging.getLogger(__name__)


@click.command()
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-t", "--target", type=str, help="target in the given dataset to forecast")
def train_lstm(file, target):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    raw_data = pd.read_csv(file)
    assert target in raw_data.columns, f"[{target}] not in data"

    data = Data(
        raw_data,
        numeric_columns=["latitude", "longitude", "depth", "mag"],
        time_column=True,
        delta_time=True,
        drop_time_column=True,
    )

    scaler = MinMaxScaler()

    ready_data, _ = data.process(scaler=MinMaxScaler())
    trainer = build_trainer(ready_data, target)
    # Define the search space for hyperparameters
    config = {
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8]),
        "num_epochs": tune.choice([10, 20, 30]),
    }

    # Use ASHAScheduler for efficient hyperparameter optimization
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=30, grace_period=1, reduction_factor=2)

    # Run hyperparameter search
    analysis = tune.run(
        trainer,
        resources_per_trial={"cpu": 0, "gpu": 1},  # Adjust based on your resources
        config=config,
        num_samples=10,
        scheduler=scheduler,
    )

    print("Best hyperparameters found were: ", analysis.best_config)
