import logging

import click
import pandas as pd
from ray import tune as rtune
from ray.tune.schedulers import ASHAScheduler
from sklearn.preprocessing import MinMaxScaler

from lstm.train import build_trainer
from processing.data import Data
from settings import read_coordinates

logger = logging.getLogger(__name__)


def prepare_data(file):
    """Takes a file path to process the data"""
    raw_data = pd.read_csv(file)
    data = Data(
        raw_data,
        numeric_columns=["latitude", "longitude", "depth", "mag"],
        time_column=True,
        delta_time=click.confirm("Calculate days between?", default=False),
        drop_time_column=True,
    )
    logger.info("Processing data")
    latitude, longitude = read_coordinates()
    ready_data, _ = data.process(scaler=MinMaxScaler(), min_latitude=min(latitude), min_longitude=min(longitude))

    return ready_data, data


@click.command()
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-t", "--target", type=str, help="target in the given dataset to forecast", default="mag")
def tune(file, target):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    data, _ = prepare_data(file, target)
    if not target:
        target = click.prompt("Target of [{}]".format(",".join(data.columns)))
        assert target
    assert target in data.columns, f"[{target}] not in data"

    logger.info("Preparing trainer")
    trainer = build_trainer(data, target)

    import pdb

    pdb.set_trace()

    t = trainer({"window_size": 10, "hidden_size": 1, "num_layers": 1, "lr": 0.0001, "batch_size": 1, "num_epochs": 1})

    # Define the search space for hyperparameters
    config = {
        "window_size": rtune.choice([50, 100, 200]),
        "hidden_size": rtune.choice([32, 64, 128]),
        "num_layers": rtune.choice([1, 2, 3]),
        "lr": rtune.loguniform(1e-4, 1e-1),
        "batch_size": rtune.choice([2, 4, 8]),
        "num_epochs": rtune.choice([10, 20, 30]),
    }

    # Use ASHAScheduler for efficient hyperparameter optimization
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=30, grace_period=1, reduction_factor=2)

    # Run hyperparameter search
    logger.info("Running hyperparameter search")
    analysis = rtune.run(
        trainer,
        resources_per_trial={"cpu": 6, "gpu": 1},  # Adjust based on your resources
        config=config,
        num_samples=10,
        scheduler=scheduler,
    )

    print("Best hyperparameters found were: ", analysis.best_config)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune)
