import logging, pdb

import click
import pandas as pd
from ray import tune
from ray.data import from_pandas
from ray.train import RunConfig, ScalingConfig
from ray.tune.tuner import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lstm.trainable import LSTMTrainable, test_model
from processing.data import EarthquakeData
from settings import read_coordinates

logger = logging.getLogger(__name__)


@click.command(name="tune")
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-t", "--target", type=str, help="target in the given dataset to forecast", default="mag")
def tune_command(file, target):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    raw_data = pd.read_csv(file)
    latitude, longitude = read_coordinates()
    qdata = EarthquakeData(
        raw_data,
        numeric_columns=["latitude", "longitude", "depth", "mag"],
        zero_columns=["depth", "mag"],  # can be zero when scaling
        time_column=True,
        delta_time=click.confirm("Calculate days between?", default=False),
        drop_time_column=True,
        min_latitude=min(latitude),
        min_longitude=min(longitude),
    )
    logger.info("Processing data...")
    data, _ = qdata.process()

    if not target:
        target = click.prompt("Target of [{}]".format(",".join(data.columns)))
        assert target
    assert target in data.columns, f"[{target}] not in data"

    # Define the search space for hyperparameters
    test_size = 0.3
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    logger.info("Train records(%s): %s", 1 - test_size, train_data.shape[0])
    logger.info("Test records(%s): %s", test_size, test_data.shape[0])
    logger.info("Total: %s", data.shape[0])
    logger.info("Features: %s", tuple(data.columns))
    logger.info("Target: %s", target)

    pdb.set_trace()
    # Use ASHAScheduler for efficient hyperparameter optimization
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=30, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(LSTMTrainable, resources={"cpu": 2, "gpu": 1}),  # TRAINABLE
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=10),
        param_space={
            "data": train_data,
            "target": target,
            "window_size": tune.choice([50, 100, 150]),
            "hidden_size": tune.choice([32, 64, 128]),
            "num_layers": tune.choice([1, 2, 3]),
            "lr": tune.choice([1e-3, 1e-4]),
            "batch_size": tune.choice([1, 5, 10]),
            "num_epochs": tune.choice([25, 50, 100]),
        },
    )
    # pdb.set_trace()
    results = tuner.fit()

    logger.info("Results path at %s", results.experiment_path)

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final loss: {}".format(best_result.metrics["loss"]))

    test_model(best_result, test_data, target)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
