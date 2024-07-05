import logging, pdb
import os
import click
import pandas as pd
from pathlib import Path
from ray import tune
from ray.train import Result
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler

from lstm.trainable import LSTMTrainable, test_model
from data.data import EarthquakeData
from settings import read_coordinates

logger = logging.getLogger(__name__)

experiment_path = Path(os.getcwd()) / "ray_results"
experiment_path.mkdir(exist_ok=True)


@click.command(name="tune")
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-t", "--target", type=str, help="target in the given dataset to forecast", default="mag")
@click.option("-dt", "--deltatime", type=bool, help="include delta time events", default=True)
def tune_command(file, target, deltatime):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    raw_data = pd.read_csv(file)
    latitude, longitude = read_coordinates()
    qdata = EarthquakeData(
        raw_data,
        numeric_columns=["latitude", "longitude", "depth", "mag"],
        zero_columns=["depth", "mag"],  # can be zero when scaling
        time_column=True,
        delta_time=deltatime,
        drop_time_column=True,
        min_latitude=min(latitude),
        min_longitude=min(longitude),
    )
    logger.info("Processing data...")
    data, scalers = qdata.process()

    if not target:
        target = click.prompt("Target of [{}]".format(",".join(data.columns)))
        assert target
    assert target in data.columns, f"[{target}] not in data"

    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=30, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(LSTMTrainable, resources={"cpu": 2, "gpu": 1}),  # TRAINABLE
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=10),
        param_space={
            "data": data,
            "target": target,
            "test_size": tune.choice([0.3, 0.2, 0.1]),
            "sequence_size": tune.choice([50, 100, 150]),
            "hidden_size": tune.choice([32, 64, 128]),
            "num_layers": tune.choice([1, 2, 3]),
            "lr": tune.choice([1e-3, 1e-4]),
            "batch_size": tune.choice([1, 5, 10]),
            "num_epochs": tune.choice([25, 50, 100]),
        },
    )
    results = tuner.fit()

    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result("loss", "min")
    test_model(best_result, scalers.get(target))


@click.command(name="test")
@click.option("-r", "--result-path", type=str, help="result path", default=None)
def test_command(result_path):
    if not result_path:
        ray_results = Path.home() / "ray_results"
        choices = {idx: folder for idx, folder in enumerate(os.listdir(ray_results.as_posix()))}
        choice = click.prompt("\n".join(f"{idx}) {folder}" for idx, folder in choices.items()), type=int, default=None)
        assert choice is not None, choice
        choice_path = ray_results / choices.get(choice)
        result_path = next(path for item in os.listdir(choice_path) if (path := choice_path / item).is_dir())
    else:
        result_path = Path(result_path).resolve()

    logger.info("Loading %s", result_path)
    result = Result.from_path(result_path.as_posix())
    pdb.set_trace()


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
lstm_group.add_command(test_command)
