import logging, pdb
import os
import click
import pandas as pd
from dotenv import load_dotenv


from pathlib import Path
from ray import tune
from ray.tune import ResultGrid, ExperimentAnalysis
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler

from lstm.trainable import LSTMTrainable, test_result
from data.data import EarthquakeData
from settings import read_coordinates

logger = logging.getLogger(__name__)


def load_data(file: str, env: str) -> EarthquakeData:
    load_dotenv(env, override=True)
    file_path = file or os.getenv("FILE_PATH")
    assert file_path, "FILE_PATH was not provided"
    raw_data = pd.read_csv(file_path)
    latitude, longitude = read_coordinates()
    numeric_columns = os.getenv("NUMERIC_COLUMNS", "").split(",")
    zero_columns = os.getenv("ZERO_COLUMNS", "").split(",")
    numeric_columns = [col for col in numeric_columns if col]
    zero_columns = [col for col in zero_columns if col]
    time_column = os.getenv("TIME_COLUMN", "true").lower() == "true"
    delta_time = os.getenv("DELTA_TIME", "true").lower() == "true"
    drop_time = os.getenv("DROP_TIME", "true").lower() == "true"

    return EarthquakeData(
        raw_data,
        numeric_columns=numeric_columns,
        zero_columns=zero_columns,  # can be zero when scaling
        time_column=time_column,
        delta_time=delta_time,
        drop_time_column=drop_time,
        min_latitude=min(latitude),
        min_longitude=min(longitude),
    )


@click.command(name="tune")
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-t", "--target", type=str, help="target in the given dataset to forecast")
@click.option("-e", "--env", type=str, help="path to .env file with variables to be loaded", default="./.env")
def tune_command(file, target, env):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    qdata = load_data(file, env)
    logger.info("Processing data...")
    data, scalers = qdata.process()

    target = target or os.getenv("TARGET")
    if not target or not target in data.columns:
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
    test_result(best_result, scalers.get(target))


@click.command(name="test")
@click.option("-ex", "--experiment-path", type=str, help="experiment path", default=None)
@click.option("-t", "--target", type=str, help="target in the given dataset to forecast")
@click.option("-e", "--env", type=str, help="env path", default=None)
def test_command(experiment_path, target, env):
    load_dotenv(env, override=True)
    if not experiment_path:
        ray_results = Path.home() / "ray_results"
        folders = {
            idx: folder
            for idx, folder in enumerate(ray_results.glob("*"))
            if folder.is_dir()
            if folder.stem[0].isalpha()
        }
        prompt = "\n".join(f"{idx}) {folder.stem}" for idx, folder in folders.items())
        choice = click.prompt(prompt, type=int, default=None)
        assert choice is not None, choice
        result_path = folders.get(choice)
    else:
        result_path = Path(result_path).resolve()

    logger.info("Loading %s", result_path)
    analysis = ExperimentAnalysis(result_path)
    result_grid = ResultGrid(analysis)

    result = result_grid.get_best_result("loss", "min")

    target = target or os.getenv("TARGET")
    qdata = load_data(None, env)
    _, scalers = qdata.process()
    assert target in scalers, f"[{target}] not in scalers {scalers.keys()}"

    test_result(result, scalers.get(target))


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
lstm_group.add_command(test_command)
