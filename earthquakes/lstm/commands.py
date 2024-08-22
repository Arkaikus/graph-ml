import logging
import os
import click
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from ray import tune
from ray.tune import ExperimentAnalysis, ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler

from lstm.trainable import LSTMTrainable, test_result
from data.data import EarthquakeData
from settings import read_coordinates

logger = logging.getLogger(__name__)

experiment_path = Path(os.getcwd()) / "ray_results"
experiment_path.mkdir(exist_ok=True)


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


def load_data(file: str, env: str) -> EarthquakeData:
    load_dotenv(env, override=True)
    file_path = file or os.getenv("FILE_PATH")
    assert file_path, "FILE_PATH was not provided"
    raw_data = pd.read_csv(file_path)
    latitude, longitude = read_coordinates()
    return EarthquakeData(
        raw_data,
        ["latitude", "longitude", "depth", "mag"],
        "mag",
        zero_columns=["depth", "mag"],  # can be zero when scaling
        time_column=True,
        delta_time=True,
        drop_time_column=True,
        min_latitude=min(latitude),
        min_longitude=min(longitude),
    )


@click.command(name="tune")
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-e", "--env", type=str, help="env")
def tune_command(file, env):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    qdata = load_data(file, env)
    logger.info("Processing data...")
    scheduler = ASHAScheduler(metric="loss", mode="min", grace_period=1, reduction_factor=2)
    trainable = tune.with_parameters(LSTMTrainable, qdata=qdata)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 2, "gpu": 1}),  # TRAINABLE
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=10),
        param_space={
            "test_size": tune.choice([0.3, 0.2, 0.1]),
            "sequence_size": tune.choice([50, 100, 150]),
            "hidden_size": tune.choice([32, 64, 128]),
            "num_layers": tune.choice([1, 2, 3]),
            "lr": tune.choice([1e-3, 1e-4]),
            "batch_size": tune.choice([2, 5, 10]),
            "max_epochs": tune.choice([25, 50, 100]),
        },
    )
    results = tuner.fit()

    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result("loss", "min")
    test_result(best_result, qdata)


@click.command(name="test")
@click.option("-ex", "--experiment-path", type=str, help="experiment path", default=None)
@click.option("-e", "--env", type=str, help="env path", default=None)
def test_command(experiment_path, env):
    result_path = Path(experiment_path).resolve() if experiment_path else prompt_experiment()
    logger.info("Loading %s", result_path)
    analysis = ExperimentAnalysis(result_path)
    result_grid = ResultGrid(analysis)
    result = result_grid.get_best_result("loss", "min")
    qdata = load_data(None, env)
    logger.info("Processing data...")
    test_result(result, qdata)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
lstm_group.add_command(test_command)
