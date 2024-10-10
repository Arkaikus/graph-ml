import logging
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import ray
from ray import tune
from ray.tune import ExperimentAnalysis, ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler

from data.data import EarthquakeData

from lstm.trainable import LSTMTrainable, test_result
from settings import read_coordinates

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


def load_data(file: str, env: str) -> EarthquakeData:
    load_dotenv(env, override=True)
    file_path = file or os.getenv("FILE_PATH")
    assert file_path, "FILE_PATH was not provided"
    raw_data = pd.read_csv(file_path)
    latitude, longitude = read_coordinates()
    return EarthquakeData(
        raw_data,
        ["latitude", "longitude", "depth", "mag"],
        ["mag"],
        zero_columns=["depth", "mag"],  # can be zero when scaling
        time_column=True,
        delta_time=True,
        drop_time_column=True,
        min_latitude=min(latitude),
        min_longitude=min(longitude),
        min_magnitude=0,
        max_magnitude=9,
    )


@click.command(name="tune")
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-e", "--env", type=str, help="env")
@click.option("-s", "--samples", type=int, help="samples", default=-1)
# @click.option("-m", "--mode", type=str, help="pytorch/tensorflow", default="pytorch")
def tune_command(file, env, samples):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    qdata = load_data(file, env)
    logger.info("Processing data...")
    metric = os.getenv("METRIC", "loss")
    opt_mode = "min"
    logger.info("Tuning with metric %s mode: %s", metric, opt_mode)
    scheduler = ASHAScheduler(metric=metric, mode=opt_mode, grace_period=1, reduction_factor=2)
    trainable = tune.with_parameters(LSTMTrainable, qdata=qdata)

    ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 8, "gpu": 1}),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=samples,
            max_concurrent_trials=None,
        ),
        param_space={
            "lookback": tune.randint(10, 150),
            "test_size": tune.uniform(0.1, 0.3),
            "batch_size": tune.randint(2, 20),
            "hidden_size": tune.randint(32, 128),
            "lstm_layers": tune.randint(2, 10),
            "lr": tune.loguniform(1e-4, 1e-2),
            "max_epochs": tune.randint(10, 50),
            "loss_type": tune.choice(["mse", "mape", "mae"]),
        },
    )
    results = tuner.fit()

    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result(metric, opt_mode)
    test_result(best_result, qdata)


@click.command(name="test")
@click.option("-ex", "--experiment-path", type=str, help="experiment path", default=None)
@click.option("-e", "--env", type=str, help="env path", default=None)
def test_command(experiment_path, env):
    result_path = Path(experiment_path).resolve() if experiment_path else prompt_experiment()
    logger.info("Loading %s", result_path)
    analysis = ExperimentAnalysis(result_path)
    result_grid = ResultGrid(analysis)
    metric = os.getenv("METRIC", "loss")
    result = result_grid.get_best_result(metric, "min")
    qdata = load_data(None, env)
    test_result(result, qdata)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
lstm_group.add_command(test_command)
