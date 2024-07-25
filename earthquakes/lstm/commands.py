import logging
import os
import pdb
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from ray import tune, train
from ray.tune import ExperimentAnalysis, ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler

from data.data import EarthquakeData

# from lstm.trainable_fn import train_fn, test_result
from lstm.trainable import LSTMTrainable as train_fn, test_result
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
    numeric_columns = split_n_parse(os.getenv("NUMERIC_COLUMNS", ""), str)
    zero_columns = split_n_parse(os.getenv("ZERO_COLUMNS", ""), str)
    target = os.getenv("TARGET")
    scaler_mode = os.getenv("SCALER_MODE")
    time_column = os.getenv("TIME_COLUMN", "true").lower() == "true"
    delta_time = os.getenv("DELTA_TIME", "true").lower() == "true"
    drop_time = os.getenv("DROP_TIME", "true").lower() == "true"

    return EarthquakeData(
        raw_data,
        numeric_columns=numeric_columns,
        target=target,
        zero_columns=zero_columns,  # can be zero when scaling
        time_column=time_column,
        delta_time=delta_time,
        drop_time_column=drop_time,
        min_latitude=min(latitude),
        min_longitude=min(longitude),
        scaler_mode=scaler_mode,
    )


def load_parameters(env):
    load_dotenv(env, override=True)
    test_size = os.getenv("TEST_SIZE")
    sequence_size = os.getenv("SEQUENCE_SIZE")
    hidden_size = os.getenv("HIDDEN_SIZE")
    dropout = os.getenv("DROPOUT")
    num_layers = os.getenv("NUM_LAYERS")
    lr = os.getenv("LR")
    batch_size = os.getenv("BATCH_SIZE")
    epochs = os.getenv("EPOCHS")

    return {
        "test_size": tune.choice(split_n_parse(test_size, float)),
        "sequence_size": tune.choice(split_n_parse(sequence_size, int)),
        "hidden_size": tune.choice(split_n_parse(hidden_size, int)),
        "num_layers": tune.choice(split_n_parse(num_layers, int)),
        "lr": tune.choice(split_n_parse(lr, float)),
        "batch_size": tune.choice(split_n_parse(batch_size, int)),
        "epochs": tune.choice(split_n_parse(epochs, int)),
        "dropout": tune.choice(split_n_parse(dropout, float)),
    }


@click.command(name="tune")
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
@click.option("-e", "--env", type=str, help="path to .env file with variables to be loaded")
@click.option("-r", "--resume", type=bool, help="whether to continue training on an existing experiment", default=False)
@click.option("-s", "--samples", type=int, help="number of parameter space samples to take", default=10)
def tune_command(file, env, resume, samples):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    logger.info("Processing data...")
    qdata = load_data(file, env)
    param_space = load_parameters(env)
    trainable = tune.with_parameters(train_fn, qdata=qdata)
    trainable = tune.with_resources(trainable, resources={"cpu": 8, "gpu": 1})
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=30, grace_period=1, reduction_factor=2)
    tune_config = tune.TuneConfig(scheduler=scheduler, num_samples=samples)
    kwargs = {}
    if resume:
        kwargs["run_config"] = train.RunConfig(storage_path=prompt_experiment())

    # instance tuner
    tuner = tune.Tuner(trainable, tune_config=tune_config, param_space=param_space, **kwargs)
    results = tuner.fit()

    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result("loss", "min")
    test_result(best_result, qdata)


@click.command(name="test")
@click.option("-ex", "--experiment-path", type=str, help="experiment path", default=None)
@click.option("-e", "--env", type=str, help="env path", default=None)
def test_command(experiment_path, target, env):
    result_path = Path(experiment_path).resolve() if experiment_path else prompt_experiment()
    logger.info("Loading %s", result_path)
    analysis = ExperimentAnalysis(result_path)
    result_grid = ResultGrid(analysis)
    result = result_grid.get_best_result("loss", "min")
    qdata = load_data(None, env)
    test_result(result, qdata)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
lstm_group.add_command(test_command)
