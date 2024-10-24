import logging
import os
from pathlib import Path
import pickle

import click
import ray
from ray import tune
from ray.tune import ExperimentAnalysis, ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler

from data.data import EarthquakeData
from data.usgs import USGS
from data.grid import Grid
from lstm import utils

logger = logging.getLogger(__name__)


@click.command(name="tune")
@click.option(
    "--features",
    multiple=True,
    help="features to be used in the model",
    default=["latitude", "longitude", "depth", "mag"],
    callback=lambda ctx, param, value: list(value),
)
@click.option("--target", type=str, help="target to be predicted", default="mag")
@click.option("--min-lat", type=float, help="min latitude", default=-0.132)
@click.option("--max-lat", type=float, help="max latitude", default=9.796)
@click.option("--min-long", type=float, help="min longitude", default=-80.343)
@click.option("--max-long", type=float, help="max longitude", default=-72.466)
@click.option("--min-mag", type=float, help="min magnitude", default=0)
@click.option("--max-mag", type=float, help="max magnitude", default=10)
@click.option("--metric", type=str, help="metric", default="loss")
@click.option("--mode", type=str, help="mode", default="min")
@click.option("--networkx", type=bool, help="mode", default=False)
@click.option("--node-size", type=int, help="size of node in kms", default=100)
@click.option("--classify", type=bool, help="whether to regress or classify data", default=False)
@click.option("-s", "--samples", type=int, help="samples", default=-1)
def tune_command(
    features: list,
    target: str,
    min_lat: float,
    max_lat: float,
    min_long: float,
    max_long: float,
    min_mag: float,
    max_mag: float,
    metric: str,
    mode: str,
    networkx: bool,
    node_size: int,
    classify: bool,
    samples,
):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    logger.info("Downloading data...")
    latitude = (min_lat, max_lat)
    longitude = (min_long, max_long)
    raw_data = USGS(latitude, longitude).download()
    kwargs = {}
    param_space = {}
    if networkx:
        grid = Grid(latitude, longitude, node_size)
        kwargs["grid"] = grid
        nx_features = ["degree_centrality", "clustering", "betweenness_centrality", "closeness_centrality", "pagerank"]
        param_space["network_features"] = tune.choice(nx_features)
        param_space["network_lookback"] = tune.randint(1, 10)
        param_space["node_size"] = node_size

    if classify:
        from .classification import ClassificationTrainable as Trainable

        param_space["quantiles"] = tune.randint(2, 6)
    else:
        from .regression import RegressionTrainable as Trainable

    qdata = EarthquakeData(raw_data, features, [target], min_magnitude=min_mag, max_magnitude=max_mag, **kwargs)

    logger.info("Processing data...")
    utils.plot_analysis(qdata.data, features, target, Path.home() / "plots" / qdata.hash)

    logger.info("Tuning with metric %s mode: %s", metric, mode)
    scheduler = ASHAScheduler(metric=metric, mode=mode, grace_period=1, reduction_factor=2)
    trainable = tune.with_parameters(Trainable, qdata=qdata)

    ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 8, "gpu": 1}),
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=samples, max_concurrent_trials=None),
        param_space={
            "lookback": tune.randint(10, 150),
            "test_size": tune.uniform(0.1, 0.3),
            "batch_size": tune.randint(2, 20),
            "hidden_size": tune.randint(10, 150),
            "lstm_layers": tune.randint(2, 10),
            "lr": tune.loguniform(1e-4, 1e-2),
            "max_epochs": tune.randint(10, 70),
            "loss_type": tune.choice(["mse", "mape", "mae"] if not classify else ["cross_entropy"]),
            **param_space,
        },
    )
    results = tuner.fit()
    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result(metric, mode)
    trainable_cls = tune.with_parameters(Trainable, qdata=qdata)
    trainable = trainable_cls(config=best_result.config)
    trainable.test_result(best_result, metric, mode)
    # pickle qdata
    with open(Path(results.experiment_path) / "qdata.pkl", "wb") as f:
        pickle.dump(qdata, f)


@click.command(name="test")
@click.option("-ex", "--experiment-path", type=str, help="experiment path", default=None)
@click.option("--metric", type=str, help="metric", default="loss")
@click.option("--mode", type=str, help="mode", default="min")
@click.option("--classify", type=bool, help="whether to regress or classify data", default=False)
def test_command(experiment_path, metric, mode, classify):
    result_path = Path(experiment_path).resolve() if experiment_path else utils.prompt_experiment()

    logger.info("Loading qdata...")
    with open(Path(result_path) / "qdata.pkl", "rb") as f:
        qdata = pickle.load(f)

    logger.info("Loading %s", result_path)
    analysis = ExperimentAnalysis(result_path)
    result_grid = ResultGrid(analysis)
    result = result_grid.get_best_result(metric, mode)
    if classify:
        from .classification import ClassificationTrainable as Trainable
    else:
        from .regression import RegressionTrainable as Trainable

    trainable_cls = tune.with_parameters(Trainable, qdata=qdata)
    trainable = trainable_cls(config=result.config)
    trainable.test_result(result, metric, mode)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
lstm_group.add_command(test_command)
