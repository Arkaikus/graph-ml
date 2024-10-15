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
from lstm.regression import RegressionTrainable, test_result

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
@click.option("--node-size", type=int, help="size of node in kms", default=100)
@click.option("--metric", type=str, help="metric", default="loss")
@click.option("--mode", type=str, help="mode", default="min")
@click.option("--networkx", type=bool, help="mode", default=False)
@click.option(
    "--nx-features",
    multiple=True,
    help="features to be used in the model",
    default=["degree_centrality", "clustering", "betweenness_centrality", "closeness_centrality", "pagerank"],
    callback=lambda ctx, param, value: list(value),
)
@click.option("--nx-lookback", type=int, help="mode", default=5)
@click.option("-s", "--samples", type=int, help="samples", default=-1)
def tune_regression(
    features,
    target,
    min_lat,
    max_lat,
    min_long,
    max_long,
    min_mag,
    max_mag,
    node_size,
    metric,
    mode,
    networkx,
    nx_features,
    nx_lookback,
    samples,
):
    """Reads a processed .csv catalog and trains an LSTM neural network"""
    logger.info("Downloading data...")
    latitude = (min_lat, max_lat)
    longitude = (min_long, max_long)
    raw_data = USGS(latitude, longitude).download()
    kwargs = {}
    if networkx:
        grid = Grid(latitude, longitude, node_size)
        kwargs["grid"] = grid
        kwargs["network_features"] = nx_features
        kwargs["network_lookback"] = nx_lookback

    qdata = EarthquakeData(raw_data, features, [target], min_magnitude=min_mag, max_magnitude=max_mag, **kwargs)

    logger.info("Processing data...")
    utils.plot_analysis(qdata.data, features, target, Path.home() / "plots" / qdata.hash)

    logger.info("Tuning with metric %s mode: %s", metric, mode)
    scheduler = ASHAScheduler(metric=metric, mode=mode, grace_period=1, reduction_factor=2)
    trainable = tune.with_parameters(RegressionTrainable, qdata=qdata)

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
            "loss_type": tune.choice(["mse", "mape", "mae"]),
        },
    )
    results = tuner.fit()
    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result(metric, mode)
    test_result(best_result, qdata, metric, mode)
    # pickle qdata
    with open(Path(results.experiment_path) / "qdata.pkl", "wb") as f:
        pickle.dump(qdata, f)


@click.command(name="test")
@click.option("-ex", "--experiment-path", type=str, help="experiment path", default=None)
@click.option("--metric", type=str, help="metric", default="loss")
@click.option("--mode", type=str, help="mode", default="min")
def test_command(experiment_path, metric, mode):
    result_path = Path(experiment_path).resolve() if experiment_path else utils.prompt_experiment()

    logger.info("Loading qdata...")
    with open(Path(result_path) / "qdata.pkl", "rb") as f:
        qdata = pickle.load(f)

    logger.info("Loading %s", result_path)
    analysis = ExperimentAnalysis(result_path)
    result_grid = ResultGrid(analysis)
    result = result_grid.get_best_result(metric, mode)
    test_result(result, qdata, metric, mode)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_regression)
lstm_group.add_command(test_command)
