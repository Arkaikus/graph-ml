import logging
import os
import pickle
from pathlib import Path

import click
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler

from data.data import EarthquakeData
from data.grid import Grid
from data.usgs import USGS
from lstm import utils

from .classification import ClassificationTrainable as Trainable

logger = logging.getLogger(__name__)


def save_experiment_df(results: ResultGrid, metric, mode, qdata: EarthquakeData, networkx=False):
    results_df = results.get_dataframe()
    results_df = results_df.sort_values(by=metric, ascending=(mode == "min"))
    print(results_df.columns)

    results_df = results_df.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    extra_columns = ["accuracy", "config/quantiles"]

    if networkx:
        extra_columns += [
            "config/network_features",
            "config/network_lookback",
            "config/node_size",
        ]

    results_df = results_df[
        [
            "loss",
            "mean_loss",
            "test_loss",
            "config/lookback",
            "config/test_size",
            "config/batch_size",
            "config/hidden_size",
            "config/lstm_layers",
            "config/lr",
            "config/max_epochs",
        ]
        + extra_columns
    ].rename(columns={c: c.replace("config/", "").replace("_", " ") for c in results_df.columns})
    latex_table = results_df.to_latex(index=False)
    save_to = Path.cwd() / "plots" / qdata.hash
    save_to.mkdir(parents=True, exist_ok=True)
    experiment_name = Path(results.experiment_path).stem
    with open(save_to / f"{experiment_name}_results_table.tex", "w") as f:
        f.write(latex_table)

    results_df.to_csv(save_to / f"{experiment_name}_results_table.csv", index=False)
    logger.info("Saved .tex table to %s", save_to / f"{experiment_name}_results_table.tex")
    logger.info("Saved .csv table to %s", save_to / f"{experiment_name}_results_table.csv")


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
@click.option("--metric", type=str, help="metric", default="test_loss")
@click.option("--mode", type=str, help="mode", default="min")
@click.option("--networkx", type=bool, help="mode", default=False)
@click.option("--node-size", type=int, help="size of node in kms", default=100)
@click.option("--quantiles", type=int, help="number of categories", default=2)
@click.option("-s", "--samples", type=int, help="samples", default=-1)
@click.option("-resume", "--resume", type=bool, help="resume experiment", default=False)
@click.option("-ex", "--experiment", type=str, help="resume experiment path", default=None)
@click.option("-cpus", "--cpus", type=int, help="cpus to use", default=8)
@click.option("-gpus", "--gpus", type=int, help="gpus to use", default=1)
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
    quantiles: int,
    samples,
    resume,
    experiment,
    cpus,
    gpus,
):
    """
    Reads a processed .csv catalog and trains an LSTM neural network

    quakes lstm tune --quantiles 2 --samples 1 --metric accuracy --mode max
    quakes lstm tune --quantiles 2 --samples 1 --metric accuracy --mode max -ex ~/ray_results/ClassificationTrainable_2024-11-28_13-08-36
    quakes lstm tune --quantiles 2 --samples 1 --metric accuracy --mode max --networkx
    quakes lstm tune --quantiles 2 --samples 10 --metric accuracy --mode max
    """
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
        param_space["network_features"] = tune.grid_search(nx_features)
        param_space["network_lookback"] = tune.randint(1, 10)
        param_space["node_size"] = node_size

    qdata = EarthquakeData(raw_data, features, [target], min_magnitude=min_mag, max_magnitude=max_mag, **kwargs)

    logger.info("Processing data...")
    utils.plot_analysis(qdata.data, features, target, Path.cwd() / "plots" / qdata.hash)

    logger.info("Tuning with metric %s mode: %s", metric, mode)
    scheduler = ASHAScheduler(metric=metric, mode=mode, grace_period=1, reduction_factor=2)
    trainable = tune.with_parameters(Trainable, qdata=qdata)
    trainable = tune.with_resources(trainable, resources={"cpu": cpus, "gpu": gpus})
    param_space = {
        "lookback": tune.randint(10, 150),
        "test_size": tune.uniform(0.1, 0.3),
        "batch_size": tune.randint(2, 20),
        "hidden_size": tune.randint(10, 150),
        "lstm_layers": tune.randint(2, 10),
        "lr": tune.loguniform(1e-4, 1e-2),
        "max_epochs": tune.randint(10, 70),
        "quantiles": quantiles,
        **param_space,
    }

    ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)
    experiment_path = Path(experiment) if experiment else None
    if resume and not experiment_path:
        experiment_path = utils.prompt_experiment()

    if experiment_path:
        logger.info("Resuming experiment...")
        tuner = tune.Tuner.restore(
            path=experiment_path.absolute().as_posix(),
            trainable=trainable,
            resume_unfinished=True,
            resume_errored=True,
            param_space=param_space,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=samples,
                max_concurrent_trials=None,
            ),
            run_config=RunConfig(),
            param_space=param_space,
        )

    results = tuner.fit()
    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result(metric, mode)
    trainable: Trainable = tune.with_parameters(Trainable, qdata=qdata)(config=best_result.config)
    try:
        trainable.test_result(best_result, metric, mode)
    except:
        logger.error("Error testing best result")

    logger.info("Saving qdata at %s", Path(results.experiment_path) / "qdata.pkl")
    with open(Path(results.experiment_path) / "qdata.pkl", "wb") as f:
        pickle.dump(qdata, f)

    save_experiment_df(results, metric, mode, qdata, networkx)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
