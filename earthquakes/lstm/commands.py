import logging
import os
import pickle
import sys
from pathlib import Path

import click
import ray
from data.data import EarthquakeData
from data.grid import Grid
from data.usgs import USGS
from lstm import utils
from lstm.classification import ClassificationTrainable
from lstm.dataset import LOOKBACK_CHOICES, NETWORK_LOOKBACK_CHOICES, precompute_sequences
from lstm.regression import RegressionTrainable
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import RunConfig
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler
from reporting.reporter import Reporter

logger = logging.getLogger(__name__)

TASK_TRAINABLES = {
    "classification": ClassificationTrainable,
    "regression": RegressionTrainable,
}


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
@click.option("--networkx", is_flag=True, help="mode")
@click.option("--node-size", type=int, help="size of node in kms", default=100)
@click.option("--quantiles", type=int, help="number of categories (classification only)", default=2)
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("-s", "--samples", type=int, help="samples", default=-1)
@click.option("-resume", "--resume", type=bool, help="resume experiment", default=False)
@click.option("-ex", "--experiment", type=str, help="resume experiment path", default=None)
@click.option("-cpus", "--cpus", type=int, help="cpus to use", default=8)
@click.option("-gpus", "--gpus", type=int, help="gpus to use", default=1)
@click.option(
    "--mlflow-uri",
    type=str,
    help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or ./mlruns)",
    default="http://localhost:5000",
)
@click.option(
    "--force-precompute",
    is_flag=True,
    help="Recompute parquet sequences even if cache exists",
)
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
    task: str,
    samples: int,
    resume: bool,
    experiment: str | None,
    cpus: int,
    gpus: int,
    mlflow_uri: str,
    force_precompute: bool,
):
    """
    Reads a processed .csv catalog and trains an LSTM neural network.

    Examples:
      quakes lstm tune --task classification --quantiles 2 --samples 1 --metric accuracy --mode max
      quakes lstm tune --task regression --samples 1 --metric mean_test_loss --mode min
      quakes lstm tune ... -ex ~/ray_results/ClassificationTrainable_2024-11-28_13-08-36
      quakes lstm tune ... --networkx
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
        nx_features = [
            "degree_centrality",
            "clustering",
            "betweenness_centrality",
            "closeness_centrality",
            "pagerank",
        ]
        param_space["network_features"] = nx_features
        param_space["network_lookback"] = tune.choice(NETWORK_LOOKBACK_CHOICES)
        param_space["node_size"] = node_size

    qdata = EarthquakeData(raw_data, features, [target], min_magnitude=min_mag, max_magnitude=max_mag, **kwargs)

    reporter = Reporter(output_dir=Path.cwd() / "runs", tracking_uri=mlflow_uri)
    output_dir = reporter.subdir(qdata.hash)

    logger.info("Processing data...")
    utils.plot_analysis(qdata.data, features, target, output_dir)

    trainable_cls = TASK_TRAINABLES[task]
    run_name = trainable_cls.__name__
    logger.info("Tuning with metric %s mode: %s (task=%s)", metric, mode, task)
    scheduler = ASHAScheduler(metric=metric, mode=mode, grace_period=1, reduction_factor=2)

    sequences_cache_dir = precompute_sequences(
        qdata,
        task=task,
        networkx=networkx,
        quantiles=quantiles,
        out_dir=Path.cwd() / "cache" / "parquet",
        force=force_precompute,
    )
    trainable = tune.with_parameters(
        trainable_cls,
        qdata=qdata,
        output_dir=output_dir,
        sequences_cache_dir=sequences_cache_dir,
    )
    trainable = tune.with_resources(trainable, resources={"cpu": cpus, "gpu": gpus})
    param_space = {
        "lookback": tune.choice(LOOKBACK_CHOICES),
        "test_size": tune.uniform(0.1, 0.3),
        "batch_size": tune.randint(2, 20),
        "hidden_size": tune.randint(10, 150),
        "lstm_layers": tune.randint(2, 10),
        "lr": tune.loguniform(1e-4, 1e-2),
        "max_epochs": tune.randint(10, 70),
        "dropout": tune.uniform(0.0, 0.3),
        "use_attention": tune.choice([False, True]),
        **param_space,
    }
    if task == "classification":
        param_space["quantiles"] = quantiles
        param_space["loss_type"] = tune.choice(["cross_entropy", "focal", "label_smoothing"])

    ray_temp = Path.home() / ".cache" / "ray" / "tmp"
    ray_temp.mkdir(parents=True, exist_ok=True)
    # Use driver's Python/venv so workers reuse the same .venv instead of creating new ones
    runtime_env = {"py_executable": sys.executable, "env_vars": {"RAY_DISABLE_UV_RUNTIME_ENV": "1"}}
    ray.init(
        dashboard_host="0.0.0.0",
        ignore_reinit_error=True,
        _temp_dir=str(ray_temp),
        runtime_env=runtime_env,
    )
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
        run_config = RunConfig(
            name=run_name,
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_uri,
                    experiment_name="lstm",
                    tags={"task": task, "networkx": str(networkx)},
                    save_artifact=True,
                )
            ],
        )
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=samples,
                max_concurrent_trials=2,
            ),
            run_config=run_config,
            param_space=param_space,
        )

    results = tuner.fit()
    logger.info("Results path at %s", results.experiment_path)
    best_result = results.get_best_result(metric, mode)
    trainable_instance = tune.with_parameters(
        trainable_cls,
        qdata=qdata,
        output_dir=output_dir,
        sequences_cache_dir=sequences_cache_dir,
    )(config=best_result.config)
    try:
        trainable_instance.test_result(best_result, metric, mode)
    except Exception:
        logger.error("Error testing best result")

    logger.info("Saving qdata at %s", Path(results.experiment_path) / "qdata.pkl")
    with open(Path(results.experiment_path) / "qdata.pkl", "wb") as f:
        pickle.dump(qdata, f)

    reporter.log_experiment_results(results, metric, mode, qdata, task, networkx)


lstm_group = click.Group("lstm", help="tools to train and tune lstm models")
lstm_group.add_command(tune_command)
