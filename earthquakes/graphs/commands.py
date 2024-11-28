import logging
from pathlib import Path

import click
import pandas as pd
from data.data import EarthquakeData
from data.grid import Grid
from settings import read_coordinates

from .link_prediction import run_link_prediction
from .link_prediction_nn import run_link_prediction as run_link_prediction_nn
from .link_prediction_tune import tune_link_prediction

logger = logging.getLogger(__name__)


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


@click.command()
@click.option("-f", "--file", type=str, help="Catalog csv file to turn into edge list")
@click.option("-d", "--distance", type=float, help="Distance in km for the grid cell size", default=100)
@click.option(
    "--features",
    multiple=True,
    help="features to be used in the model",
    default=["latitude", "longitude"],
    callback=lambda ctx, param, value: list(value),
)
def edge_list(file, distance, features):
    """
    Convert catalog csv file to edge list

    ## Usage

    quakes graphs edge-list -f csv/9923ce9a42736848b544e335a4d7c5fb.csv -d 30
    quakes graphs edge-list -f csv/9923ce9a42736848b544e335a4d7c5fb.csv -d 50
    quakes graphs edge-list -f csv/9923ce9a42736848b544e335a4d7c5fb.csv -d 100
    """
    file_path = Path(file)
    assert file_path.exists()

    latitude, longitude = read_coordinates()
    df = pd.read_csv(file_path)
    grid = Grid(latitude, longitude, distance)
    data = EarthquakeData(df, features=features, targets=[], grid=grid)
    logger.info("Processing data with nodes")
    nodes = data.data["node"].values

    logger.info("Saving edges list to csv")
    # nodes are ordered by time 0->1 means 1 happened after 0
    # source nodes[:-1] takes from 0 to n-1
    # target nodes[1:] takes from 1 to n
    pd.DataFrame(zip(nodes[:-1], nodes[1:]), columns=["source", "target"]).to_csv(
        f"csv/edges_{int(distance)}_{data.hash}.csv", index=False
    )
    data.data.to_csv(f"csv/clean_{int(distance)}_{data.hash}.csv", index=False)


@click.command()
@click.option("-f", "--file", type=str, help="Csv containing a list of graph edges")
@click.option("-nn", "--neural-network", type=bool, help="wheter to use neural networks", default=False)
def link_prediction(file, neural_network):
    """
    Takes an edge_list generated .csv file and runs the Stamille's Graph Machine Learning Book, Link prediciton algorithm

    ## Usage

    quakes graphs link-prediction -f csv/edges_100_95fb6d07e15e056a498ec10f366fbe4c.csv
    quakes graphs link-prediction -f csv/edges_100_95fb6d07e15e056a498ec10f366fbe4c.csv -nn True

    ## Output

    for 10km grid distance

    ROC AUC score 0.7696390658174098
    Precission 0.7560483870967742
    Recall 0.7961783439490446
    F1 score 0.7755946225439504
    """
    roc, acc, recall, f1 = (run_link_prediction_nn if neural_network else run_link_prediction)(file, test_size=0.3)
    print("ROC AUC score", roc)
    print("Precission", acc)
    print("Recall", recall)
    print("F1 score", f1)


@click.command()
@click.option("-f", "--file", type=str, help="Csv containing a list of graph edges")
@click.option("-s", "--samples", type=int, help="samples", default=-1)
@click.option("-resume", "--resume", type=bool, help="resume experiment", default=False)
@click.option("-ex", "--experiment", type=str, help="resume experiment path", default=None)
def link_prediction_tune(file, samples, resume, experiment):
    """
    Takes an edge_list generated .csv file and runs the Stamille's Graph Machine Learning Book, Link prediciton algorithm

    ## Usage

    quakes graphs link-prediction-tune -f csv/edges_30_95fb6d07e15e056a498ec10f366fbe4c.csv -s 10
    quakes graphs link-prediction-tune -f csv/edges_50_95fb6d07e15e056a498ec10f366fbe4c.csv -s 10
    quakes graphs link-prediction-tune -f csv/edges_100_95fb6d07e15e056a498ec10f366fbe4c.csv -s 10

    ## Output

    for 10km grid distance
    """
    experiment_path = experiment
    if resume and not experiment_path:
        experiment_path = prompt_experiment()

    experiment_path = Path(experiment_path) if isinstance(experiment_path, str) else experiment_path
    roc, acc, recall, f1 = tune_link_prediction(file, test_size=0.3, samples=samples, experiment_path=experiment_path)
    print("ROC AUC score", roc)
    print("Precission", acc)
    print("Recall", recall)
    print("F1 score", f1)


graphs_group = click.Group("graphs", help="tools to train with graphs")
graphs_group.add_command(edge_list)
graphs_group.add_command(link_prediction)
graphs_group.add_command(link_prediction_tune)
