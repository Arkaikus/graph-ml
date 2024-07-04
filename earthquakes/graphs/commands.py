import logging
from pathlib import Path

import click
import pandas as pd

from data.data import EarthquakeData
from data.grid import Grid
from settings import read_coordinates

from .link_prediction import run_link_prediction

logger = logging.getLogger(__name__)


@click.command()
@click.option("-f", "--file", type=str, help="Catalog csv file to turn into edge list")
@click.option("-d", "--distance", type=float, help="Distance in km for the grid cell size", default=100)
def edge_list(file, distance):
    """Convert catalog csv file to edge list"""
    file_path = Path(file)
    assert file_path.exists()

    latitude, longitude = read_coordinates()
    df = pd.read_csv(file_path)
    grid = Grid(latitude, longitude, distance)
    data = EarthquakeData(
        df,
        numeric_columns=["latitude", "longitude", "depth", "mag"],
        time_column=True,
        scaler_mode=None,
    )
    logger.info("Processing data with nodes")
    result, _ = data.process(grid)
    nodes = result["node"].values

    logger.info("Saving edges list to csv")
    # nodes are ordered by time 0->1 means 1 happened after 0
    # source nodes[:-1] takes from 0 to n-1
    # target nodes[1:] takes from 1 to n
    pd.DataFrame(zip(nodes[:-1], nodes[1:]), columns=["source", "target"]).to_csv(
        f"csv/edges_{int(distance)}_{data.hash}.csv", index=False
    )
    result.to_csv(f"csv/clean_{int(distance)}_{data.hash}.csv", index=False)


@click.command()
@click.option("-f", "--file", type=str, help="Csv containing a list of graph edges")
def link_prediction(file):
    """
    Takes an edge_list generated .csv file and runs the Stamille's Graph Machine Learning Book, Link prediciton algorithm
    """
    roc, acc, recall, f1 = run_link_prediction(file, test_size=0.3)
    print("ROC AUC score", roc)
    print("Precission", acc)
    print("Recall", recall)
    print("F1 score", f1)


graphs_group = click.Group("graphs", help="tools to train with graphs")
graphs_group.add_command(edge_list)
graphs_group.add_command(link_prediction)
