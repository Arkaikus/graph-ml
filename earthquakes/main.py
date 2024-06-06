import click
import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from processing.data import DataBuilder
from processing.grid import Grid

from usgs import USGS
from commands.link_forecast import link_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_coordinates(latitude=None, longitude=None):
    load_dotenv(override=True)
    latitude = latitude or os.getenv("LATITUDE", "").split(",")
    longitude = longitude or os.getenv("LONGITUDE", "").split(",")
    return latitude, longitude


@click.command()
@click.option("-lt", "--latitude", type=tuple, help="Latitude range")
@click.option("-lg", "--longitude", type=tuple, help="Longitude range")
def download_usgs(latitude, longitude):
    """Download USGS data from given latitude and longitude"""
    latitude, longitude = read_coordinates(latitude, longitude)
    usgs = USGS(latitude, longitude)
    df = usgs.download()
    print(df)


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
    processor = DataBuilder(numeric_columns=["latitude", "longitude", "depth", "mag"], time_column=True)
    logger.info("Processing data")
    data = processor.build_data(df, grid)
    nodes = data["node"].values
    logger.info("Saving edges list to csv")
    pd.DataFrame(zip(nodes[:-1], nodes[1:]), columns=["target", "source"]).to_csv(
        f"csv/edges_{int(distance)}_{processor.hash}.csv", index=False
    )


if __name__ == "__main__":
    group = click.Group()
    group.add_command(download_usgs)
    group.add_command(edge_list)
    group.add_command(link_forecast)
    group()
