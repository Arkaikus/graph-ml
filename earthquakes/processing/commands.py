import click

from processing.usgs import USGS
from settings import read_coordinates


@click.command()
@click.option("-lt", "--latitude", type=tuple, help="Latitude range")
@click.option("-lg", "--longitude", type=tuple, help="Longitude range")
def download_usgs(latitude, longitude):
    """Download USGS data from given latitude and longitude"""
    latitude, longitude = read_coordinates(latitude, longitude)
    usgs = USGS(latitude, longitude)
    df = usgs.download()
    print(df)
