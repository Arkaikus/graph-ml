import click
from data.usgs import USGS
from settings import read_coordinates


@click.command()
@click.option(
    "-lt",
    "--latitude",
    type=str,
    default=None,
    help="Latitude range as 'min,max' (e.g. -0.132,9.796). Falls back to env LATITUDE.",
)
@click.option(
    "-lg",
    "--longitude",
    type=str,
    default=None,
    help="Longitude range as 'min,max' (e.g. -80.343,-72.466). Falls back to env LONGITUDE.",
)
def download(latitude, longitude):
    """Download USGS data from given latitude and longitude."""
    lat_tuple, long_tuple = read_coordinates(latitude, longitude)
    usgs = USGS(lat_tuple, long_tuple)
    df = usgs.download()
    print(df)


usgs_group = click.Group(name="usgs", help="tools to interact with USGS")
usgs_group.add_command(download)
