import click

from processing.usgs import USGS
from settings import read_coordinates


@click.command()
@click.option("-lt", "--latitude", type=tuple, help="Latitude range")
@click.option("-lg", "--longitude", type=tuple, help="Longitude range")
def download(latitude, longitude):
    """Download USGS data from given latitude and longitude"""
    latitude, longitude = read_coordinates(latitude, longitude)
    usgs = USGS(latitude, longitude)
    df = usgs.download()
    print(df)


@click.command()
@click.option("-f", "--file", type=str, help="csv earthquake catalog to be processed")
def plot_geo(file):
    """Takes a csv catalog, processes it and plots the contents"""
    import geopandas as gpd
    import geoplot as gplt
    import geoplot.crs as gcrs
    import matplotlib.pyplot as plt
    import pandas as pd
    from shapely.geometry import Point

    data = pd.read_csv(file)

    # Create a GeoDataFrame
    geometry = [Point(xy) for xy in zip(data["longitude"], data["latitude"])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry)

    # Set up the plot
    size = (20, 20)
    # plt.figure(figsize=size)
    world: gpd.GeoDataFrame = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Plot the base map
    ax = world.plot(figsize=size, color="white", edgecolor="black")

    # Plot the earthquake data
    gplt.pointplot(gdf, ax=ax, scale="mag", hue="depth", legend=True, cmap="viridis")

    # Add titles and labels
    plt.title("Earthquake Locations and Magnitudes")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.gcf().savefig(f"{file}.png")


usgs_group = click.Group(name="usgs", help="tools to interact with USGS")
usgs_group.add_command(download)
usgs_group.add_command(plot_geo)
