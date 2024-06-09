import click

from graphs.commands import edge_list
from graphs.link_forecast import link_forecast
from lstm.commands import train_lstm
from processing.commands import download_usgs


def main():
    group = click.Group()
    group.add_command(download_usgs)
    group.add_command(edge_list)
    group.add_command(link_forecast)
    group.add_command(train_lstm)
    group()


if __name__ == "__main__":
    main()
