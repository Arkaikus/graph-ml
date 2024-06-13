import click

from graphs.commands import edge_list
from graphs.commands import link_prediction
from lstm.commands import tune_lstm
from processing.commands import download_usgs


def main():
    group = click.Group()
    group.add_command(download_usgs)
    group.add_command(edge_list)
    group.add_command(link_prediction)
    group.add_command(tune_lstm)
    group()


if __name__ == "__main__":
    main()
