import click

from graphs.commands import graphs_group
from lstm.commands import lstm_group
from processing.commands import usgs_group


def main():
    group = click.Group()
    group.add_command(graphs_group)
    group.add_command(lstm_group)
    group.add_command(usgs_group)
    group()


if __name__ == "__main__":
    main()
