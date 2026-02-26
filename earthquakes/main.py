# Use non-interactive backend before any matplotlib imports.
# Prevents "RuntimeError: main thread is not in main loop" when Ray workers
# create/close figures (tkinter cleanup fails in worker processes).
import matplotlib

matplotlib.use("Agg")

import click

from data.commands import usgs_group
from graphs.commands import graphs_group
from lstm.commands import lstm_group


def main():
    group = click.Group()
    group.add_command(graphs_group)
    group.add_command(lstm_group)
    group.add_command(usgs_group)
    group()


if __name__ == "__main__":
    main()
