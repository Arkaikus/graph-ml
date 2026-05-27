"""Geohash LSTM earthquake prediction pipeline CLI."""

import logging

import click

from geohash.commands import inspect_group, predict_cmd, train_cmd
from geohash.commands.inspect import (
    compare_runs_cmd,
    inspect_run_cmd,
    list_runs_cmd,
    plot_run_cmd,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@click.group()
@click.version_option()
def cli():
    """Geohash-based LSTM for earthquake magnitude prediction."""
    pass


cli.add_command(train_cmd)
cli.add_command(predict_cmd)
cli.add_command(inspect_group)
cli.add_command(list_runs_cmd)
cli.add_command(inspect_run_cmd)
cli.add_command(compare_runs_cmd)
cli.add_command(plot_run_cmd)


if __name__ == "__main__":
    cli()
