import logging
from pathlib import Path

import click
from lstm.plot import plot_analysis

logger = logging.getLogger(__name__)


def split_n_parse(string: str, _type: type):
    return [_type(part) for part in string.split(",") if part]


def prompt_experiment():
    ray_results = Path.home() / "ray_results"
    folders = {
        idx: folder
        for idx, folder in enumerate(
            ray_results.glob("*"),
        )
        if folder.is_dir()
        if folder.stem[0].isalpha()
    }
    prompt = "\n".join(f"{idx}) {folder.stem}" for idx, folder in folders.items())
    choice = click.prompt(prompt, type=int, default=None)
    assert choice is not None, choice
    return folders.get(choice)


__all__ = ["split_n_parse", "prompt_experiment", "plot_analysis"]
