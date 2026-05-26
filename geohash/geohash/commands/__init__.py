"""CLI commands package."""

from .inspect import inspect_group
from .train import train_cmd

__all__ = ["train_cmd", "inspect_group"]
