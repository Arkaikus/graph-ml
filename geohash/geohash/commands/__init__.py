"""CLI commands package."""

from .inspect import inspect_group
from .predict import predict_cmd
from .train import train_cmd

__all__ = ["train_cmd", "predict_cmd", "inspect_group"]
