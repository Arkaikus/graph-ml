"""Test CLI command registration."""

from geohash.main import cli


def test_cli_exposes_top_level_run_commands():
    """Top-level commands should match the planned user-facing CLI."""
    command_names = set(cli.commands)

    assert "train" in command_names
    assert "inspect" in command_names
    assert "list-runs" in command_names
    assert "inspect-run" in command_names
    assert "compare-runs" in command_names
    assert "plot-run" in command_names