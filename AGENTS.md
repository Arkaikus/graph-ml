# AGENTS.md – guidance for LLMs

This file helps AI agents navigate and modify the **earthquakes** package.

## What this repo is

CLI-driven ML pipeline for earthquake catalog data: USGS ingestion, graph link prediction (Node2Vec + RF/NN), and LSTM classification/regression. Hyperparameter search uses Ray Tune.

## Layout (key paths)

- **Project root:** `earthquakes/pyproject.toml`, `earthquakes/uv.lock` for the earthquakes app; `geohash/pyproject.toml`, `geohash/uv.lock` for the geohash app; `Makefile` at repo root dispatches into the app directory.
- **Entry:** `earthquakes/main.py` → Click group; console script `quakes` from `earthquakes/pyproject.toml`.
- **CLI commands:** `earthquakes/data/commands.py` (usgs), `earthquakes/graphs/commands.py` (graphs), `earthquakes/lstm/commands.py` (lstm).
- **Config:** `earthquakes/config.py` (Pydantic: RunConfig, GridConfig, USGSQueryParams); `earthquakes/settings.py` (load from env + overrides).
- **Data:** `earthquakes/data/data.py` (EarthquakeData), `earthquakes/data/grid.py` (Grid, uses GridConfig), `earthquakes/data/usgs.py` (USGS client), `earthquakes/data/store.py` (cache), `earthquakes/data/hash.py` (Hashable for cache keys).
- **Graph pipeline:** `earthquakes/graphs/edge_splitter.py`, `earthquakes/graphs/link_prediction.py`, `earthquakes/graphs/link_prediction_nn.py`, `earthquakes/graphs/link_prediction_tune.py`, `earthquakes/graphs/model.py`.
- **LSTM:** `earthquakes/lstm/model.py` (LSTMModel), `earthquakes/lstm/trainable/base.py` (BaseLSTMTrainable), `earthquakes/lstm/classification/` (ClassificationTrainable), `earthquakes/lstm/regression/` (RegressionTrainable), `earthquakes/lstm/plot.py`, `earthquakes/lstm/utils.py`.
- **Tests:** `earthquakes/tests/` (pytest; `pythonpath = ["earthquakes"]` via pyproject.toml).

## Running things

- Install: `cd earthquakes && uv sync` (or `pip install -e .`). Run CLI: `cd earthquakes && uv run quakes --help` or `quakes` if venv on PATH.
- Lint/format: `make lint`, `make lint-fix`, `make format` (Makefile dispatches into `earthquakes/`).
- Tests: `cd earthquakes && uv run pytest tests/ -v`.

## Conventions

- Use **Pydantic** in `config.py` for all new config/validation; wire CLI or callers to it.
- **Grid** is always constructed via **GridConfig** (no fallback).
- Outputs go under **`plots/`** (or `RunConfig.output_dir`). Don’t introduce new top-level output dirs without aligning with config.
- Ray: pass large data via **`tune.with_parameters`**; keep **param_space** to scalars; set **RunConfig(name=...)**, **max_concurrent_trials**.
- Catch **specific exceptions** (no bare `except:`). Use **`isinstance`** for type checks.

## Docs to read first

- **ARCHITECTURE.md** – data flow, components, and dependencies.
- **CHANGELOG.md** – what was changed in the upgrade.
- **README.md** – setup and example commands.
