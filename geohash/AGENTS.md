# AGENTS.md – geohash LSTM package

CLI-driven LSTM pipeline for earthquake magnitude prediction using geohash embeddings and spatiotemporal features.

## Source of truth for AI agents

- **Rules:** `.cursor/rules/` — core, data pipeline, evaluation, model conventions
- **Skills:** `.cursor/skills/` — train-run, eval-methodology, add-feature, inference workflows

Read the relevant skill before changing splits, features, baselines, or inference.

## Layout

| Path | Role |
|------|------|
| `geohash/main.py` | Click CLI entry |
| `geohash/config.py` | Pydantic RunConfig |
| `geohash/data/` | USGS fetch, features, split, windowing |
| `geohash/model/lstm.py` | NextMagnitudeLSTM + GeohashEncoder |
| `geohash/training/` | trainer, evaluator, baselines |
| `geohash/inference/` | artifact load/save, predict helpers |
| `geohash/store.py` | Run persistence |
| `geohash/commands/` | train, inspect, predict |
| `tests/` | pytest |

## Commands

```bash
cd geohash && uv sync
uv run quakes-geohash train --experiment-name baseline
uv run quakes-geohash predict --run-dir .geohash-runs/baseline-...
uv run pytest tests/ -v
```

## Critical invariants

1. **Temporal event split** by default — never window-index split in production.
2. **Per-window deltas** for all window modes.
3. **Vocab from train only**; OOV → `<UNK>`.
4. **Baselines** stored in every run's `metrics.json`.
5. Artifacts: `preprocess.json` + `model_config.json` alongside weights.

## Outputs

Runs save to `.geohash-runs/{experiment_name}-{timestamp}/`.

See `README.md` for full CLI options and artifact tree.
