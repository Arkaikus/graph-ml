---
name: geohash-train-run
description: Runs geohash LSTM training experiments and inspects artifacts. Use when training models, comparing runs, or interpreting metrics from quakes-geohash CLI.
---

# Geohash Train Run

## Quick start

```bash
cd geohash && uv sync
uv run quakes-geohash train --experiment-name baseline --epochs 12
uv run quakes-geohash inspect list-runs
uv run quakes-geohash inspect compare-runs baseline v2
```

## Key CLI flags

- `--window-mode temporal|spatial|hybrid`
- `--split-strategy temporal_event` (default) or `window_index` (debug only)
- `--input-mode full|numeric_only|geohash_only`
- `--geohash-encoding flat|hierarchical`

## Run artifacts (`.geohash-runs/{name}-{ts}/`)

| File | Purpose |
|------|---------|
| `config.json` | Full RunConfig snapshot |
| `metrics.json` | Loss curves, final metrics, baselines, beats_baseline |
| `preprocess.json` | stoi, scaler mean/std, feature cols |
| `model_config.json` | Architecture hyperparams |
| `model_final.pt` | Weights |
| `predictions.csv` | Test set targets vs predicted |

## Interpreting metrics

- Compare model RMSE to `baselines.persistence` in metrics.json.
- Negative R² means worse than predicting the mean — investigate split leakage or collapsed predictions.
- Val loss drives early stopping; test metrics are held-out final eval.
