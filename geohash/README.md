# Geohash LSTM — Earthquake Magnitude Prediction

CLI-driven LSTM pipeline for predicting earthquake magnitudes using spatial geohash encoding and spatiotemporal features.

## Features

- **Data**: USGS earthquake catalog ingestion with geographic bounding box filtering
- **Spatial Encoding**: Flat cell-level or hierarchical char-level geohash embeddings
- **Spatial Features**: Per-window deltas including `log1p(haversine)` inter-event distance
- **Windowing Modes**: Temporal (chronological), Spatial (cluster-based), or Hybrid
- **Evaluation**: Temporal event split (70/10/20 train/val/test), baselines, ablation modes
- **Model**: LSTM combining geohash embeddings + numerical features
- **Training**: Validation-based early stopping, LR scheduling, gradient clipping
- **Inference**: Full artifact bundle + `predict` CLI
- **CLI**: Train, predict, inspect, compare runs
- **AI guidance**: `.cursor/rules`, `.cursor/skills`, and `AGENTS.md`

## Installation

```bash
uv sync
```

## Quick Start

### Train

```bash
quakes-geohash train --experiment-name baseline --epochs 10
```

**Split & evaluation (default):**
- `--split-strategy temporal_event` — split events by time (recommended)
- `--train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2`
- `--split-strategy window_index` — legacy debug only (leaks overlapping windows)

**Model ablation:**
- `--input-mode full|numeric_only|geohash_only`

**Geohash encoding:**
- `--geohash-encoding flat|hierarchical`

**Windowing:**
- `--window-mode temporal|spatial|hybrid`
- `--spatial-radius-km 50 --temporal-window-days 30`

**Training:**
- `--early-stop-patience 5` (uses validation set)
- `--lr-scheduler plateau|cosine|none`
- `--gradient-clip 1.0`

### Predict

```bash
quakes-geohash predict \
  --run-dir .geohash-runs/baseline-20260526_120000 \
  --input-csv events.csv \
  --output predictions.csv
```

Input CSV must include: `time_ms`, `latitude`, `longitude`, `magnitude`, `depth_km`.

### Inspect runs

```bash
quakes-geohash inspect list-runs
quakes-geohash inspect inspect-run baseline
quakes-geohash inspect compare-runs baseline v2
quakes-geohash inspect plot-run baseline
```

`compare-runs` shows model RMSE, persistence baseline RMSE, and `beats_persist`.

## Run Directory Structure

Runs save to `.geohash-runs/{experiment_name}-{YYYYMMDD_HHMMSS}/`:

```
baseline-20260526_143022/
├── config.json
├── preprocess.json          # stoi, scaler, feature cols, encoding
├── model_config.json        # architecture hyperparams
├── metrics.json             # curves + baselines + beats_baseline
├── predictions.csv
├── training_curves.png
├── predictions_scatter.png
├── window_spot_check.png
└── model_final.pt
```

### metrics.json shape

```json
{
  "model": {"rmse": 0.4, "mae": 0.26, "r2": 0.05, "loss": 0.16},
  "baselines": {
    "mean": {"rmse": 0.5, "mae": 0.3, "r2": 0.0},
    "persistence": {"rmse": 0.45, "mae": 0.28, "r2": 0.02},
    "linear_numeric": {"rmse": 0.42, "mae": 0.27, "r2": 0.03}
  },
  "beats_baseline": {"persistence": true, "mean": true, "linear_numeric": false}
}
```

## Architecture

```
geohash/
├── data/           # USGS, features, split, windowing
├── model/          # NextMagnitudeLSTM, GeohashEncoder
├── training/       # trainer, evaluator, baselines
├── inference/      # artifact load/save
├── commands/       # train, predict, inspect
├── config.py
├── store.py
└── main.py
```

## AI agent source of truth

- **Rules:** `.cursor/rules/` — pipeline invariants, evaluation, model conventions
- **Skills:** `.cursor/skills/` — train-run, eval-methodology, add-feature, inference
- **Entry:** `AGENTS.md`

## Testing

```bash
uv run pytest tests/ -v --cov=geohash
```

Key test modules:
- `test_split.py` — temporal split, no leakage, OOV handling
- `test_data.py` — per-window deltas, windowing
- `test_baselines.py` — baseline metrics
- `test_inference.py` — artifact round-trip
- `test_model.py` — flat/hierarchical encoding, ablation modes

## Example workflow

```bash
uv sync --extra dev

# Default temporal event split + baselines
quakes-geohash train --experiment-name v1 --epochs 20

# Spatial windowing
quakes-geohash train --experiment-name v1-spatial \
  --window-mode spatial --spatial-radius-km 50 --epochs 20

# Ablation: numeric features only
quakes-geohash train --experiment-name v1-numeric --input-mode numeric_only

# Hierarchical geohash encoding
quakes-geohash train --experiment-name v1-hier --geohash-encoding hierarchical

quakes-geohash inspect compare-runs v1 v1-spatial v1-numeric
```

## Dependencies

- torch, pandas, numpy, requests, click, matplotlib, pydantic, scipy

## Troubleshooting

**No events returned** — widen bounds or date range.

**Empty val/test after split** — fetch more events or adjust `--train-ratio` / `--val-ratio` / `--test-ratio`.

**Negative R²** — compare against `baselines.persistence` in metrics.json; ensure `temporal_event` split is used.

**Legacy window_index split** — debug only; overlaps train/test contexts when stride=1.
