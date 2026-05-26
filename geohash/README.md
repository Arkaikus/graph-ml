# Geohash LSTM — Earthquake Magnitude Prediction

CLI-driven LSTM pipeline for predicting earthquake magnitudes using spatial geohash encoding and temporal features.

## Features

- **Data**: USGS earthquake catalog ingestion with geographic bounding box filtering
- **Spatial Encoding**: Geohash tokens for earthquake locations
- **Model**: LSTM combining geohash embeddings + numerical features (magnitude, depth, time deltas)
- **CLI**: Full train/inspect/compare workflow
- **Run Management**: Automatic artifact storage (config, metrics, predictions, plots, model weights)

## Installation

```bash
uv sync
```

## Quick Start

### 1. Train a Model

```bash
quakes-geohash train --experiment-name "baseline" --epochs 10 --learning-rate 1e-3
```

**Common Options:**
- `--experiment-name`: Identifier for run (default: auto-generated)
- `--epochs`: Training epochs (default: 12)
- `--batch-size`: Batch size (default: 64)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--hidden-size`: LSTM hidden dimension (default: 64)
- `--embedding-dim`: Geohash embedding dimension (default: 16)
- `--device`: `cpu` or `cuda` (default: `cpu`)
- `--min-lat`, `--max-lat`, `--min-lon`, `--max-lon`: Geographic bounds (CA by default)

### 2. List Previous Runs

```bash
quakes-geohash inspect list-runs
```

Shows all saved runs with final RMSE, MAE, and loss.

### 3. Inspect a Run

```bash
quakes-geohash inspect inspect-run baseline
```

Shows configuration, final metrics, predictions preview, and artifact locations.

### 4. Compare Multiple Runs

```bash
quakes-geohash inspect compare-runs baseline experiment_v2
```

Side-by-side metrics comparison; highlights best RMSE.

### 5. View Training Curves

```bash
quakes-geohash inspect plot-run baseline
```

Opens training curves (loss, RMSE, MAE, summary stats).

## Run Directory Structure

Runs are saved to `~/.geohash-runs/{experiment_name}-{YYYYMMDD_HHMMSS}/`:

```
baseline-20240526_143022/
├── config.json              # Full configuration snapshot
├── metrics.json             # train_loss, test_loss, rmse, mae per epoch + finals
├── predictions.csv          # Test predictions (target vs predicted)
├── training_curves.png      # Training history visualization
└── model_final.pt           # Model weights (PyTorch state dict)
```

## Architecture

```
geohash/
├── data/                    # Data fetching & preprocessing
│   ├── usgs.py             # USGS API client
│   ├── features.py         # Geohash encoding, feature engineering
│   └── dataset.py          # Windowing, collation, standardization
├── model/
│   └── lstm.py             # NextMagnitudeLSTM class
├── training/
│   ├── trainer.py          # Training loop
│   └── evaluator.py        # Evaluation metrics
├── commands/
│   ├── train.py            # 'train' command
│   └── inspect.py          # 'list-runs', 'inspect-run', 'compare-runs', 'plot-run'
├── config.py               # Pydantic configuration models
├── store.py                # Run storage & retrieval
├── utils.py                # Utilities (set_seed, etc.)
└── main.py                 # CLI entry point
```

## Configuration

All config can be set via CLI options or environment variables:

```bash
# Via CLI
quakes-geohash train --epochs 20 --learning-rate 5e-4

# Via environment (if implemented)
export GEOHASH_EPOCHS=20 GEOHASH_LR=5e-4
```

### Config Models (Pydantic)

- **USGSQueryParams**: Bounds, date range, magnitude filter, query limit
- **WindowConfig**: Sliding window min/max lengths, stride
- **GeohashConfig**: Geohash precision (1-12 characters)
- **ModelConfig**: Embedding dim, hidden size, num layers, dropout
- **TrainingConfig**: Batch size, epochs, learning rate, seed, device
- **ExperimentConfig**: Experiment name, output directory

## Testing

```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=geohash

# Run specific test
uv run pytest tests/test_config.py -v
```

**Coverage**: ≥80% of core modules (config, data, model, training, store)

Test modules:
- `test_config.py`: Pydantic validation
- `test_data.py`: Geohash encoding, feature engineering, windowing
- `test_model.py`: LSTM forward pass, shapes, gradients
- `test_trainer.py`: Training loop, evaluation
- `test_store.py`: Run storage/loading, artifact persistence

## Example Workflow

```bash
# Install
uv sync --extra dev

# Train baseline model
quakes-geohash train --experiment-name "v1" --epochs 5

# View results
quakes-geohash inspect inspect-run v1

# Train variant with different hyperparams
quakes-geohash train --experiment-name "v2" --hidden-size 128 --epochs 5

# Compare
quakes-geohash inspect compare-runs v1 v2

# View winner's training curves
quakes-geohash inspect plot-run v2
```

## Development

### Project Layout
```
geohash/
├── geohash/               # Package source
├── tests/                 # Test modules
├── pyproject.toml         # Dependencies, scripts, config
└── README.md             # This file
```

### Code Quality
- Type hints throughout
- Docstrings (Google/NumPy style)
- Constants at module level
- Logging instead of print()

### Adding Features
1. Update relevant module (data/, model/, training/, commands/)
2. Add tests in tests/
3. Run `pytest tests/ --cov` and ensure ≥80% coverage
4. Update README with new CLI commands/options

## Dependencies

**Core:**
- torch ≥2.0.0
- pandas ≥2.0.0
- numpy ≥1.24.0
- requests ≥2.31.0
- click ≥8.0.0
- matplotlib ≥3.5.0

**Dev:**
- pytest ≥7.0.0
- pytest-cov ≥4.0.0

## Troubleshooting

### "No events returned"
Widen geographic bounds or date range. Use `--min-lat`, `--max-lat`, `--min-lon`, `--max-lon`, or check USGS service.

### CUDA not available
Install CUDA-enabled PyTorch, or use `--device cpu` (default).

### Plot won't open
Check `~/.geohash-runs/{run_dir}/training_curves.png` directly.

## Future Extensions

- Ray Tune hyperparameter search
- MLflow experiment tracking
- Resume training from checkpoint
- Distributed multi-GPU training
- Interactive web dashboard
