---
name: geohash-inference
description: Loads saved geohash run artifacts and runs magnitude prediction. Use when implementing predict CLI, deploying models, or reloading preprocess bundles.
---

# Geohash Inference

## Artifact bundle

Required files in run directory:

- `model_final.pt` — state dict
- `model_config.json` — vocab_size, embedding_dim, hidden_size, num_layers, dropout, input_mode, encoding
- `preprocess.json` — stoi, numeric_mean, numeric_std, numeric_cols, geohash_precision, encoding, input_mode

Load via `geohash.inference.artifacts.load_run_bundle(run_dir)`.

## Predict CLI

```bash
uv run quakes-geohash predict --run-dir .geohash-runs/baseline-20260526_120000 \
  --input-csv events.csv --output predictions.csv
```

## OOV handling

- Flat encoding: unknown geohash strings map to `stoi["<UNK>"]`.
- Hierarchical: unknown chars map to char-level UNK id.
- Apply same z-score (train mean/std) from preprocess.json.

## Reconstruct model

```python
from geohash.inference.artifacts import load_run_bundle, build_model_from_bundle
bundle = load_run_bundle(run_dir)
model = build_model_from_bundle(bundle)
model.load_state_dict(bundle["model_state_dict"])
```
