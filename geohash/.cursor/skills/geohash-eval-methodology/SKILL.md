---
name: geohash-eval-methodology
description: Enforces correct train/val/test splits, leakage prevention, and baseline comparisons for geohash magnitude forecasting. Use when changing splits, metrics, baselines, ablations, or R² reporting.
---

# Geohash Eval Methodology

## Split algorithm (required default)

1. Sort catalog by `time_ms`.
2. Cut at train_ratio / val_ratio / test_ratio boundaries (default 70/10/20).
3. Build vocab from train geohashes only; OOV → `<UNK>`.
4. Build windows separately per split DataFrame.
5. Standardize numerics using train statistics only.

## Forbidden patterns

- `samples[:split_idx]` on overlapping sliding windows (window_index split) — leaks context.
- Global `df.diff()` deltas inside spatial windows — misaligned features.
- Early stopping on test set — use validation only.

## Baseline checklist

Every training run must compute on test windows:

- **mean**: train target mean
- **persistence**: last magnitude in window
- **linear_numeric**: lstsq on flattened numeric features

Store in `metrics.json` and set `beats_baseline.persistence` if model RMSE < baseline RMSE.

## Ablation

Use `--input-mode` to isolate geohash vs numeric contribution before claiming spatial encoding helps.
