---
name: geohash-add-feature
description: Adds new earthquake features to the geohash LSTM pipeline. Use when adding feature augmentation, engineering columns, or extending numeric inputs.
---

# Geohash Add Feature

## Workflow

1. Add computation in `geohash/data/features.py`.
2. If static per event: include in `add_base_features` output columns.
3. If sequential delta: add to `compute_window_features(hist)` (first row = 0).
4. Append name to `_NUMERIC_COLS` in `geohash/data/dataset.py` (order matters).
5. Update `tests/test_data.py` with shape and value checks.
6. Document in `README.md` CLI/config section if user-facing.

## Spatial rule

Deltas must reflect **consecutive rows within the window** (sorted by `time_ms`), not global catalog order.

```python
hist = hist.sort_values("time_ms")
deltas = compute_window_features(hist)
x_num = np.column_stack([hist[_STATIC_COLS].values, deltas])
```

## Do not

- Add features via global `df.diff()` unless only used in deprecated paths.
- Change `_NUMERIC_COLS` order without updating saved `preprocess.json` schema notes.
