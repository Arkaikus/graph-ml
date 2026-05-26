## Plan: Geohash Pipeline Refactoring

**TL;DR**: Break the monolithic main.py into a modular package with Pydantic config, Click CLI (train/list-runs/inspect-run/compare-runs/plot-run), persistent run storage (`~/.geohash-runs/{experiment_name}-{datetime}/`), and comprehensive pytest coverage. Follow the proven patterns from earthquakes package.

---

### **Steps**

#### **Phase 1: Config & Structure (Foundation)**
1. Create `geohash/config.py` with Pydantic models:
   - `USGSQueryParams`, `TrainingConfig`, `WindowConfig`, `GeohashConfig`, `ExperimentConfig`
   - Validators for bounds (min < max), positive values
2. Reorganize existing functions into modules:
   - `geohash/data/` (usgs.py, features.py, dataset.py)
   - `geohash/model/lstm.py`
   - `geohash/utils.py`
3. Add `geohash/store.py` — `RunStore` class for saving/loading runs to `~/.geohash-runs/`
   - Saves: config.json, metrics.json, predictions.csv, training_curves.png, model_final.pt

**Dependency**: None (foundation layer)

#### **Phase 2: CLI Commands** (*depends on Phase 1*)
1. Refactor main.py as Click Group entry point
2. Create `geohash/commands/train.py` — `train` subcommand (accepts all config as CLI args, calls `RunStore.save_run()`)
3. Create `geohash/commands/inspect.py` — subcommands:
   - `list-runs` (table of all runs with RMSE/MAE)
   - `inspect-run` (full config + metrics for one run)
   - `compare-runs` (side-by-side metrics for 2+ runs)
   - `plot-run` (regenerate or view training curves)

**Dependency**: Phase 1 (config + store)

#### **Phase 3: Testing & Code Quality** (*parallel with Phase 2*)
1. `tests/test_config.py` — Pydantic validation, bounds checking
2. `tests/test_data.py` — Geohash encoding, feature engineering, windowing
3. `tests/test_model.py` — Model init, forward pass shapes, packed sequences
4. `tests/test_trainer.py` — Single epoch, gradient flow, eval mode
5. `tests/test_store.py` — Save/load run artifacts, list runs
6. Add type hints throughout; convert print() → logging; add docstrings

**Dependency**: Phase 1 (tests import from modules)

#### **Phase 4: Integration & Polish** (*depends on Phase 2 & 3*)
1. Update pyproject.toml — add console script `quakes-geohash = geohash.main:cli`
2. Update README.md — new CLI usage, examples, config reference
3. Handle edge cases — device fallback, run collision, graceful errors

**Dependency**: Phase 2 & 3

---

### **Relevant Files**
- main.py — Replace with Click entry point; move logic to modules
- pyproject.toml — Update: console script, pytest config, version
- [earthquakes/config.py](.validation.py) — Reference for Pydantic patterns
- [earthquakes/data/store.py](.store.py) — Reference for storage class
- [earthquakes/lstm/utils.py](.utils.py) — Reference for run inspection (`prompt_experiment`)
- [earthquakes/main.py](.main.py) — Reference for Click group structure

---

### **New Directory Structure**
```
geohash/
├── geohash/
│   ├── config.py          # Pydantic models
│   ├── store.py           # RunStore class
│   ├── main.py            # CLI entry (Click Group)
│   ├── data/
│   │   ├── usgs.py
│   │   ├── features.py
│   │   └── dataset.py
│   ├── model/
│   │   └── lstm.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── commands/
│   │   ├── train.py
│   │   └── inspect.py
│   └── utils.py
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_trainer.py
│   └── test_store.py
└── README.md (updated)

Runs stored in: ~/.geohash-runs/{experiment_name}-{datetime}/
  ├── config.json
  ├── metrics.json
  ├── predictions.csv
  ├── training_curves.png
  └── model_final.pt
```

---

### **Verification**
1. **Automated**: `pytest geohash/tests/ -v --cov=geohash` → ≥80% coverage, all pass
2. **CLI**: 
   - `uv run quakes-geohash train --epochs 1` → creates run with all artifacts
   - `uv run quakes-geohash list-runs` → shows saved run(s)
   - `uv run quakes-geohash inspect-run {run_name}` → displays metrics + path to model
   - `uv run quakes-geohash compare-runs {run1} {run2}` → side-by-side table
3. **Manual**: Train run → inspect → compare multiple runs

---

### **Key Decisions**
- **Standalone package**: Keeps geohash independent; avoids coupling to earthquakes changes
- **JSON + PNG storage**: Human-readable metadata, viewable plots; follows earthquakes pattern
- **{experiment_name}-{datetime} naming**: Sortable, human-friendly, avoids UUID opacity
- **Pydantic + Click**: Type-safe config, automatic help text, early validation
- **Store class**: Encapsulates I/O; easier testing and extensibility

---