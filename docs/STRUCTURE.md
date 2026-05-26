# Project organization: rating and transition plan

## Current structure (brief)

```
earthquakes/
  main.py, config.py, settings.py, setup.py   # entry + app config at root
  data/          # ingestion, cleaning, grid, cache (data/commands = usgs CLI)
  graphs/        # edge list, splitter, link prediction, tune
  lstm/          # classification, regression, tune, plot, utils
  tests/
```

## Rating: **6.5 / 10**

**Strengths**

- **Clear domain split:** `data`, `graphs`, `lstm` map to ingestion/data, graph pipeline, and LSTM pipeline. Easy to find where things live.
- **Single CLI entry:** `main.py` + Click groups; subcommands live next to their domain (`data/commands`, `graphs/commands`, `lstm/commands`).
- **Config centralized:** `config.py` (Pydantic) and `settings.py` give one place for validated app/run config.
- **Tests colocated:** `tests/` with pytest and `pythonpath = ["."]` is straightforward.

**Weaknesses**

- **Root clutter:** `main.py`, `config.py`, `settings.py` at repo root mix “package layout” with “app entry”. Anyone looking for “the package” sees both app and library surface.
- **Naming confusion:** `data/commands.py` is the **usgs** CLI, not “data commands” in general. `data/graphs.py` is graph **construction** (nodes2graph, networkx_property) used by the data layer, not the graph **pipeline** (which lives in `graphs/`).
- **No explicit public API:** No `earthquakes/__init__.py` or `earthquakes/api.py` that re-exports public types (e.g. `EarthquakeData`, `Grid`, `RunConfig`). Imports are “from data.x”, “from validaiton”, etc., so the package is not clearly a single importable unit.
- **Large modules:** `data/data.py`, `graphs/link_prediction_tune.py`, `lstm/classification.py` do a lot; splitting would improve readability and testability.
- **Experiments/notebooks:** `lstm/examples/` and `experiments/*.sh` are at the edge of the layout; not clearly “scripts” vs “library”.

## Proposed transition plan (incremental)

### Phase A – Low risk (no import renames)

1. **Rename for clarity (optional but recommended)**  
   - `data/commands.py` → `data/usgs_commands.py` (or keep and add a short docstring: “USGS download CLI”).  
   - Add a one-line comment at top of `data/graphs.py`: “Graph construction helpers for the data layer (nodes2graph, networkx_property), not the graph training pipeline in graphs/.”

2. **Document public surface**  
   - Add `docs/public-api.md` (or a section in ARCHITECTURE.md) listing the main types and functions that external code or scripts should use (e.g. `EarthquakeData`, `Grid`, `RunConfig`, `load_run_config`, CLI entry point). No code move yet.

3. **Keep root layout for now**  
   - Leaving `main.py`, `config.py`, `settings.py` at root is acceptable while the package is a single app+library. If the repo later becomes multi-package, move them under an `earthquakes/` package dir (see Phase C).

### Phase B – Medium risk (optional splits)

4. **Split large modules**  
   - `data/data.py`: consider extracting “cleaning” (e.g. `data/cleaning.py`) and “sequences” (e.g. `data/sequences.py`) if they grow or are tested separately.  
   - `graphs/link_prediction_tune.py`: split into “trainable + tune runner” vs “plotting + metrics writing” so the tune script is easier to follow.  
   - `lstm/classification.py`: consider moving “training loop” vs “checkpoint/plot” into smaller functions or a helper module.

5. **Scripts vs library**  
   - Move `experiments/*.sh` into a single `scripts/` or `experiments/` dir and document in README that these are example invocations.  
   - Keep `lstm/examples/` as notebooks; add a short README there describing they are for exploration/demos.

### Phase C – Higher risk (layout change)

6. **Single-package layout (only if needed)**  
   - If the repo should be installable as a single package `earthquakes` with `import earthquakes` and submodules:  
     - Introduce `earthquakes/` as the package directory.  
     - Move `main.py` → `earthquakes/__main__.py` or keep as script that does `from earthquakes.main import main`.  
     - Move `config.py`, `settings.py` → `earthquakes/config.py`, `earthquakes/settings.py`.  
     - Move `data/`, `graphs/`, `lstm/` under `earthquakes/` and fix all imports and `pyproject.toml` (e.g. `packages = ["earthquakes"]`, script entry `quakes = earthquakes.main:main`).  
   - This is a one-time, coordinated refactor (run tests and CLI after each step).

7. **Optional src layout**  
   - For stricter “no import from repo root” discipline, use `src/earthquakes/` and install in editable mode so only `earthquakes` is on PYTHONPATH. Defer until the team wants that.

## Recommendation

- **Short term:** Do Phase A (naming/docs, no structural change). Optionally do Phase B.4 (split one or two large files) when touching those areas.
- **Later:** Consider Phase C only if you need a cleaner `import earthquakes` story or multiple installable targets; otherwise the current flat layout is acceptable for a single-app pipeline.
