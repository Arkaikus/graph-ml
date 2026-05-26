# earthquakes package

## setup

Install the package with **uv** (recommended):

```bash
cd earthquakes
uv sync
```

Or with pip:

```bash
cd earthquakes
pip install -e .
```

Ensure dependencies are installed (see `earthquakes/pyproject.toml`). Run `uv sync` from the `earthquakes/` directory.

## download data with

```bash
quakes usgs download -lt "-0.132,9.796" -lg "-80.343,-72.466"
```

## run experiments with

```bash
quakes lstm tune --features latitude longitude depth mag --target mag --min-lat -0.132 --max-lat 9.796 --min-long -80.343 --max-long -72.466 --min-mag 0 --max-mag 10 --node-size 100 --metric loss --samples 10

quakes lstm tune --samples 100

# With network features (grid search over degree_centrality, clustering, betweenness_centrality, closeness_centrality, pagerank)
quakes lstm tune --samples 100 --networkx True --node-size 50
quakes lstm tune --samples 100 --networkx True --node-size 100
quakes lstm tune --samples 100 --networkx True --node-size 150
```

## troubleshooting

if `quakes` command is not available add this to `.bashrc`

```bash
# Add ~/.local/bin to PATH
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
```

If you installed with `uv sync`, run `uv run quakes --help` from the `earthquakes/` directory, or ensure the venv's bin is on your PATH.

if nvidia cuda is failing try

```bash
sudo modprobe --remove nvidia_uvm
sudo modprobe nvidia_uvm
```
