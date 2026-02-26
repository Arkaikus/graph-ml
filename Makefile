.PHONY: format lint lint-fix edge-list tune-graphs tune-graphs-resume tune-lstm tune-lstm-resume tune-lstm-networkx tune-lstm-networkx-resume
SHELL := /bin/bash

# Run from repo root. Use uv run so ruff from dev deps is used.
format:
	uv run ruff format earthquakes

lint:
	uv run ruff check earthquakes

lint-fix:
	uv run ruff check earthquakes --fix

clean:
	rm -rf ~/.cache/ray/tmp/*

# this will download the csv file to csv/9923ce9a42736848b544e335a4d7c5fb.csv (relative to cwd)
download-default:
	@echo "Downloading default data..."
	uv run quakes usgs download -lt "-0.132,9.796" -lg "-80.343,-72.466"

# Experiments (from former experiments/ scripts). Override CSV or EXPERIMENT_DIR as needed. Run from repo root.
CSV ?= csv/9923ce9a42736848b544e335a4d7c5fb.csv
EDGE_HASH ?= 95fb6d07e15e056a498ec10f366fbe4c

edge-list:
	uv run quakes graphs edge-list -f $(CSV) -d 30
	uv run quakes graphs edge-list -f $(CSV) -d 50
	uv run quakes graphs edge-list -f $(CSV) -d 100

tune-graphs:
	uv run quakes graphs link-prediction-tune -f csv/edges_30_$(EDGE_HASH).csv -s 10
	uv run quakes graphs link-prediction-tune -f csv/edges_50_$(EDGE_HASH).csv -s 10
	uv run quakes graphs link-prediction-tune -f csv/edges_100_$(EDGE_HASH).csv -s 10

tune-graphs-resume:
	@test -n "$(EXPERIMENT_DIR)" || (echo "Usage: make tune-graphs-resume EXPERIMENT_DIR=~/ray_results/GNNTrainable_..." ; exit 1)
	uv run quakes graphs link-prediction-tune -f csv/edges_30_$(EDGE_HASH).csv -s 10 -ex $(EXPERIMENT_DIR)
	uv run quakes graphs link-prediction-tune -f csv/edges_50_$(EDGE_HASH).csv -s 10 -ex $(EXPERIMENT_DIR)
	uv run quakes graphs link-prediction-tune -f csv/edges_100_$(EDGE_HASH).csv -s 10 -ex $(EXPERIMENT_DIR)

tune-lstm:
	uv run --active quakes lstm tune --quantiles 2 --samples 10 --metric accuracy --mode max
	uv run --active quakes lstm tune --quantiles 3 --samples 10 --metric accuracy --mode max
	uv run --active quakes lstm tune --quantiles 4 --samples 10 --metric accuracy --mode max

tune-lstm-resume:
	@test -n "$(EXPERIMENT_DIR)" || (echo "Usage: make tune-lstm-resume EXPERIMENT_DIR=~/ray_results/ClassificationTrainable_..." ; exit 1)
	uv run quakes lstm tune --quantiles 2 --samples 10 --metric accuracy --mode max -ex $(EXPERIMENT_DIR)
	uv run quakes lstm tune --quantiles 3 --samples 10 --metric accuracy --mode max -ex $(EXPERIMENT_DIR)
	uv run quakes lstm tune --quantiles 4 --samples 10 --metric accuracy --mode max -ex $(EXPERIMENT_DIR)

tune-lstm-networkx:
	uv run quakes lstm tune --quantiles 2 --samples 100 --metric accuracy --mode max --networkx
	uv run quakes lstm tune --quantiles 3 --samples 100 --metric accuracy --mode max --networkx
	uv run quakes lstm tune --quantiles 4 --samples 100 --metric accuracy --mode max --networkx

tune-lstm-networkx-resume:
	@test -n "$(EXPERIMENT_DIR)" || (echo "Usage: make tune-lstm-networkx-resume EXPERIMENT_DIR=~/ray_results/ClassificationTrainable_..." ; exit 1)
	uv run quakes lstm tune --quantiles 2 --samples 100 --metric accuracy --mode max --networkx -ex $(EXPERIMENT_DIR)
	uv run quakes lstm tune --quantiles 3 --samples 100 --metric accuracy --mode max --networkx -ex $(EXPERIMENT_DIR)
	uv run quakes lstm tune --quantiles 4 --samples 100 --metric accuracy --mode max --networkx -ex $(EXPERIMENT_DIR)
