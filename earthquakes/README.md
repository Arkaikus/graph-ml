# earthquakes package

## setup

install this package locally with `pip install -Ie .

## run experiments with

```bash

quakes lstm tune --features latitude longitude depth mag --target mag --min-lat -0.132 --max-lat 9.796 --min-long -80.343 --max-long -72.466 --min-mag 0 --max-mag 10 --node-size 100 --metric loss --samples 10

quakes lstm tune --samples 100

quakes lstm tune --samples 100 --netwrokx t --nx-feature degree_centrality --node-size 50
quakes lstm tune --samples 100 --netwrokx t --nx-feature clustering --node-size 50
quakes lstm tune --samples 100 --netwrokx t --nx-feature betweenness_centrality --node-size 50
quakes lstm tune --samples 100 --netwrokx t --nx-feature closeness_centrality --node-size 50
quakes lstm tune --samples 100 --netwrokx t --nx-feature pagerank --node-size 50

quakes lstm tune --samples 100 --netwrokx t --nx-feature degree_centrality --node-size 100
quakes lstm tune --samples 100 --netwrokx t --nx-feature clustering --node-size 100
quakes lstm tune --samples 100 --netwrokx t --nx-feature betweenness_centrality --node-size 100
quakes lstm tune --samples 100 --netwrokx t --nx-feature closeness_centrality --node-size 100
quakes lstm tune --samples 100 --netwrokx t --nx-feature pagerank --node-size 100

quakes lstm tune --samples 100 --netwrokx t --nx-feature degree_centrality --node-size 150
quakes lstm tune --samples 100 --netwrokx t --nx-feature clustering --node-size 150
quakes lstm tune --samples 100 --netwrokx t --nx-feature betweenness_centrality --node-size 150
quakes lstm tune --samples 100 --netwrokx t --nx-feature closeness_centrality --node-size 150
quakes lstm tune --samples 100 --netwrokx t --nx-feature pagerank --node-size 150


```

## troubleshooting

if `quakes` command is not available add this to `.bashrc`

```bash
# Add ~/.local/bin to PATH
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi

```

if nvidia cuda is failing try

```bash
sudo modprobe --remove nvidia_uvm
sudo modprobe nvidia_uvm
```