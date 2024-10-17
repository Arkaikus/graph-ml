# quakes lstm tune --samples 10

quakes lstm tune --samples 10 --networkx t --nx-feature degree_centrality --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-feature clustering --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-feature betweenness_centrality --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-feature closeness_centrality --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-feature pagerank --node-size 50

quakes lstm tune --samples 10 --networkx t --nx-feature degree_centrality --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-feature clustering --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-feature betweenness_centrality --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-feature closeness_centrality --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-feature pagerank --node-size 100

quakes lstm tune --samples 10 --networkx t --nx-feature degree_centrality --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-feature clustering --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-feature betweenness_centrality --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-feature closeness_centrality --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-feature pagerank --node-size 150