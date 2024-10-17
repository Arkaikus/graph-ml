# quakes lstm tune --samples 10

quakes lstm tune --samples 10 --networkx t --nx-features degree_centrality --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-features clustering --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-features betweenness_centrality --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-features closeness_centrality --node-size 50
quakes lstm tune --samples 10 --networkx t --nx-features pagerank --node-size 50

quakes lstm tune --samples 10 --networkx t --nx-features degree_centrality --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-features clustering --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-features betweenness_centrality --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-features closeness_centrality --node-size 100
quakes lstm tune --samples 10 --networkx t --nx-features pagerank --node-size 100

quakes lstm tune --samples 10 --networkx t --nx-features degree_centrality --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-features clustering --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-features betweenness_centrality --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-features closeness_centrality --node-size 150
quakes lstm tune --samples 10 --networkx t --nx-features pagerank --node-size 150