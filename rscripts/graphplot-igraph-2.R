# Load necessary libraries
library(igraph)
library(RColorBrewer)

# Set the working directory
# setwd("path_to_your_directory")

# Read the CSV file
edges <- read.csv(
  "~/Documents/graph-ml/earthquakes/csv/edges_100_c4a6eecb25e66625f702075639ef551a.csv"
)

# Create a graph
G <- graph_from_data_frame(edges, directed = TRUE)

# Calculate node degrees
degrees <- igraph::degree(G)

# Calculate node colors based on edge count
node_colors <- brewer.pal(8, "Dark2")[cut(
  degrees,
  breaks = 8,
  labels = 1:8,
  include.lowest = TRUE
)]

# Set node sizes based on degrees
node_sizes <- degrees

# Plot the graph
plot(
  G,
  vertex.size = node_sizes,
  vertex.color = node_colors,
  vertex.label = V(G)$name,
  main = "Graph Characteristics"
)