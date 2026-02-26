# Load required libraries
library(tidygraph)
library(ggraph)
library(readr)

# Set your working directory if needed
# setwd("path/to/your/directory")

# Read edge list from CSV file
edges <- read.csv(
  "~/Documents/graph-ml/earthquakes/csv/edges_1_c4a6eecb25e66625f702075639ef551a.csv",
  header = TRUE
)

print(edges)

# Create a tidygraph object
graph <- tbl_graph(edges = edges, directed = TRUE)

print(graph)

# Plot the graph using ggraph
ggraph(graph, layout = "fr") +   # Fruchterman-Reingold layout
  geom_edge_link(aes(edge_alpha = 0.8), show.legend = FALSE) +  # Edges with transparency
  geom_node_point(color = "blue", size = 5) +  # Nodes
  # theme_void() +  # Remove background and axes
  ggtitle("Network Graph")  # Title of the graph
