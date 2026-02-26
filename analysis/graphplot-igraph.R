# Load required libraries
library(igraph)


# Read edge list from CSV file
edges <- read.csv(
  "~/Documents/graph-ml/earthquakes/csv/edges_1_c4a6eecb25e66625f702075639ef551a.csv"
)

# Create graph object from edge list
g <- graph_from_data_frame(edges, directed = TRUE)  # Adjust directed as needed

# Plot the graph
plot(
  g,
  # Layout algorithm for node positioning
  layout = layout_with_fr(g),
  # Node color
  vertex.color = "lightblue",
  # Node size
  vertex.size = 1,
  # Label size
  vertex.label = NA,
  # edge.width = 10,       # Arrow size (if directed)
  edge.arrow.size = 0.5,       # Arrow size (if directed)
  frame = TRUE,                # Draw frame around plot
  margin = c(0, 0, 0, 0)       # Margins around plot
)

# Print summary of the graph
cat("Graph Summary:\n")
print(g)
