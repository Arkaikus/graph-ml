# Load required libraries
library(network)
library(sna)
library(readr)
library(RColorBrewer)

# Set your working directory if needed
# setwd("path/to/your/directory")

# Read edge list from CSV file
edges <- read.csv(
  "~/Documents/graph-ml/earthquakes/csv/edges_100_c4a6eecb25e66625f702075639ef551a.csv"
)

# Create a network object
# Extract unique nodes
nodes <- unique(c(edges$source, edges$target))

# Initialize an empty network
net <- network.initialize(length(nodes), directed = FALSE)
# Set vertex names
network.vertex.names(net) <- nodes

# Add edges to the network
for (i in 1:nrow(edges)) {
  add.edge(
    net,
    tail = match(edges$source[i], nodes),
    head = match(edges$target[i], nodes)
  )
}

# Calculate the degree of each node
node_degrees <- degree(net, cmode = "outdegree") # Out-degree for directed graph

# Create a color gradient based on node degree
color_palette <- brewer.pal(9, "YlOrRd") # 9 colors from yellow to red
edge_colors <- sapply(node_degrees, function(x)
  color_palette[cut(x, breaks = 9, labels = FALSE)])

# Assign edge colors based on source node degree
edge_col_vec <- sapply(seq_len(nrow(edges)), function(i) {
  src_idx <- match(edges$source[i], nodes)
  edge_colors[src_idx]
})

# Plot the network
plot.network(
  net,
  displaylabels = TRUE,
  # Display node labels
  vertex.col = "lightblue",
  # Node color
  vertex.cex = 2,
  # Node size
  edge.col = edge_col_vec,
  # Edge color
  edge.lwd = 2,
  # Edge width
  label.pos = 5,
  # Label position
  label.cex = 1.2,
)            # Label size
