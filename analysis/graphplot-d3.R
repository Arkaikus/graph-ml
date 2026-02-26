# Install the package
if (!require("networkD3"))
  install.packages("networkD3")

# Load the library
library(networkD3)

# Read edge list from CSV file
edges <- read.csv(
  "~/Documents/graph-ml/earthquakes/csv/edges_100_c4a6eecb25e66625f702075639ef551a.csv"
)

# Create a node list
nodes <- data.frame(name = unique(c(edges$source, edges$target)))
nodes$id <- 0:(nrow(nodes) - 1)

# Merge nodes with edges to get the source and target IDs
edges <- merge(edges, nodes, by.x = "source", by.y = "name")
edges <- merge(
  edges,
  nodes,
  by.x = "target",
  by.y = "name",
  suffixes = c(".source", ".target")
)

# Create the networkD3 object
network <- simpleNetwork(edges[, c("id.source", "id.target")])

# Plot the network
simpleNetwork(
  edges[, c("id.source", "id.target")],
  Source = "id.source",
  Target = "id.target",
  height = 600,
  width = 800,
  linkDistance = 200,
  # charge = -500,
  fontSize = 14,
  linkColour = "gray",
  nodeColour = "blue",
  zoom = TRUE
)