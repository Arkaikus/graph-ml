# Install the required packages if not already installed
if (!require("leaflet")) install.packages("leaflet")
if (!require("dplyr")) install.packages("dplyr")

# Load the required packages
library(leaflet)
library(dplyr)

# Assuming your CSV file is named "earthquake_data.csv" and is located in your working directory
earthquake_data <- read.csv("~/Documents/graph-ml/earthquakes/csv/9923ce9a42736848b544e335a4d7c5fb.csv")

# Remove rows with any missing values
earthquake_data <- earthquake_data[complete.cases(earthquake_data), ]

# Convert columns to numeric
earthquake_data$latitude <- as.numeric(as.character(earthquake_data$latitude))
earthquake_data$longitude <- as.numeric(as.character(earthquake_data$longitude))
earthquake_data$depth <- as.numeric(as.character(earthquake_data$depth))
earthquake_data$mag <- as.numeric(as.character(earthquake_data$mag))

# Create a color palette for depth values
depth_pal <- colorNumeric(palette = "viridis", domain = earthquake_data$depth)
# mag_pal <- colorNumeric(palette = "inferno", domain = earthquake_data$mag)
# Create a leaflet map
map <- leaflet(earthquake_data) %>%
  addTiles() %>%  # Add default OpenStreetMap tile layer

  # addPolylines(
  #   lng = ~earthquake_data$longitude,
  #   lat = ~earthquake_data$latitude,
  #   color = "gray",
  #   weight = 2
  # ) %>%
  addCircles(
    lng = ~longitude,
    lat = ~latitude,
    radius = ~(depth*500),  # Adjust the scale factor as needed
    color = ~depth_pal(depth),
    stroke = FALSE,
    fillOpacity = 0.5
  ) %>%
  addLegend(
    position = "bottomright",
    pal = depth_pal,
    values = ~depth,
    title = "Depth",
    opacity = 1
  ) %>%
  # addLegend(
  #   position = "bottomright",
  #   pal = mag_pal,
  #   values = ~mag,
  #   title = "Magnitude",
  #   opacity = 1
  # ) %>%
  addScaleBar(position = "bottomleft") %>%  # Add a scale bar
  fitBounds(
    ~min(longitude), ~min(latitude),
    ~max(longitude), ~max(latitude)
  )  # Set latitude and longitude limits based on the data

# Display the map
print(map)
