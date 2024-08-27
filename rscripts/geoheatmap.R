# Install the required packages if not already installed
if (!require("leaflet")) install.packages("leaflet")
if (!require("dplyr")) install.packages("dplyr")

# Load the required packages
library(leaflet)
library(dplyr)

# Assuming your CSV file is named "earthquake_data.csv" and is located in your working directory
earthquake_data <- read.csv("~/Documents/graph-ml/earthquakes/csv/9923ce9a42736848b544e335a4d7c5fb.csv")

# Remove rows with any missing values
earthquake_data <- na.omit(earthquake_data)

# Convert columns to numeric
earthquake_data$latitude <- as.numeric(as.character(earthquake_data$latitude))
earthquake_data$longitude <- as.numeric(as.character(earthquake_data$longitude))
earthquake_data$depth <- as.numeric(as.character(earthquake_data$depth))
earthquake_data$mag <- as.numeric(as.character(earthquake_data$mag))

# Create a color palette for depth values
depth_pal <- colorNumeric(palette = "viridis", domain = earthquake_data$depth)
mag_pal <- colorNumeric(palette = "YlOrRd", domain = earthquake_data$mag)

# Define custom gradient colors for the heatmap
custom_gradient <- {
  "0.0" = "white"
  "0.5" = "yellow"
  "1.0" = "red"
}

# Create a leaflet map
map <- leaflet(earthquake_data) %>%
  addTiles() %>%  # Add default OpenStreetMap tile layer
  addHeatmap(
    lng = ~longitude,
    lat = ~latitude,
    intensity = ~(mag*1000),
    blur = 10,  # Adjust the blur radius
    max = 10000,   # Maximum intensity value
    radius = 30,  # Adjust the radius of each point
    gradient = custom_gradient,  # Use custom gradient colors
    layerId = "heatmap"  # Unique ID for the heatmap layer
  ) %>%
  addCircles(
    lng = ~longitude,
    lat = ~latitude,
    radius = ~(mag * 1000),  # Adjust the scale factor as needed
    color = ~mag_pal(mag),
    # stroke = FALSE,
    # fillOpacity = 1
  ) %>%
  # addPolylines(
  #   lng = ~earthquake_data$longitude,
  #   lat = ~earthquake_data$latitude,
  #   color = ~mag_pal(mag),
  #   weight = 0.5,
  #   opacity = 0.5,
  # ) %>%
  # addLegend(
  #   position = "bottomright",
  #   pal = depth_pal,
  #   values = ~depth,
  #   title = "Depth",
  #   opacity = 1
  # ) %>%
  addLegend(
    position = "bottomright",
    pal = mag_pal,
    values = ~mag,
    title = "Magnitude",
    opacity = 1
  ) %>%
  addScaleBar(position = "bottomleft") %>%  # Add a scale bar
  fitBounds(
    ~min(longitude), ~min(latitude),
    ~max(longitude), ~max(latitude)
  )  # Set latitude and longitude limits based on the data

# Display the map
print(map)
