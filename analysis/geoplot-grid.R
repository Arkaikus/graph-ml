# Load necessary libraries
library(leaflet)
library(sf)
library(geosphere)
library(mapview)
library(webshot)

# Install PhantomJS if not already installed
if (!webshot::is_phantomjs_installed()) {
  webshot::install_phantomjs()
}

# Assuming your CSV file is named "earthquake_data.csv" and is located in your working directory
original_earthquake_data <- read.csv("~/Documents/graph-ml/earthquakes/csv/9923ce9a42736848b544e335a4d7c5fb.csv")
original_earthquake_data <- original_earthquake_data[, c("latitude", "longitude", "depth", "mag", "time")]
print(paste("Number of records", nrow(original_earthquake_data)))

# Remove rows with any missing values
earthquake_data <- original_earthquake_data[complete.cases(original_earthquake_data), ]
print(paste("Number of usable records", nrow(earthquake_data)))

# Convert columns to numeric
earthquake_data$latitude <- as.numeric(as.character(earthquake_data$latitude))
earthquake_data$longitude <- as.numeric(as.character(earthquake_data$longitude))
earthquake_data$depth <- as.numeric(as.character(earthquake_data$depth))
earthquake_data$magnitude <- as.numeric(as.character(earthquake_data$mag))
earthquake_data <- earthquake_data[earthquake_data$magnitude > 0, ]

# Define function to calculate latitude and longitude sequences
calculate_sequences <- function(lat_range, lon_range, step_km) {
  lat_steps <- ceiling(distGeo(c(mean(lon_range), min(lat_range)), c(mean(lon_range), max(lat_range))) / (step_km * 1000))
  lon_steps <- ceiling(distGeo(c(min(lon_range), mean(lat_range)), c(max(lon_range), mean(lat_range))) / (step_km * 1000))
  
  lat_seq <- seq(min(lat_range), max(lat_range), length.out = lat_steps)
  lon_seq <- seq(min(lon_range), max(lon_range), length.out = lon_steps)
  
  return(list(lat_seq = lat_seq, lon_seq = lon_seq, lat_steps = lat_steps, lon_steps = lon_steps))
}

# Define function to create grid lines
create_grid_lines <- function(lat_seq, lon_seq, lat_range, lon_range) {
  lines <- list()
  
  for (lat in lat_seq) {
    lines <- c(lines, list(st_linestring(matrix(c(lon_range[1], lat, lon_range[2], lat), ncol = 2, byrow = TRUE))))
  }
  
  for (lon in lon_seq) {
    lines <- c(lines, list(st_linestring(matrix(c(lon, lat_range[1], lon, lat_range[2]), ncol = 2, byrow = TRUE))))
  }
  
  grid <- st_sfc(lines, crs = 4326)
  return(grid)
}

# Define function to create a colored square
create_colored_square <- function(lat_seq, lon_seq, x, y, color = "blue") {
  lat_min <- lat_seq[x]
  lat_max <- lat_seq[x + 1]
  lon_min <- lon_seq[y]
  lon_max <- lon_seq[y + 1]
  
  square <- st_polygon(list(matrix(c(
    lon_min, lat_min,
    lon_min, lat_max,
    lon_max, lat_max,
    lon_max, lat_min,
    lon_min, lat_min
  ), ncol = 2, byrow = TRUE)))
  
  square_sf <- st_sfc(square, crs = 4326)
  attr(square_sf, "color") <- color
  attr(square_sf, "opacity") <- 0.5
  return(square_sf)
}

# Define function to create multiple colored squares
create_colored_squares <- function(lat_seq, lon_seq, positions, color = "blue") {
  squares <- lapply(positions, function(pos) {
    create_colored_square(lat_seq, lon_seq, pos[1], pos[2], color)
  })
  return(squares)
}

# Convert latitude and longitude to index positions
convert_to_index <- function(lat, lon, lat_seq, lon_seq) {
  x <- findInterval(lat, lat_seq)
  y <- findInterval(lon, lon_seq)
  return(c(x, y))
}

# Extract a chunk of 10 rows from earthquake_data and convert to index positions
chunk <- head(earthquake_data, 1000)
positions <- mapply(function(lat, lon) convert_to_index(lat, lon, sequences$lat_seq, sequences$lon_seq), chunk$latitude, chunk$longitude, SIMPLIFY = FALSE)

# Define latitude and longitude ranges and step in kilometers
lat_range <- c(-0.132, 9.796)
lon_range <- c(-80.343, -72.466)
step_km <- 10

sequences <- calculate_sequences(lat_range, lon_range, step_km)
grid <- create_grid_lines(sequences$lat_seq, sequences$lon_seq, lat_range, lon_range)

# Create colored squares for these positions
squares <- create_colored_squares(sequences$lat_seq, sequences$lon_seq, positions, color = "blue")

# Add these squares to the map
map <- leaflet() %>%
  addTiles() %>%
  addPolylines(data = grid, color = "gray", opacity=0.2)

for (square in squares) {
  map <- map %>%
    addPolygons(data = square, color = attr(square, "color"), fillOpacity = 0.5)
}

map <- map %>%
  setView(lng = mean(lon_range), lat = mean(lat_range), zoom = 8) %>%
  addLabelOnlyMarkers(
    lng = rep(min(lon_range) - 0.2, sequences$lat_steps),
    lat = sequences$lat_seq,
    label = lapply(1:sequences$lat_steps, function(x) x),
    labelOptions = labelOptions(noHide = TRUE, direction = 'right', textOnly = TRUE)
  ) %>%
  addLabelOnlyMarkers(
    lng = sequences$lon_seq,
    lat = rep(min(lat_range) - 0.2, sequences$lon_steps),
    label = lapply(1:sequences$lon_steps, function(y) y),
    labelOptions = labelOptions(noHide = TRUE, direction = 'top', textOnly = TRUE)
  )

# Calculate padding
padding <- 0.1 # 10% padding
lat_padding <- diff(lat_range) * padding
lon_padding <- diff(lon_range) * padding

# Set the bounding box
bbox <- c(
  left = min(lon_range) - lon_padding,
  bottom = min(lat_range) - lat_padding,
  right = max(lon_range) + lon_padding,
  top = max(lat_range) + lat_padding
)

# Set the output file name
output_file <- "~/Pictures/earthquake_map.png"

# Save the map as a PNG image with padding
mapshot(map, 
        file = output_file,
        remove_url = TRUE,
        vwidth = 2000,  # Adjust these values as needed
        vheight = 2000,  # Adjust these values as needed
        # zoom = 0,       # This ensures the entire map is captured
        bbox = bbox)